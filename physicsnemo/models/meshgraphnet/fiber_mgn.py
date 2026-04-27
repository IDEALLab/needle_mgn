# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fiber-direction equivariant MeshGraphNet.

Each node carries a unit fiber-direction vector ``d_i`` (the local material
axis, e.g. collagen fiber orientation).  The model augments standard
MeshGraphNet with:

  * Two rotation-invariant edge scalars added to every edge update:
      - ``cos θ_ij = d_i · ê_ij``  (fiber–edge alignment)
      - ``cos φ_ij = d_i · d_j``   (relative fiber rotation)

  * A vector message ``α_ij * ê_ij`` that is rotation-equivariant because
    ``α_ij`` is a rotation-invariant scalar and ``ê_ij`` rotates with the
    coordinate frame.  Per-node vector features ``V_i = Σ_j α_ij ê_ij`` are
    accumulated by scatter-add.

  * An equivariant decoder for the vector outputs (displacements u, v, a):
      ``out_k = α_k V_i + β_k d_i + γ_k (V_i × d_i)``
    where ``(α_k, β_k, γ_k)`` are scalar coefficients from an MLP.  The
    scalar outputs (EVF, stress S, contact pressure) are decoded by a
    standard MLP.

Usage
-----
The model expects the input graph to have a ``fiber_dir`` attribute
(``torch.Tensor`` of shape ``(N_nodes, 3)``) containing the per-node unit
fiber directions.  In the needle-tissue dataset this is computed from
``node_props["mat_fiber"]`` normalised to unit length.

The first ``n_vec_outputs * 3`` columns of the output correspond to the
equivariantly decoded vector fields; the remaining columns are scalar
outputs from the standard MLP decoder.  This layout matches the TARGET_KEYS
ordering ``[u (3), v (3), a (3), evf (1), s (6), cpress (1)]`` when
``n_vec_outputs=3``.
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

import physicsnemo  # noqa: F401
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn import get_activation
from physicsnemo.nn.module.gnn_layers.mesh_edge_block import MeshEdgeBlock
from physicsnemo.nn.module.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from physicsnemo.nn.module.gnn_layers.utils import GraphType, aggregate_and_concat


@dataclass
class MetaData(ModelMetaData):
    """Metadata for FiberEquivariantMGN."""

    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    onnx: bool = False
    func_torch: bool = True
    auto_grad: bool = True


class _FiberEquivNodeBlock(nn.Module):
    """Node update block that incorporates the equivariant vector aggregate.

    Concatenates: [agg(efeat), h_i, ||V_i||, V_i · d_i] → MLP → residual.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of node and edge hidden features.
    aggregation : str
        Edge aggregation method (``"sum"`` or ``"mean"``).
    num_layers : int
        Number of hidden layers in the node MLP.
    activation_fn : nn.Module
        Activation function.
    norm_type : str or None
        LayerNorm type.
    """

    def __init__(
        self,
        hidden_dim: int,
        aggregation: str = "sum",
        num_layers: int = 2,
        activation_fn: nn.Module = nn.ReLU(),
        norm_type: str = "LayerNorm",
    ):
        super().__init__()
        self.aggregation = aggregation
        # Input: agg_efeat (hidden) + h_i (hidden) + ||V|| (1) + V·d (1)
        self.node_mlp = MeshGraphMLP(
            input_dim=hidden_dim * 2 + 2,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=num_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(
        self,
        efeat: torch.Tensor,
        nfeat: torch.Tensor,
        V: torch.Tensor,
        fiber_dir: torch.Tensor,
        graph: GraphType,
    ) -> torch.Tensor:
        # Aggregate scalar edge features and concat with node features → (N, 2*hidden)
        cat_feat = aggregate_and_concat(efeat, nfeat, graph, self.aggregation)
        # Append invariant summaries of the vector aggregate
        V_norm = torch.linalg.norm(V, dim=-1, keepdim=True)   # (N, 1)
        V_dot_d = (V * fiber_dir).sum(-1, keepdim=True)        # (N, 1)
        full_in = torch.cat([cat_feat, V_norm, V_dot_d], dim=-1)  # (N, 2*h+2)
        return self.node_mlp(full_in) + nfeat


class _FiberEquivProcessor(nn.Module):
    """Equivariant message-passing processor.

    Each step applies:

    1. ``MeshEdgeBlock`` to update scalar edge features.
    2. ``alpha_head`` (linear layer) to extract a scalar weight per edge.
    3. Vector messages: ``vec_msg_ij = alpha_ij * ê_ij``.
    4. Scatter-add vector messages to destination nodes: ``V_i = Σ_j vec_msg_ij``.
    5. ``_FiberEquivNodeBlock`` to update node features using scalar aggregate
       + invariant summaries of ``V_i``.

    Parameters
    ----------
    processor_size : int
        Number of message-passing steps.
    hidden_dim : int
        Dimensionality of node and edge hidden features.
    num_layers_node : int
        Hidden layers in each node MLP.
    num_layers_edge : int
        Hidden layers in each edge MLP.
    aggregation : str
        Edge aggregation method.
    activation_fn : nn.Module
        Activation function.
    norm_type : str
        LayerNorm type.
    """

    def __init__(
        self,
        processor_size: int = 15,
        hidden_dim: int = 128,
        num_layers_node: int = 2,
        num_layers_edge: int = 2,
        aggregation: str = "sum",
        activation_fn: nn.Module = nn.ReLU(),
        norm_type: str = "LayerNorm",
    ):
        super().__init__()
        self.processor_size = processor_size

        self.edge_blocks = nn.ModuleList(
            [
                MeshEdgeBlock(
                    input_dim_nodes=hidden_dim,
                    input_dim_edges=hidden_dim,
                    output_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    hidden_layers=num_layers_edge,
                    activation_fn=activation_fn,
                    norm_type=norm_type,
                )
                for _ in range(processor_size)
            ]
        )
        self.alpha_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(processor_size)]
        )
        self.node_blocks = nn.ModuleList(
            [
                _FiberEquivNodeBlock(
                    hidden_dim=hidden_dim,
                    aggregation=aggregation,
                    num_layers=num_layers_node,
                    activation_fn=activation_fn,
                    norm_type=norm_type,
                )
                for _ in range(processor_size)
            ]
        )

    def forward(
        self,
        nfeat: torch.Tensor,
        efeat: torch.Tensor,
        graph: GraphType,
        e_hat: torch.Tensor,
        fiber_dir: torch.Tensor,
    ):
        """Run all message-passing steps.

        Parameters
        ----------
        nfeat : torch.Tensor
            Node features, shape ``(N, hidden_dim)``.
        efeat : torch.Tensor
            Edge features, shape ``(E, hidden_dim)``.
        graph : GraphType
            Graph with ``edge_index`` of shape ``(2, E)``.
        e_hat : torch.Tensor
            Unit edge vectors from physical coords, shape ``(E, 3)``.
        fiber_dir : torch.Tensor
            Unit fiber directions per node, shape ``(N, 3)``.

        Returns
        -------
        nfeat : torch.Tensor
            Updated node features, ``(N, hidden_dim)``.
        V : torch.Tensor
            Final-step vector aggregate, ``(N, 3)``.
        """
        n_nodes = nfeat.shape[0]
        dst = graph.edge_index[1]  # (E,) destination node indices

        V = torch.zeros(n_nodes, 3, device=nfeat.device, dtype=nfeat.dtype)

        for edge_block, alpha_head, node_block in zip(
            self.edge_blocks, self.alpha_heads, self.node_blocks
        ):
            # 1. Scalar edge update (residual inside MeshEdgeBlock)
            efeat, _ = edge_block(efeat, nfeat, graph)

            # 2. Scalar weight → vector message
            alpha = alpha_head(efeat)          # (E, 1)
            vec_msg = alpha * e_hat            # (E, 3)

            # 3. Scatter-add to destination nodes
            V = torch.zeros(n_nodes, 3, device=efeat.device, dtype=efeat.dtype)
            dst_exp = dst.unsqueeze(-1).expand_as(vec_msg)   # (E, 3)
            V.scatter_add_(0, dst_exp, vec_msg)

            # 4. Node update using scalar aggregate + V summaries
            nfeat = node_block(efeat, nfeat, V, fiber_dir, graph)

        return nfeat, V


class FiberEquivariantMGN(Module):
    r"""Fiber-direction equivariant MeshGraphNet.

    Extends MeshGraphNet with rotational equivariance for vector outputs by
    incorporating per-node fiber direction vectors ``d_i`` (e.g. material fiber
    orientation). Invariant edge features ``cos θ_ij`` and ``cos φ_ij`` enrich
    the edge encoder; equivariant vector messages ``α_ij ê_ij`` are scattered
    to nodes and used both during message passing and in the final decoder.

    Parameters
    ----------
    input_dim_nodes : int
        Number of node input features.
    input_dim_edges : int
        Number of edge input features (before the two fiber-invariant scalars
        are appended internally).
    output_dim : int
        Total number of output features per node (vector + scalar).
    n_vec_outputs : int, optional, default=3
        Number of 3-D vector outputs decoded equivariantly (e.g. 3 for u, v, a).
        The first ``n_vec_outputs * 3`` columns of the output tensor are the
        equivariant vector predictions; the remaining ``output_dim - n_vec_outputs * 3``
        columns are scalar predictions.
    processor_size : int, optional, default=15
        Number of message-passing steps.
    mlp_activation_fn : str, optional, default="relu"
        Activation function name.
    num_layers_node_processor : int, optional, default=2
        Hidden MLP layers per node update block.
    num_layers_edge_processor : int, optional, default=2
        Hidden MLP layers per edge update block.
    hidden_dim_processor : int, optional, default=128
        Hidden feature size in the processor.
    hidden_dim_node_encoder : int, optional, default=128
        Hidden feature size in the node encoder.
    num_layers_node_encoder : int, optional, default=2
        Number of hidden layers in the node encoder.
    hidden_dim_edge_encoder : int, optional, default=128
        Hidden feature size in the edge encoder.
    num_layers_edge_encoder : int, optional, default=2
        Number of hidden layers in the edge encoder.
    hidden_dim_node_decoder : int, optional, default=128
        Hidden feature size in the decoders.
    num_layers_node_decoder : int, optional, default=2
        Number of hidden layers in the decoders.
    aggregation : str, optional, default="sum"
        Edge aggregation method (``"sum"`` or ``"mean"``).
    norm_type : str, optional, default="LayerNorm"
        Normalization type.

    Forward
    -------
    node_features : torch.Tensor
        Input node features of shape :math:`(N_{nodes}, D_{in}^{node})`.
    edge_features : torch.Tensor
        Input edge features of shape :math:`(N_{edges}, D_{in}^{edge})`.
        The first three columns must be ``rel_pos`` and the fourth must be
        ``edge_len`` (this is the standard layout produced by the dataset).
    graph : GraphType
        PyG Data object. Must carry a ``fiber_dir`` attribute of shape
        :math:`(N_{nodes}, 3)` with unit fiber direction vectors.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(N_{nodes}, D_{out})`. Columns
        ``[0 : n\_vec\_outputs*3]`` are the equivariant vector predictions;
        columns ``[n\_vec\_outputs*3 :]`` are the scalar predictions.

    Examples
    --------
    >>> model = FiberEquivariantMGN(input_dim_nodes=30, input_dim_edges=7,
    ...                             output_dim=17, n_vec_outputs=3)
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> n, e = 20, 50
    >>> graph = Data(edge_index=torch.randint(0, n, (2, e)),
    ...              fiber_dir=torch.nn.functional.normalize(
    ...                  torch.randn(n, 3), dim=-1))
    >>> nf = torch.randn(n, 30)
    >>> ef = torch.randn(e, 7)
    >>> out = model(nf, ef, graph)
    >>> out.shape
    torch.Size([20, 17])
    """

    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        output_dim: int,
        n_vec_outputs: int = 3,
        processor_size: int = 15,
        mlp_activation_fn: str = "relu",
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        hidden_dim_processor: int = 128,
        hidden_dim_node_encoder: int = 128,
        num_layers_node_encoder: int = 2,
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: int = 2,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: int = 2,
        aggregation: Literal["sum", "mean"] = "sum",
        norm_type: Literal["LayerNorm", "TELayerNorm"] = "LayerNorm",
    ):
        super().__init__(meta=MetaData())

        self.input_dim_nodes = input_dim_nodes
        self.input_dim_edges = input_dim_edges
        self.output_dim = output_dim
        self.n_vec_outputs = n_vec_outputs

        scalar_dim = output_dim - n_vec_outputs * 3
        if scalar_dim < 0:
            raise ValueError(
                f"output_dim={output_dim} is less than n_vec_outputs*3={n_vec_outputs * 3}"
            )
        self.scalar_dim = scalar_dim

        activation_fn = get_activation(mlp_activation_fn)

        # Edge encoder: base edge features + 2 fiber invariants (cos_theta, cos_phi)
        self.edge_encoder = MeshGraphMLP(
            input_dim=input_dim_edges + 2,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_edge_encoder,
            hidden_layers=num_layers_edge_encoder,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

        # Standard node encoder
        self.node_encoder = MeshGraphMLP(
            input_dim=input_dim_nodes,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_node_encoder,
            hidden_layers=num_layers_node_encoder,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

        # Equivariant processor
        self.processor = _FiberEquivProcessor(
            processor_size=processor_size,
            hidden_dim=hidden_dim_processor,
            num_layers_node=num_layers_node_processor,
            num_layers_edge=num_layers_edge_processor,
            aggregation=aggregation,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

        # Decoder inputs: [h_i (hidden), ||V|| (1), V·d (1)]
        decoder_in_dim = hidden_dim_processor + 2

        # Equivariant vector decoder: outputs n_vec_outputs * 3 scalar coefficients
        # for the local basis {V, d, V×d}
        self.vec_coef_head = MeshGraphMLP(
            input_dim=decoder_in_dim,
            output_dim=n_vec_outputs * 3,
            hidden_dim=hidden_dim_node_decoder,
            hidden_layers=num_layers_node_decoder,
            activation_fn=activation_fn,
            norm_type=None,
        )

        # Standard scalar decoder
        if scalar_dim > 0:
            self.scalar_decoder = MeshGraphMLP(
                input_dim=decoder_in_dim,
                output_dim=scalar_dim,
                hidden_dim=hidden_dim_node_decoder,
                hidden_layers=num_layers_node_decoder,
                activation_fn=activation_fn,
                norm_type=None,
            )
        else:
            self.scalar_decoder = None

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        graph: GraphType,
        **kwargs,
    ) -> torch.Tensor:
        fiber_dir = graph.fiber_dir  # (N, 3) unit fiber directions

        ei = graph.edge_index
        src, dst = ei[0], ei[1]  # noqa: F841

        # Unit edge vectors from the first 4 columns of edge_features
        rel_pos = edge_features[:, :3]                                 # (E, 3)
        edge_len = edge_features[:, 3:4].clamp(min=1e-8)              # (E, 1)
        e_hat = rel_pos / edge_len                                     # (E, 3)

        # Fiber-direction invariants per edge
        d_src = fiber_dir[src]                                         # (E, 3)
        d_dst = fiber_dir[dst]                                         # (E, 3)
        cos_theta = (d_src * e_hat).sum(-1, keepdim=True)             # (E, 1)
        cos_phi = (d_src * d_dst).sum(-1, keepdim=True)               # (E, 1)

        edge_features_aug = torch.cat(
            [edge_features, cos_theta, cos_phi], dim=-1
        )  # (E, input_dim_edges + 2)

        efeat = self.edge_encoder(edge_features_aug)
        nfeat = self.node_encoder(node_features)

        nfeat, V = self.processor(nfeat, efeat, graph, e_hat, fiber_dir)

        # --- Decoder --------------------------------------------------------
        V_norm = torch.linalg.norm(V, dim=-1, keepdim=True)           # (N, 1)
        V_dot_d = (V * fiber_dir).sum(-1, keepdim=True)               # (N, 1)
        decoder_in = torch.cat([nfeat, V_norm, V_dot_d], dim=-1)      # (N, h+2)

        # Equivariant vector outputs via local basis {V, d, V×d}
        VcrossD = torch.linalg.cross(V, fiber_dir)                    # (N, 3)
        # basis columns: V, d, VcrossD — shape (N, 3, 3)
        basis = torch.stack([V, fiber_dir, VcrossD], dim=-1)

        # Scalar coefficients: (N, n_vec_outputs * 3) → (N, n_vec_out, 3)
        vec_coefs = self.vec_coef_head(decoder_in)
        vec_coefs = vec_coefs.view(-1, self.n_vec_outputs, 3)

        # vec_out[n, k, :] = Σ_b coefs[n, k, b] * basis[n, :, b]
        vec_out = torch.einsum("nkb,nbd->nkd", vec_coefs, basis)      # (N, n_vec_out, 3)
        vec_out = vec_out.reshape(-1, self.n_vec_outputs * 3)          # (N, n_vec_out*3)

        if self.scalar_decoder is not None:
            scalar_out = self.scalar_decoder(decoder_in)               # (N, scalar_dim)
            return torch.cat([vec_out, scalar_out], dim=-1)
        return vec_out
