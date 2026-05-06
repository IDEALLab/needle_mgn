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
from typing import Literal, Union

import torch
import torch.nn as nn

import physicsnemo  # noqa: F401
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn import KolmogorovArnoldNetwork, get_activation
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
        extra_invariants_in: int = 0,
    ):
        super().__init__()
        self.aggregation = aggregation
        # Input: agg_efeat (hidden) + h_i (hidden) + ||V|| (1) + V·d (1)
        # Plus optional extra rotation-invariant scalars (e.g. ||W|| and W·d
        # when a second equivariant aggregate W is being tracked).
        self.node_mlp = MeshGraphMLP(
            input_dim=hidden_dim * 2 + 2 + extra_invariants_in,
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
        W: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        # Aggregate scalar edge features and concat with node features → (N, 2*hidden)
        cat_feat = aggregate_and_concat(efeat, nfeat, graph, self.aggregation)
        # Append invariant summaries of the vector aggregate
        V_norm = torch.linalg.norm(V, dim=-1, keepdim=True)   # (N, 1)
        V_dot_d = (V * fiber_dir).sum(-1, keepdim=True)        # (N, 1)
        full_in = torch.cat([cat_feat, V_norm, V_dot_d], dim=-1)  # (N, 2*h+2)
        if W is not None:
            W_norm = torch.linalg.norm(W, dim=-1, keepdim=True)
            W_dot_d = (W * fiber_dir).sum(-1, keepdim=True)
            full_in = torch.cat([full_in, W_norm, W_dot_d], dim=-1)
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
        extra_decoder_basis: bool = False,
    ):
        super().__init__()
        self.processor_size = processor_size
        self.extra_decoder_basis = extra_decoder_basis

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
        # Second equivariant message head producing W = Σⱼ βᵢⱼ · (d_i × ê_ij).
        # Aggregating d_i × ê_ij guarantees W ⊥ d_i, so the decoder basis
        # {V, d, V×d, W, W×d} is reliably 3-D even when V ∥ d (the bottleneck
        # diagnosed for needle nodes whose neighbours are mostly axial).
        if extra_decoder_basis:
            self.beta_heads = nn.ModuleList(
                [nn.Linear(hidden_dim, 1) for _ in range(processor_size)]
            )
        node_block_extra = 2 if extra_decoder_basis else 0
        self.node_blocks = nn.ModuleList(
            [
                _FiberEquivNodeBlock(
                    hidden_dim=hidden_dim,
                    aggregation=aggregation,
                    num_layers=num_layers_node,
                    activation_fn=activation_fn,
                    norm_type=norm_type,
                    extra_invariants_in=node_block_extra,
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
        W : torch.Tensor or None
            Final-step ``(d × ê)`` aggregate, ``(N, 3)``.  Only returned
            when ``extra_decoder_basis=True`` was passed at construction.
        """
        n_nodes = nfeat.shape[0]
        src = graph.edge_index[0]
        dst = graph.edge_index[1]  # (E,) destination node indices

        V = torch.zeros(n_nodes, 3, device=nfeat.device, dtype=nfeat.dtype)
        W = None
        d_cross_e = None
        if self.extra_decoder_basis:
            # d_i × ê_ij is the per-edge equivariant vector that's by
            # construction perpendicular to d_i.  Computed once outside
            # the loop because both factors are fixed across layers.
            d_src = fiber_dir[src]
            d_cross_e = torch.linalg.cross(d_src, e_hat, dim=-1)   # (E, 3)
            W = torch.zeros(n_nodes, 3, device=nfeat.device, dtype=nfeat.dtype)

        for layer_idx, (edge_block, alpha_head, node_block) in enumerate(
            zip(self.edge_blocks, self.alpha_heads, self.node_blocks)
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

            if self.extra_decoder_basis:
                beta = self.beta_heads[layer_idx](efeat)       # (E, 1)
                vec_msg_w = beta * d_cross_e                   # (E, 3)
                W = torch.zeros(
                    n_nodes, 3, device=efeat.device, dtype=efeat.dtype
                )
                W.scatter_add_(0, dst_exp, vec_msg_w)

            # 4. Node update using scalar aggregate + V (and optionally W)
            nfeat = node_block(efeat, nfeat, V, fiber_dir, graph, W=W)

        return nfeat, V, W


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
        extra_edge_invariants: bool = False,
        extra_decoder_basis: bool = False,
    ):
        super().__init__(meta=MetaData())

        self.input_dim_nodes = input_dim_nodes
        self.input_dim_edges = input_dim_edges
        self.output_dim = output_dim
        self.n_vec_outputs = n_vec_outputs
        self.extra_edge_invariants = extra_edge_invariants
        self.extra_decoder_basis = extra_decoder_basis

        scalar_dim = output_dim - n_vec_outputs * 3
        if scalar_dim < 0:
            raise ValueError(
                f"output_dim={output_dim} is less than n_vec_outputs*3={n_vec_outputs * 3}"
            )
        self.scalar_dim = scalar_dim

        activation_fn = get_activation(mlp_activation_fn)

        # Edge encoder: base edge features + 2 fiber invariants (cos_theta, cos_phi).
        # When extra_edge_invariants=True we additionally append:
        #   cos_theta_dst = d_j · ê_ij        (destination-side fiber alignment)
        #   bond_corr     = cos_theta · cos_theta_dst   (l=2-flavour correlator)
        #   dv_along_edge = (v_j - v_i) · ê_ij           (compression rate along edge)
        #   dv_norm       = ||v_j - v_i||                (relative speed)
        # so velocity-driven asymmetries can modulate the per-edge α weights.
        edge_extra = 6 if extra_edge_invariants else 2
        self.edge_encoder = MeshGraphMLP(
            input_dim=input_dim_edges + edge_extra,
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
            extra_decoder_basis=extra_decoder_basis,
        )

        # Decoder inputs: [h_i (hidden), ||V|| (1), V·d (1)]
        # plus optional [||W|| (1), W·d (1)] when the W aggregate is tracked.
        decoder_in_dim = hidden_dim_processor + 2
        if extra_decoder_basis:
            decoder_in_dim += 2

        # Number of basis vectors used to express each equivariant output:
        #   3 = {V, d, V × d}                                 (default)
        #   5 = {V, d, V × d, W, W × d}                       (extra_decoder_basis)
        # The W basis vectors are guaranteed perpendicular to d, so the basis
        # spans 3-D even when V ∥ d (which happens for needle nodes whose
        # neighbours are mostly axial — the bottleneck causing x-z flattening).
        self.n_basis = 5 if extra_decoder_basis else 3

        # Equivariant vector decoder: outputs n_vec_outputs * n_basis scalar
        # coefficients for the local equivariant basis.
        self.vec_coef_head = MeshGraphMLP(
            input_dim=decoder_in_dim,
            output_dim=n_vec_outputs * self.n_basis,
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
        **kwargs,  # noqa: ARG002 — kept for API compatibility with sibling models
    ) -> torch.Tensor:
        fiber_dir = graph.fiber_dir  # (N, 3) unit fiber directions

        ei = graph.edge_index
        src, dst = ei[0], ei[1]

        # Unit edge vectors from the first 4 columns of edge_features
        rel_pos = edge_features[:, :3]                                 # (E, 3)
        edge_len = edge_features[:, 3:4].clamp(min=1e-8)              # (E, 1)
        e_hat = rel_pos / edge_len                                     # (E, 3)

        # Fiber-direction invariants per edge
        d_src = fiber_dir[src]                                         # (E, 3)
        d_dst = fiber_dir[dst]                                         # (E, 3)
        cos_theta = (d_src * e_hat).sum(-1, keepdim=True)             # (E, 1)
        cos_phi = (d_src * d_dst).sum(-1, keepdim=True)               # (E, 1)

        if self.extra_edge_invariants:
            # Symmetric counterpart of cos_theta plus an l=2-flavour
            # alignment correlator.
            cos_theta_dst = (d_dst * e_hat).sum(-1, keepdim=True)
            bond_corr = cos_theta * cos_theta_dst
            # Velocity-driven invariants — break the y-symmetry that locks
            # alpha to fiber-only signals.  Requires graph.node_velocity
            # (already supplied by the dataset / infer pipeline).
            v_node = graph.node_velocity                               # (N, 3)
            v_src = v_node[src]
            v_dst = v_node[dst]
            dv = v_dst - v_src
            dv_along_edge = (dv * e_hat).sum(-1, keepdim=True)
            dv_norm = torch.linalg.norm(dv, dim=-1, keepdim=True)
            edge_features_aug = torch.cat(
                [
                    edge_features,
                    cos_theta, cos_phi,
                    cos_theta_dst, bond_corr,
                    dv_along_edge, dv_norm,
                ],
                dim=-1,
            )  # (E, input_dim_edges + 6)
        else:
            edge_features_aug = torch.cat(
                [edge_features, cos_theta, cos_phi], dim=-1
            )  # (E, input_dim_edges + 2)

        efeat = self.edge_encoder(edge_features_aug)
        nfeat = self.node_encoder(node_features)

        nfeat, V, W = self.processor(nfeat, efeat, graph, e_hat, fiber_dir)

        # --- Decoder --------------------------------------------------------
        V_norm = torch.linalg.norm(V, dim=-1, keepdim=True)           # (N, 1)
        V_dot_d = (V * fiber_dir).sum(-1, keepdim=True)               # (N, 1)
        decoder_in = torch.cat([nfeat, V_norm, V_dot_d], dim=-1)      # (N, h+2)
        if self.extra_decoder_basis:
            W_norm = torch.linalg.norm(W, dim=-1, keepdim=True)
            W_dot_d = (W * fiber_dir).sum(-1, keepdim=True)
            decoder_in = torch.cat([decoder_in, W_norm, W_dot_d], dim=-1)

        # Equivariant vector outputs via local basis {V, d, V×d} or
        # {V, d, V×d, W, W×d} when extra_decoder_basis is set.
        VcrossD = torch.linalg.cross(V, fiber_dir, dim=-1)            # (N, 3)
        basis_vecs = [V, fiber_dir, VcrossD]
        if self.extra_decoder_basis:
            WcrossD = torch.linalg.cross(W, fiber_dir, dim=-1)
            basis_vecs.extend([W, WcrossD])
        basis = torch.stack(basis_vecs, dim=-1)                       # (N, 3, n_basis)

        # Scalar coefficients: (N, n_vec_outputs * n_basis) → (N, n_vec_out, n_basis)
        vec_coefs = self.vec_coef_head(decoder_in)
        vec_coefs = vec_coefs.view(-1, self.n_vec_outputs, self.n_basis)

        # vec_out[n, k, x] = Σ_b coefs[n, k, b] * basis[n, x, b]
        # i.e. each output is a learned linear combination of the basis
        # vectors {V, d, V×d} (and {W, W×d} when extra_decoder_basis is set).
        # The contraction is over basis_idx (length n_basis) — NOT xyz —
        # so the output transforms as a proper 1o vector under rotation
        # of the basis vectors.  The previous "nkb,nbd->nkd" form happened
        # to typecheck only because n_basis=3=xyz and was equivalent to a
        # different (rotation-non-equivariant) computation.
        vec_out = torch.einsum("nkb,nxb->nkx", vec_coefs, basis)      # (N, n_vec_out, 3)
        vec_out = vec_out.reshape(-1, self.n_vec_outputs * 3)          # (N, n_vec_out*3)

        if self.scalar_decoder is not None:
            scalar_out = self.scalar_decoder(decoder_in)               # (N, scalar_dim)
            return torch.cat([vec_out, scalar_out], dim=-1)
        return vec_out


class FiberEquivariantKAN(FiberEquivariantMGN):
    r"""Fiber-direction equivariant MeshGraphNet with a KAN node encoder.

    Identical to :class:`FiberEquivariantMGN` except the MLP node encoder is
    replaced by a Kolmogorov–Arnold Network (KAN), following the same pattern
    as :class:`~physicsnemo.models.meshgraphnet.MeshGraphKAN`.

    The KAN encoder maps ``input_dim_nodes`` → ``hidden_dim_processor`` using
    Fourier-series basis functions parameterised by ``num_harmonics``.

    Parameters
    ----------
    input_dim_nodes : int
        Number of node features.
    input_dim_edges : int
        Number of edge features.
    output_dim : int
        Total number of output features (vector + scalar).
    n_vec_outputs : int, optional, default=3
        Number of 3-D vector outputs decoded equivariantly.
    num_harmonics : int, optional, default=5
        Number of Fourier harmonics for the KAN node encoder.
    processor_size : int, optional, default=15
        Number of message-passing steps.
    hidden_dim_processor : int, optional, default=128
        Hidden feature size throughout the model.
    aggregation : str, optional, default="sum"
        Edge aggregation method.

    All other parameters are forwarded to :class:`FiberEquivariantMGN`.
    """

    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        output_dim: int,
        n_vec_outputs: int = 3,
        num_harmonics: int = 5,
        processor_size: int = 15,
        mlp_activation_fn: str = "relu",
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        hidden_dim_processor: int = 128,
        hidden_dim_node_encoder: int = 128,
        num_layers_node_encoder: Union[int, None] = 2,  # ignored for KAN
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: int = 2,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: int = 2,
        aggregation: Literal["sum", "mean"] = "sum",
        norm_type: Literal["LayerNorm", "TELayerNorm"] = "LayerNorm",
        extra_edge_invariants: bool = False,
        extra_decoder_basis: bool = False,
    ):
        super().__init__(
            input_dim_nodes=input_dim_nodes,
            input_dim_edges=input_dim_edges,
            output_dim=output_dim,
            n_vec_outputs=n_vec_outputs,
            processor_size=processor_size,
            mlp_activation_fn=mlp_activation_fn,
            num_layers_node_processor=num_layers_node_processor,
            num_layers_edge_processor=num_layers_edge_processor,
            hidden_dim_processor=hidden_dim_processor,
            hidden_dim_node_encoder=hidden_dim_node_encoder,
            num_layers_node_encoder=num_layers_node_encoder,
            hidden_dim_edge_encoder=hidden_dim_edge_encoder,
            num_layers_edge_encoder=num_layers_edge_encoder,
            hidden_dim_node_decoder=hidden_dim_node_decoder,
            num_layers_node_decoder=num_layers_node_decoder,
            aggregation=aggregation,
            norm_type=norm_type,
            extra_edge_invariants=extra_edge_invariants,
            extra_decoder_basis=extra_decoder_basis,
        )

        # Replace the MLP node encoder with a KAN.
        self.node_encoder = KolmogorovArnoldNetwork(
            input_dim=input_dim_nodes,
            output_dim=hidden_dim_processor,
            num_harmonics=num_harmonics,
            add_bias=True,
        )
