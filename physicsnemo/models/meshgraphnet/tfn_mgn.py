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

"""Tensor Field Network MeshGraphNet (TFNMeshGraphNet).

An SE(3)-equivariant graph neural network for mesh-based physical simulation,
built on the Tensor Field Network framework using the e3nn library.

Architecture overview
---------------------
**Encoder** — a single equivariant linear map converts the separated scalar /
vector input features into a common hidden irrep space:

  ``n_node_scalar × 0e + n_node_vec × 1o  →  irreps_hidden``

**Processor** — ``processor_size`` equivariant message-passing blocks, each of
which:

  1. Computes real solid spherical harmonics ``Y_ℓ(r̂_ij)`` of the unit edge
     direction up to ``l_max`` (parity-consistent with the edge vector being a
     polar ``1o`` quantity, so ``Y_0→0e``, ``Y_1→1o``, ``Y_2→2e``, ...).
  2. Feeds the edge length through a Gaussian radial basis (``n_radial_basis``
     bumps up to ``r_max`` mm) plus any extra per-edge scalars (length and
     edge-type one-hot) into a 2-hidden-layer MLP that produces the per-edge
     weight tensor for the tensor product.
  3. Computes the equivariant message ``m_ij = TP(h_i, Y(r̂_ij), w_ij)``
     using a ``TensorProduct`` in ``"uvw"`` mode with ``shared_weights=False``
     (weights supplied by the radial MLP per edge).
  4. Aggregates ``V_j = Σ_i m_ij`` by scatter-add.
  5. Adds a self-connection ``Linear(h_j)`` and applies a nonlinearity
     (SiLU on ``l=0`` channels; identity on ``l>0`` channels, which is
     equivariant) plus LayerNorm on scalar channels.

**Decoder** — a single equivariant linear projects the hidden irreps to the
output irreps:

  ``irreps_hidden  →  n_vec_outputs × 1o  +  n_scalar_out × 0e``

The first ``n_vec_outputs × 3`` columns of the output match the vector
TARGET_KEYS (``u``, ``v``, ``a`` displacements) and are decoded equivariantly;
the remaining columns match the scalar TARGET_KEYS (``evf``, ``s``,
``cpress``).

Input feature splitting
-----------------------
The model expects the PyG graph object to carry two extra attributes set by
the dataset:

* ``graph.x_scalar``  — ``(N, n_node_scalar)`` normalised scalar features
  (``evf``, stress ``s`` components, ``cpress``, and all scalar material
  properties).
* ``graph.x_vec``     — ``(N, n_node_vec × 3)`` normalised vector features
  stored consecutively: ``u`` (3), ``v`` (3), ``a`` (3), ``mat_fiber`` (3).

Absolute coordinates (``coord``) are deliberately excluded to preserve
translational equivariance; geometric information is encoded entirely through
the edge displacement vectors.

The standard ``node_features`` argument accepted by the ``forward`` signature
is ignored — it is kept only so the model can be dropped into training loops
written for the other variants without modification.

Equivariance properties
-----------------------
* SE(3) equivariance: rotating / reflecting all positions and vector inputs
  produces the same rotation of the vector outputs.  Scalar outputs are
  exactly invariant.
* The only approximate part is the treatment of the 6-component Voigt stress
  tensor ``s`` as ``6 × 0e`` scalars.  Proper equivariance for stress would
  require decomposing it into ``0e + 2e`` irreps and redesigning the target
  normalisation, which is left as a future extension.
"""

import math
from dataclasses import dataclass
from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

try:
    from e3nn import o3
except ImportError as _e3nn_err:
    raise ImportError(
        "TFNMeshGraphNet requires the e3nn library.  "
        "Install it with:  pip install e3nn>=0.6.0"
    ) from _e3nn_err

import physicsnemo  # noqa: F401
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn.module.gnn_layers.utils import GraphType


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@dataclass
class MetaData(ModelMetaData):
    """Metadata for TFNMeshGraphNet."""

    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    onnx: bool = False
    func_torch: bool = True
    auto_grad: bool = True


# ---------------------------------------------------------------------------
# Radial basis
# ---------------------------------------------------------------------------

class _RadialBasis(nn.Module):
    """Gaussian radial basis functions with a smooth hard cutoff at ``r_max``.

    Parameters
    ----------
    n_basis : int
        Number of Gaussian basis functions.
    r_max : float
        Hard cutoff distance (mm).  Basis functions beyond ``r_max`` are
        suppressed by a smooth envelope.
    """

    def __init__(self, n_basis: int, r_max: float):
        super().__init__()
        self.n_basis = n_basis
        self.r_max = r_max
        centers = torch.linspace(0.0, r_max, n_basis)
        self.register_buffer("centers", centers)
        self.width = r_max / n_basis

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Evaluate basis functions.

        Parameters
        ----------
        r : torch.Tensor
            Edge lengths, shape ``(E,)``.

        Returns
        -------
        torch.Tensor
            Basis values, shape ``(E, n_basis)``.
        """
        rbf = torch.exp(-((r[:, None] - self.centers[None, :]) / self.width) ** 2)
        # Smooth envelope: (1 - r/r_max)^2, clamped to zero beyond r_max
        envelope = (1.0 - (r / self.r_max).clamp(max=1.0)).pow(2)
        return rbf * envelope[:, None]


# ---------------------------------------------------------------------------
# Equivariant nonlinearity
# ---------------------------------------------------------------------------

class _ScalarActivation(nn.Module):
    """Apply SiLU to ``l=0`` irrep channels; leave higher-``l`` channels unchanged.

    This is a valid equivariant nonlinearity because:

    * Scalars (``l=0``) are rotation-invariant, so any pointwise function
      applied to them is also equivariant.
    * Higher-``l`` irreps pass through unchanged (identity is trivially
      equivariant).

    Parameters
    ----------
    irreps : o3.Irreps
        The irrep spec of the tensor this activation is applied to.
    """

    def __init__(self, irreps: o3.Irreps):
        super().__init__()
        self.irreps = irreps
        slices: List = []
        offset = 0
        for mul, ir in irreps:
            end = offset + mul * ir.dim
            slices.append((offset, end, ir.l == 0))
            offset = end
        self.slices = slices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = []
        for start, end, is_scalar in self.slices:
            chunk = x[:, start:end]
            if is_scalar:
                chunk = F.silu(chunk)
            parts.append(chunk)
        return torch.cat(parts, dim=1)


# ---------------------------------------------------------------------------
# TFN message-passing layer
# ---------------------------------------------------------------------------

def _tp_instructions(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
) -> List:
    """Build ``TensorProduct`` instructions for all valid CG paths in ``uvw`` mode."""
    instr = []
    for i, (_, ir1) in enumerate(irreps_in1):
        for j, (_, ir2) in enumerate(irreps_in2):
            for k, (_, ir_out) in enumerate(irreps_out):
                if ir_out in ir1 * ir2:
                    instr.append((i, j, k, "uvw", True))
    return instr


class _TFNLayer(nn.Module):
    """One equivariant message-passing step.

    For each edge ``(i→j)``:
      1. Computes per-edge weights from the radial embedding via an MLP.
      2. Evaluates the tensor product ``TP(h_i, Y_ℓ(r̂_ij), weights)``.
      3. Scatter-adds messages to destination nodes.
      4. Adds a learnable self-connection ``Linear(h_j)``.
      5. Applies LayerNorm on the scalar channels then ``_ScalarActivation``.

    Parameters
    ----------
    irreps_hidden : o3.Irreps
        Input and output irreps of the hidden node features.
    irreps_sh : o3.Irreps
        Irreps of the pre-computed spherical harmonics.
    radial_input_dim : int
        Dimension of the per-edge scalar input to the radial MLP
        (``n_radial_basis + n_edge_extra_scalar``).
    radial_mlp_hidden : int
        Hidden dimension of the 2-layer radial MLP.
    """

    def __init__(
        self,
        irreps_hidden: o3.Irreps,
        irreps_sh: o3.Irreps,
        radial_input_dim: int,
        radial_mlp_hidden: int,
    ):
        super().__init__()

        instr = _tp_instructions(irreps_hidden, irreps_sh, irreps_hidden)
        self.tp = o3.TensorProduct(
            irreps_hidden,
            irreps_sh,
            irreps_hidden,
            instructions=instr,
            shared_weights=False,
            internal_weights=False,
        )

        # Radial MLP: edge scalars → weight vector for the tensor product
        self.radial_net = nn.Sequential(
            nn.Linear(radial_input_dim, radial_mlp_hidden),
            nn.SiLU(),
            nn.Linear(radial_mlp_hidden, radial_mlp_hidden),
            nn.SiLU(),
            nn.Linear(radial_mlp_hidden, self.tp.weight_numel),
        )

        # Equivariant self-connection
        self.linear_skip = o3.Linear(irreps_hidden, irreps_hidden)

        # Scalar LayerNorm + equivariant activation
        n_scalar = sum(mul for mul, ir in irreps_hidden if ir.l == 0)
        self.layer_norm = nn.LayerNorm(n_scalar) if n_scalar > 0 else None
        self._n_scalar = n_scalar
        self.act = _ScalarActivation(irreps_hidden)

    def forward(
        self,
        h: torch.Tensor,
        sh: torch.Tensor,
        radial: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        h : torch.Tensor
            Node features, shape ``(N, irreps_hidden.dim)``.
        sh : torch.Tensor
            Spherical harmonics per edge, shape ``(E, irreps_sh.dim)``.
        radial : torch.Tensor
            Radial embedding per edge, shape ``(E, radial_input_dim)``.
        src : torch.Tensor
            Source node indices, shape ``(E,)``.
        dst : torch.Tensor
            Destination node indices, shape ``(E,)``.
        n_nodes : int
            Total number of nodes ``N``.

        Returns
        -------
        torch.Tensor
            Updated node features, shape ``(N, irreps_hidden.dim)``.
        """
        # Per-edge TP weights from radial network
        weights = self.radial_net(radial)  # (E, weight_numel)

        # Equivariant messages
        messages = self.tp(h[src], sh, weights)  # (E, hidden_dim)

        # Scatter-add to destination nodes
        agg = torch.zeros(n_nodes, messages.shape[1], device=h.device, dtype=messages.dtype)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        # Self-connection + combine
        h_new = agg + self.linear_skip(h)

        # LayerNorm on scalar channels only (norm equivariant for l=0)
        if self.layer_norm is not None and self._n_scalar > 0:
            h_new = torch.cat(
                [self.layer_norm(h_new[:, : self._n_scalar]), h_new[:, self._n_scalar :]],
                dim=1,
            )

        return self.act(h_new)


# ---------------------------------------------------------------------------
# Full TFN model
# ---------------------------------------------------------------------------

class TFNMeshGraphNet(Module):
    r"""SE(3)-equivariant Tensor Field Network for mesh-based simulation.

    Implements the architecture described in the module docstring.  The model
    is a drop-in replacement for :class:`~physicsnemo.models.meshgraphnet.MeshGraphNet`
    and accepts the same ``forward(node_features, edge_features, graph)``
    signature, but draws its actual node inputs from ``graph.x_scalar`` and
    ``graph.x_vec`` (set by the dataset) rather than the flat ``node_features``
    tensor.

    Parameters
    ----------
    n_node_scalar : int
        Number of scalar (``0e``) node input features.  For the standard
        needle-tissue dataset: 15 (with cpress) or 14 (without).
    n_node_vec : int
        Number of 3-D vector (``1o``) node input features.  For the standard
        dataset this is 4 (``u``, ``v``, ``a``, ``mat_fiber``).
    output_dim : int
        Total output features per node (must equal ``n_vec_outputs * 3 + n_scalar_out``).
    irreps_hidden : str, optional, default="16x0e + 8x1o + 4x2e"
        Hidden irrep specification.  Increasing multiplicity improves capacity
        at the cost of quadratically more TP weights.
    l_max : int, optional, default=2
        Maximum spherical harmonic order.  ``l_max=1`` is cheaper;
        ``l_max=3`` captures more angular structure.
    n_radial_basis : int, optional, default=8
        Number of Gaussian radial basis functions.
    r_max : float, optional, default=60.0
        Radial basis cutoff in mm.  Set to the maximum expected edge length in
        the dataset.
    n_edge_extra_scalar : int, optional, default=4
        Number of extra edge scalar features appended to the radial basis
        before the radial MLP.  For the standard edge layout
        ``[rel_pos(3), edge_len(1), edge_type_onehot(3)]`` this is 4
        (``edge_len`` + ``edge_type_onehot``).
    processor_size : int, optional, default=5
        Number of TFN message-passing layers.  Due to the higher cost per
        layer compared to MeshGraphNet, smaller values are typical.
    radial_mlp_hidden : int, optional, default=64
        Hidden dimension of the per-edge radial MLP.
    n_vec_outputs : int, optional, default=3
        Number of 3-D vector outputs decoded equivariantly (e.g. 3 for u, v,
        a).  The first ``n_vec_outputs * 3`` columns of the output correspond
        to ``n_vec_outputs × 1o`` irreps; the remaining columns are scalars.
    checkpoint_layers : bool, optional, default=False
        When ``True``, wraps each processor layer in
        ``torch.utils.checkpoint.checkpoint`` during training.  This trades
        compute for memory: the per-edge TP weight tensor (shape
        ``(E, weight_numel)``) is not stored across layers for backprop and is
        instead recomputed on the backward pass.  Strongly recommended for
        large meshes (E > 100 K edges) where ``irreps_hidden`` is non-trivial.

    Forward
    -------
    node_features : torch.Tensor
        Ignored by the TFN model (kept for API compatibility).  Pass
        ``graph.x`` as usual; the model reads ``graph.x_scalar`` and
        ``graph.x_vec`` instead.
    edge_features : torch.Tensor
        Edge features of shape :math:`(N_{edges}, 7)`.  Must follow the layout
        ``[rel_pos(3), edge_len(1), edge_type_onehot(3)]``.
    graph : GraphType
        PyG Data with ``edge_index``, ``x_scalar``
        :math:`(N_{nodes}, n\_node\_scalar)`, and ``x_vec``
        :math:`(N_{nodes}, n\_node\_vec \times 3)`.

    Outputs
    -------
    torch.Tensor
        Output of shape :math:`(N_{nodes}, D_{out})`.  Columns
        ``[0 : n\_vec\_outputs * 3]`` are equivariantly decoded vector
        outputs; the remaining columns are invariant scalar outputs.

    Examples
    --------
    >>> model = TFNMeshGraphNet(n_node_scalar=15, n_node_vec=4,
    ...                         output_dim=17, processor_size=2,
    ...                         irreps_hidden="8x0e + 4x1o + 2x2e")
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> n, e = 20, 50
    >>> src = torch.randint(0, n, (e,))
    >>> dst = torch.randint(0, n, (e,))
    >>> pos = torch.randn(n, 3)
    >>> rel_pos = pos[src] - pos[dst]
    >>> edge_len = torch.linalg.norm(rel_pos, dim=-1, keepdim=True).clamp(min=1e-8)
    >>> edge_type = torch.zeros(e, 3); edge_type[:, 0] = 1.0
    >>> ef = torch.cat([rel_pos, edge_len, edge_type], dim=-1)
    >>> graph = Data(
    ...     edge_index=torch.stack([src, dst]),
    ...     x_scalar=torch.randn(n, 15),
    ...     x_vec=torch.randn(n, 12),
    ... )
    >>> out = model(graph.x_scalar, ef, graph)   # node_features arg ignored
    >>> out.shape
    torch.Size([20, 17])
    """

    def __init__(
        self,
        n_node_scalar: int,
        n_node_vec: int,
        output_dim: int,
        irreps_hidden: str = "16x0e + 8x1o + 4x2e",
        l_max: int = 2,
        n_radial_basis: int = 8,
        r_max: float = 60.0,
        n_edge_extra_scalar: int = 4,
        processor_size: int = 5,
        radial_mlp_hidden: int = 64,
        n_vec_outputs: int = 3,
        checkpoint_layers: bool = False,
    ):
        super().__init__(meta=MetaData())

        self.n_node_scalar = n_node_scalar
        self.n_node_vec = n_node_vec
        self.n_vec_outputs = n_vec_outputs
        self.output_dim = output_dim
        self.checkpoint_layers = checkpoint_layers

        n_scalar_out = output_dim - n_vec_outputs * 3
        if n_scalar_out < 0:
            raise ValueError(
                f"output_dim={output_dim} < n_vec_outputs*3={n_vec_outputs * 3}"
            )

        # Build irrep specs
        irreps_node_input = o3.Irreps(f"{n_node_scalar}x0e + {n_node_vec}x1o")
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        irreps_out = o3.Irreps(f"{n_vec_outputs}x1o + {n_scalar_out}x0e")
        irreps_sh = o3.Irreps.spherical_harmonics(l_max)

        # Spherical harmonics up to l_max
        self.sh = o3.SphericalHarmonics(
            irreps_sh, normalize=True, normalization="component"
        )

        # Radial basis
        self.rbf = _RadialBasis(n_radial_basis, r_max)
        radial_input_dim = n_radial_basis + n_edge_extra_scalar

        # Node encoder: map input irreps → hidden irreps
        self.node_encoder = o3.Linear(irreps_node_input, self.irreps_hidden)

        # TFN processor layers
        self.layers = nn.ModuleList(
            [
                _TFNLayer(
                    self.irreps_hidden,
                    irreps_sh,
                    radial_input_dim,
                    radial_mlp_hidden,
                )
                for _ in range(processor_size)
            ]
        )

        # Decoder: hidden → output irreps
        self.node_decoder = o3.Linear(self.irreps_hidden, irreps_out)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        graph: GraphType,
        **kwargs,
    ) -> torch.Tensor:
        # ---- Node features from graph attributes (not flat node_features) ----
        # graph.x_scalar: (N, n_node_scalar)
        # graph.x_vec:    (N, n_node_vec * 3)  — consecutive xyz per vector
        # Cast to float32: e3nn ops (Linear, TensorProduct, SphericalHarmonics)
        # always run in float32 regardless of AMP autocast context.
        x_scalar = graph.x_scalar.float()
        x_vec = graph.x_vec.float()
        N = x_scalar.shape[0]

        # Build equivariant input tensor: [scalars | vectors]  (matches irreps_node_input layout)
        node_in = torch.cat([x_scalar, x_vec], dim=-1)  # (N, n_scalar + n_vec*3)
        h = self.node_encoder(node_in)  # (N, irreps_hidden.dim)

        # ---- Edge preprocessing ------------------------------------------------
        rel_pos = edge_features[:, :3].float()   # (E, 3) — physical displacement
        r = edge_features[:, 3].float()          # (E,)   — edge length
        edge_extra = edge_features[:, 3:].float()  # (E, n_edge_extra_scalar) — len + type

        # Unit edge vector → spherical harmonics
        r_hat = rel_pos / r.unsqueeze(-1).clamp(min=1e-8)
        sh = self.sh(r_hat)  # (E, irreps_sh.dim)

        # Radial basis + edge scalars → input to radial MLP
        rbf = self.rbf(r)  # (E, n_radial_basis)
        radial = torch.cat([rbf, edge_extra], dim=-1)  # (E, n_rbf + n_extra)

        # ---- Message passing ---------------------------------------------------
        src, dst = graph.edge_index
        for layer in self.layers:
            if self.checkpoint_layers and self.training:
                # Recompute this layer's forward on the backward pass instead of
                # storing its activations (especially the large per-edge weight
                # tensor).  use_reentrant=False supports non-tensor args (N).
                h = gradient_checkpoint(layer, h, sh, radial, src, dst, N,
                                        use_reentrant=False)
            else:
                h = layer(h, sh, radial, src, dst, N)

        # ---- Decode ------------------------------------------------------------
        return self.node_decoder(h)  # (N, output_dim)
