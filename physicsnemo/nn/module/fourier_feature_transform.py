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

r"""Random Fourier Feature transform (Tancik et al., 2020).

Reference: "Fourier Features Let Networks Learn High Frequency Functions in
Low Dimensional Domains", Tancik et al., NeurIPS 2020.
"""

import torch
import torch.nn as nn
from jaxtyping import Float


class FourierFeatureTransform(nn.Module):
    r"""Random Fourier Feature (RFF) input encoding.

    Maps an input tensor :math:`\mathbf{x}` of shape
    :math:`(\ldots, D_\text{in})` to

    .. math::

        \gamma(\mathbf{x}) =
        \bigl[\sin(\mathbf{x}\mathbf{B}),\; \cos(\mathbf{x}\mathbf{B})\bigr]
        \in \mathbb{R}^{2 \cdot n_\text{freq}}

    where :math:`\mathbf{B} \in \mathbb{R}^{D_\text{in} \times n_\text{freq}}`
    is sampled once from :math:`\mathcal{N}(0, \sigma^2)` and fixed
    (not trained).

    This encoding encourages networks to learn high-frequency functions that
    a plain MLP would otherwise struggle to represent (spectral bias).

    Parameters
    ----------
    in_features : int
        Dimensionality of the input :math:`D_\text{in}`.
    n_frequencies : int
        Number of random frequency projections.  Output dimensionality is
        ``2 * n_frequencies``.
    scale : float, optional, default=1.0
        Standard deviation :math:`\sigma` of the Gaussian from which
        :math:`\mathbf{B}` is sampled.  Larger values emphasise higher
        frequencies; smaller values emphasise smoother functions.

    Notes
    -----
    ``B`` is registered as a non-persistent buffer so it is moved to the
    correct device with the module but is **not** included in
    ``state_dict()`` by default.  If you need reproducible checkpoint
    reloads, save ``B`` explicitly or use a fixed ``torch.manual_seed``
    before constructing the module.

    Example
    -------
    >>> import torch
    >>> from physicsnemo.nn import FourierFeatureTransform
    >>> fft = FourierFeatureTransform(in_features=19, n_frequencies=64)
    >>> x = torch.randn(100, 19)
    >>> fft(x).shape
    torch.Size([100, 128])
    """

    def __init__(
        self,
        in_features: int,
        n_frequencies: int,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        B = torch.randn(in_features, n_frequencies) * scale
        # Non-persistent: excluded from state_dict so checkpoints stay lean.
        # Persistence can be enabled by re-registering with persistent=True if
        # exact reproducibility across saves/loads is required.
        self.register_buffer("B", B, persistent=False)
        self._in_features = in_features
        self._n_frequencies = n_frequencies

    @property
    def out_features(self) -> int:
        """Output feature dimensionality: ``2 * n_frequencies``."""
        return 2 * self._n_frequencies

    def forward(
        self,
        x: Float[torch.Tensor, "... in_features"],
    ) -> Float[torch.Tensor, "... out_features"]:
        r"""Apply the random Fourier feature mapping.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(\ldots, D_\text{in})`.

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape :math:`(\ldots, 2 \cdot n_\text{freq})`.
        """
        proj = x @ self.B  # (..., n_freq)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
