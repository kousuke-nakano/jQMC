"""Jastrow module.

Precision Zones:
    - ``jastrow``: forward Jastrow evaluation (compute_Jastrow_part, J1/J2/J3).
    - ``kinetic``: Jastrow derivatives (compute_grads_and_laplacian_Jastrow_*).
    - ``mcmc``: ratio and update functions.

See :mod:`jqmc._precision` for details.
"""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the jqmc project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# python modules
import itertools
from collections.abc import Callable

# set logger
from logging import getLogger

# jqmc module
from typing import TYPE_CHECKING, Any, Sequence

# jax modules
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import linen as nn
from flax import struct
from jax import grad, hessian, jit, vmap
from jax import typing as jnpt
from jax.tree_util import tree_flatten, tree_unflatten

from ._precision import get_dtype_jnp, get_dtype_np
from ._setting import EPS_safe_distance, atol_consistency
from .atomic_orbital import (
    AOs_cart_data,
    AOs_sphe_data,
    ShellPrimMap,
    _aos_cart_to_sphe,
    _aos_sphe_to_cart,
    compute_AOs,
    compute_AOs_grad,
    compute_AOs_laplacian,
    compute_AOs_value_grad_lap,
)
from .molecular_orbital import (
    MOs_data,
    compute_MOs,
    compute_MOs_grad,
    compute_MOs_laplacian,
    compute_MOs_value_grad_lap,
)
from .structure import Structure_data

if TYPE_CHECKING:  # typing-only import to avoid circular dependency
    from .wavefunction import VariationalParameterBlock

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


def _ensure_flax_trace_level_compat() -> None:
    """Safely handle missing ``flax.core.tracers.trace_level`` attribute.

    Some Flax versions expose ``trace_level``, others do not. When absent, we
    simply no-op to avoid AttributeError during NN Jastrow initialization.
    """
    try:
        from flax.core import tracers as flax_tracers  # type: ignore
    except Exception:
        return

    trace_level = getattr(flax_tracers, "trace_level", None)
    if trace_level is None:
        return
    if getattr(trace_level, "_jqmc_patched", False):
        return

    # Mark as patched to prevent repeated checks; do not mutate further when the
    # attribute exists but already works.
    setattr(trace_level, "_jqmc_patched", True)


def _flatten_params_with_treedef(params: Any) -> tuple[jnp.ndarray, Any, list[tuple[int, ...]]]:
    """Flatten a PyTree of params into a 1D vector, returning treedef and shapes.

    This helper is defined at module scope so that closures built from it
    are picklable (needed for storing NN_Jastrow_data inside
    Hamiltonian_data via pickle).
    """
    leaves, treedef = tree_flatten(params)
    flat = jnp.concatenate([jnp.ravel(x) for x in leaves])
    shapes: list[tuple[int, ...]] = [tuple(x.shape) for x in leaves]
    return flat, treedef, shapes


def _make_flatten_fn(treedef: Any) -> Callable[[Any], jnp.ndarray]:
    """Create a flatten function based on a reference treedef.

    The resulting function flattens any params PyTree that matches the
    same treedef structure into a 1D JAX array.
    """

    def flatten_fn(p: Any) -> jnp.ndarray:
        leaves_p, treedef_p = tree_flatten(p)
        # Optional: could assert treedef_p == treedef for extra safety.
        return jnp.concatenate([jnp.ravel(x) for x in leaves_p])

    # Expose the treedef used to build this function so that pickle can
    # correctly restore it as a top-level function reference rather than a
    # local closure. This makes the object picklable when NN_Jastrow_data
    # instances are stored inside Hamiltonian_data.
    flatten_fn.__module__ = __name__
    flatten_fn.__qualname__ = "_make_flatten_fn_flatten_fn"

    return flatten_fn


def _make_unflatten_fn(treedef: Any, shapes: Sequence[tuple[int, ...]]) -> Callable[[jnp.ndarray], Any]:
    """Create an unflatten function using a treedef and per-leaf shapes."""

    def unflatten_fn(flat_vec: jnp.ndarray) -> Any:
        leaves_new = []
        idx = 0
        for shape in shapes:
            size = int(np.prod(shape))
            leaves_new.append(flat_vec[idx : idx + size].reshape(shape))
            idx += size
        return tree_unflatten(treedef, leaves_new)

    # As with _make_flatten_fn, make sure this nested function is picklable by
    # giving it a stable module and qualname so that pickle can resolve it as
    # a top-level attribute.
    unflatten_fn.__module__ = __name__
    unflatten_fn.__qualname__ = "_make_unflatten_fn_unflatten_fn"

    return unflatten_fn


def _ensure_flax_trace_level_compat() -> None:
    """Patch Flax trace-level helper for newer JAX EvalTrace objects.

    Some JAX versions return EvalTrace objects without a ``level`` attribute,
    which older Flax releases assume exists. This patch makes the lookup safe.
    """
    try:
        from flax.core import tracers as flax_tracers
    except Exception:
        return

    trace_level = getattr(flax_tracers, "trace_level", None)
    if trace_level is None:
        return
    if getattr(trace_level, "_jqmc_patched", False):
        return

    def _trace_level_safe(main):
        if main is None:
            return float("-inf")
        return getattr(main, "level", float("-inf"))

    _trace_level_safe._jqmc_patched = True
    flax_tracers.trace_level = _trace_level_safe


class NNJastrow(nn.Module):
    r"""PauliNet-inspired NN that outputs a three-body Jastrow correction.

    The network implements the iteration rules described in the PauliNet
    manuscript (Eq. 1–2). Electron embeddings :math:`\mathbf{x}_i^{(n)}` are
    iteratively refined by three message channels:

    * ``(+ )``: same-spin electrons, enforcing antisymmetry indirectly by keeping
        the messages exchange-equivariant.
    * ``(- )``: opposite-spin electrons, capturing pairing terms.
    * ``(n)``: nuclei, represented by fixed species embeddings.

    After ``num_layers`` iterations the final electron embeddings are summed and
    fed through :math:`\eta_\theta` to produce a symmetric correction that is
    added on top of the analytic three-body Jastrow.
    """

    hidden_dim: int = 64
    num_layers: int = 3
    num_rbf: int = 32
    cutoff: float = 5.0
    species_lookup: npt.NDArray[np.int32] | jnp.ndarray | tuple[int, ...] | None = None
    num_species: int | None = None

    class PhysNetRadialLayer(nn.Module):
        r"""Cuspless PhysNet-inspired radial features :math:`e_k(r)`.

        The basis follows Eq. (3) in the PauliNet supplement with a PhysNet-style
        envelope that forces both the value and the derivative of each Gaussian
        to vanish at the cutoff and the origin.  These features are reused across
        all message channels, ensuring consistent geometric encoding.
        """

        num_rbf: int
        cutoff: float

        @nn.compact
        def __call__(self, distances: jnp.ndarray) -> jnp.ndarray:
            r"""Evaluate the PhysNet radial envelope :math:`e_k(r)`.

            The basis functions follow PauliNet's implementation Eq. (12)
            [Nat. Chem. 12, 891-897 (2020)] (https://doi.org/10.1038/s41557-020-0544-y):

            .. math::

                e_k(r) = r^2 \exp\left[-r - \frac{(r-\mu_k)^2}{\sigma_k^2}\right]

            where :math:`\mu_k` and :math:`\sigma_k` are fixed hyperparameters distributed up to :math:`r_c`.
            Note that unlike PhysNet, this PauliNet implementation does not enforce a hard spatial cutoff
            on the basis functions themselves, relying instead on natural decay.

            Args:
                distances: Array of shape ``(...,)`` containing non-negative inter-particle
                    distances in Bohr. Arbitrary batch dimensions are supported.

            Returns:
                jnp.ndarray: ``distances.shape + (num_rbf,)`` radial feature tensor.

            Raises:
                ValueError: If ``num_rbf`` is not strictly positive.
            """
            if self.num_rbf <= 0:
                raise ValueError("num_rbf must be positive for PhysNet radial features.")

            q = jnp.linspace(0.0, 1.0, self.num_rbf + 2, dtype=distances.dtype)[1:-1]
            mu = self.cutoff * q**2
            sigma = (1.0 / 7.0) * (1.0 + self.cutoff * q)

            d = distances[..., None]
            mu = mu[None, ...]
            sigma = sigma[None, ...]

            features = (d**2) * jnp.exp(-d - ((d - mu) ** 2) / (sigma**2 + EPS_safe_distance))
            return features

    class TwoLayerMLP(nn.Module):
        r"""Utility MLP used for :math:`w_\theta`, :math:`h_\theta`, :math:`g_\theta`, and :math:`\eta_\theta`."""

        width: int
        out_dim: int

        @nn.compact
        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            """Apply a SiLU-activated two-layer perceptron.

            Args:
                x: Input tensor of shape ``(..., features)`` whose trailing axis is interpreted
                    as the feature dimension.

            Returns:
                jnp.ndarray: Tensor with the same leading dimensions as ``x`` and a trailing
                dimension of ``out_dim``.
            """
            y = nn.Dense(self.width)(x)
            y = nn.silu(y)
            y = nn.Dense(self.out_dim)(y)
            return y

    class PauliNetBlock(nn.Module):
        r"""Single PauliNet message-passing iteration following Eq. (1).

        Each block mixes three message channels per electron: same-spin ``(+ )``,
        opposite-spin ``(- )``, and nucleus-electron ``(n)``. The sender network
        is shared across channels to match the PauliNet weight-tying scheme, while
        separate weighting/receiver networks parameterize the contribution of every
        channel.
        """

        hidden_dim: int

        def setup(self):
            """Instantiate the shared sender/receiver networks for this block."""
            self.sender_net = NNJastrow.TwoLayerMLP(width=self.hidden_dim, out_dim=self.hidden_dim)
            self.weight_same = NNJastrow.TwoLayerMLP(width=self.hidden_dim, out_dim=self.hidden_dim)
            self.weight_opposite = NNJastrow.TwoLayerMLP(width=self.hidden_dim, out_dim=self.hidden_dim)
            self.weight_nuc = NNJastrow.TwoLayerMLP(width=self.hidden_dim, out_dim=self.hidden_dim)

            self.receiver_same = NNJastrow.TwoLayerMLP(width=self.hidden_dim, out_dim=self.hidden_dim)
            self.receiver_opposite = NNJastrow.TwoLayerMLP(width=self.hidden_dim, out_dim=self.hidden_dim)
            self.receiver_nuc = NNJastrow.TwoLayerMLP(width=self.hidden_dim, out_dim=self.hidden_dim)

        def _aggregate_pair_channel(
            self,
            weights_net: nn.Module,
            radial_features: jnp.ndarray,
            sender_proj: jnp.ndarray,
            mask: jnp.ndarray | None = None,
        ) -> jnp.ndarray:
            """Aggregate electron-electron messages for a given spin sector.

            Args:
                weights_net: Channel-specific MLP producing pair weights of shape
                    ``(n_i, n_j, hidden_dim)`` from PhysNet features.
                radial_features: Output of ``PhysNetRadialLayer`` for the considered
                    electron pair distances.
                sender_proj: Projected sender embeddings (``n_j, hidden_dim``).
                mask: Optional ``(n_i, n_j)`` mask that zeroes self-interactions in the
                    same-spin channel.

            Returns:
                jnp.ndarray: Aggregated messages of shape ``(n_i, hidden_dim)``.
            """
            weights = weights_net(radial_features)
            if mask is not None:
                weights = weights * mask[..., None]
            messages = weights * sender_proj[None, :, :]
            return jnp.sum(messages, axis=1)

        def _aggregate_nuclear_channel(
            self,
            weights_net: nn.Module,
            radial_features: jnp.ndarray,
            nuclear_embeddings: jnp.ndarray,
        ) -> jnp.ndarray:
            """Aggregate messages coming from the fixed nuclear embeddings.

            Args:
                weights_net: MLP that maps electron-nucleus PhysNet features to weights.
                radial_features: Electron-nucleus features with shape ``(n_e, n_nuc, hidden_dim)``.
                nuclear_embeddings: Learned species embeddings ``(n_nuc, hidden_dim)``.

            Returns:
                jnp.ndarray: ``(n_e, hidden_dim)`` messages summarizing nuclear influence.
            """
            if nuclear_embeddings.shape[0] == 0:
                return jnp.zeros((radial_features.shape[0], self.hidden_dim), dtype=radial_features.dtype)
            weights = weights_net(radial_features)
            messages = weights * nuclear_embeddings[None, :, :]
            return jnp.sum(messages, axis=1)

        def __call__(
            self,
            x_up: jnp.ndarray,
            x_dn: jnp.ndarray,
            feat_up_up: jnp.ndarray,
            feat_up_dn: jnp.ndarray,
            feat_dn_dn: jnp.ndarray,
            feat_up_nuc: jnp.ndarray,
            feat_dn_nuc: jnp.ndarray,
            nuclear_embeddings: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            r"""Apply Eq. (1) to update spin-resolved embeddings.

            Args:
                x_up: ``(n_up, hidden_dim)`` features for :math:`\alpha` electrons.
                x_dn: ``(n_dn, hidden_dim)`` features for :math:`\beta` electrons.
                feat_*: PhysNet feature tensors for every pair/channel computed outside the block.
                nuclear_embeddings: ``(n_nuc, hidden_dim)`` lookup embeddings per species.

            Returns:
                Tuple[jnp.ndarray, jnp.ndarray]: Updated ``(n_up, hidden_dim)`` and
                ``(n_dn, hidden_dim)`` embeddings to be fed into the next block.
            """
            n_up = x_up.shape[0]
            sender_proj = self.sender_net(jnp.concatenate([x_up, x_dn], axis=0))
            sender_up = sender_proj[:n_up]
            sender_dn = sender_proj[n_up:]

            mask_up = 1.0 - jnp.eye(feat_up_up.shape[0], dtype=feat_up_up.dtype)
            mask_dn = 1.0 - jnp.eye(feat_dn_dn.shape[0], dtype=feat_dn_dn.dtype)

            z_same_up = self._aggregate_pair_channel(self.weight_same, feat_up_up, sender_up, mask_up)
            z_same_dn = self._aggregate_pair_channel(self.weight_same, feat_dn_dn, sender_dn, mask_dn)

            z_op_up = self._aggregate_pair_channel(self.weight_opposite, feat_up_dn, sender_dn)
            z_op_dn = self._aggregate_pair_channel(self.weight_opposite, jnp.swapaxes(feat_up_dn, 0, 1), sender_up)

            z_nuc_up = self._aggregate_nuclear_channel(self.weight_nuc, feat_up_nuc, nuclear_embeddings)
            z_nuc_dn = self._aggregate_nuclear_channel(self.weight_nuc, feat_dn_nuc, nuclear_embeddings)

            delta_up = self.receiver_same(z_same_up) + self.receiver_opposite(z_op_up) + self.receiver_nuc(z_nuc_up)
            delta_dn = self.receiver_same(z_same_dn) + self.receiver_opposite(z_op_dn) + self.receiver_nuc(z_nuc_dn)

            return x_up + delta_up, x_dn + delta_dn

    def setup(self):
        """Instantiate PauliNet components and validate required metadata.

        Raises:
            ValueError: If ``species_lookup`` or ``num_species`` were not provided via
                the host dataclass before module initialization.
        """
        if self.species_lookup is None or self.num_species is None:
            raise ValueError("NNJastrow requires species_lookup and num_species to be set before initialization.")
        self.featurizer = NNJastrow.PhysNetRadialLayer(num_rbf=self.num_rbf, cutoff=self.cutoff)
        self.blocks = tuple(NNJastrow.PauliNetBlock(hidden_dim=self.hidden_dim) for _ in range(self.num_layers))
        self.spin_embedding = nn.Embed(num_embeddings=2, features=self.hidden_dim)
        self.init_env_net = NNJastrow.TwoLayerMLP(width=self.hidden_dim, out_dim=self.hidden_dim)
        self.readout_net = NNJastrow.TwoLayerMLP(width=self.hidden_dim, out_dim=1)
        self.nuclear_species_embedding = nn.Embed(num_embeddings=self.num_species, features=self.hidden_dim)

    def _pairwise_distances(self, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        """Compute pairwise Euclidean distances with numerical stabilization.

        Args:
            A: ``(n_a, 3)`` Cartesian coordinates.
            B: ``(n_b, 3)`` Cartesian coordinates.

        Returns:
            jnp.ndarray: ``(n_a, n_b)`` matrix with a small epsilon added before the square
            root to keep gradients finite when particles coincide.
        """
        dtype_jnp = get_dtype_jnp("jastrow_eval")
        if A.shape[0] == 0 or B.shape[0] == 0:
            return jnp.zeros((A.shape[0], B.shape[0]), dtype=dtype_jnp)
        # Reconstruct differences in caller-supplied precision (fp64 from MCMC
        # walker state) via JAX promotion when one operand is fp64, then downcast
        # to the jastrow_eval zone. Avoids catastrophic cancellation without
        # hardcoding fp64.
        diff = (A[:, None, :] - B[None, :, :]).astype(dtype_jnp)
        return jnp.sqrt(jnp.sum(diff**2, axis=-1) + EPS_safe_distance)

    def _nuclear_embeddings(self, Z_n: jnp.ndarray) -> jnp.ndarray:
        """Convert atomic numbers into learned embedding vectors.

        Args:
            Z_n: Integer array of atomic numbers with shape ``(n_nuc,)``.

        Returns:
            jnp.ndarray: ``(n_nuc, hidden_dim)`` embeddings looked up through
            ``species_lookup``. Returns an empty array when no nuclei are present.
        """
        dtype_jnp = get_dtype_jnp("jastrow_eval")
        n_nuc = Z_n.shape[0]
        if n_nuc == 0:
            return jnp.zeros((0, self.hidden_dim), dtype=dtype_jnp)

        lookup = jnp.asarray(self.species_lookup)
        species_ids = jnp.take(lookup, Z_n.astype(jnp.int32), mode="clip")
        return self.nuclear_species_embedding(species_ids)

    def _initial_electron_features(
        self,
        n_up: int,
        n_dn: int,
        feat_up_nuc: jnp.ndarray,
        feat_dn_nuc: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""Form the iteration-0 embeddings incorporating spin and nuclei.

        Args:
            n_up: Number of spin-up electrons.
            n_dn: Number of spin-down electrons.
            feat_up_nuc: PhysNet features ``(n_up, n_nuc, num_rbf)``.
            feat_dn_nuc: PhysNet features ``(n_dn, n_nuc, num_rbf)``.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Spin-conditioned embeddings that already
            include the ``h_\theta`` initialization term from PauliNet.
        """
        spin_ids = jnp.concatenate([jnp.zeros((n_up,), dtype=jnp.int32), jnp.ones((n_dn,), dtype=jnp.int32)], axis=0)
        spin_embed = self.spin_embedding(spin_ids)
        x_up = spin_embed[:n_up]
        x_dn = spin_embed[n_up:]

        if feat_up_nuc.size:
            x_up = x_up + jnp.sum(self.init_env_net(feat_up_nuc), axis=1)
        if feat_dn_nuc.size:
            x_dn = x_dn + jnp.sum(self.init_env_net(feat_dn_nuc), axis=1)

        return x_up, x_dn

    def __call__(
        self,
        r_up: jnp.ndarray,
        r_dn: jnp.ndarray,
        R_n: jnp.ndarray,
        Z_n: jnp.ndarray,
    ) -> jnp.ndarray:
        r"""Evaluate :math:`J_\text{NN}` in Eq. (2) for the provided configuration.

        Args:
            r_up: ``(n_up, 3)`` spin-up electron coordinates in Bohr.
            r_dn: ``(n_dn, 3)`` spin-down electron coordinates in Bohr.
            R_n: ``(n_nuc, 3)`` nuclear positions.
            Z_n: ``(n_nuc,)`` atomic numbers matching ``R_n``.

        Returns:
            jnp.ndarray: Scalar NN-corrected three-body Jastrow contribution.

        Notes:
            The network is permutation equivariant within each spin channel and rotation
            invariant by construction of the PhysNet radial features.
        """
        # Forward r_up/r_dn/R_n as-is (Principle 3a — no parameter rebind).
        # `_pairwise_distances` reconstructs the differences in caller-supplied
        # precision and downcasts to the jastrow_eval zone at the use site.
        Z_n = jnp.asarray(Z_n)

        n_up = r_up.shape[0]
        n_dn = r_dn.shape[0]

        feat_up_up = self.featurizer(self._pairwise_distances(r_up, r_up))
        feat_dn_dn = self.featurizer(self._pairwise_distances(r_dn, r_dn))
        feat_up_dn = self.featurizer(self._pairwise_distances(r_up, r_dn))
        feat_up_nuc = self.featurizer(self._pairwise_distances(r_up, R_n))
        feat_dn_nuc = self.featurizer(self._pairwise_distances(r_dn, R_n))

        nuclear_embeddings = self._nuclear_embeddings(Z_n)
        x_up, x_dn = self._initial_electron_features(n_up, n_dn, feat_up_nuc, feat_dn_nuc)

        for block in self.blocks:
            x_up, x_dn = block(
                x_up,
                x_dn,
                feat_up_up,
                feat_up_dn,
                feat_dn_dn,
                feat_up_nuc,
                feat_dn_nuc,
                nuclear_embeddings,
            )

        x_final = jnp.concatenate([x_up, x_dn], axis=0)
        j_vals = self.readout_net(x_final)
        j_val = jnp.sum(j_vals)
        return j_val


@struct.dataclass
class Jastrow_one_body_data:
    r"""One-body Jastrow parameters and structure metadata.

    The one-body term models electron–nucleus correlations.  Two functional
    forms are available, selected by ``jastrow_1b_type``:

    * ``'exp'`` (default) — exponential form:

      .. math::

         f(r_{eN}) = -A \, \frac{1}{2a} \bigl(1 - e^{-a\,c\,r_{eN}}\bigr)

    * ``'pade'`` — Padé form:

      .. math::

         f(r_{eN}) = -A \, \frac{r_{eN}}{2\,(1 + a\,c\,r_{eN})}

    where :math:`A = (2 Z_{\text{eff}})^{3/4}`, :math:`c = (2 Z_{\text{eff}})^{1/4}`,
    and :math:`a` is ``jastrow_1b_param``.

    The numerical value is returned without the ``exp`` wrapper; callers
    attach ``exp(J)`` to the wavefunction.

    Args:
        jastrow_1b_param (float): Parameter *a* controlling the one-body decay.
        jastrow_1b_type (str): Functional form — ``'exp'`` or ``'pade'``.
            Stored as a compile-time constant (``pytree_node=False``).
        structure_data (Structure_data): Nuclear positions and charges.
        core_electrons (tuple[float]): Removed core electrons per nucleus (for ECPs).
    """

    jastrow_1b_param: float = struct.field(pytree_node=True, default=1.0)  #: One-body Jastrow exponent parameter.
    jastrow_1b_type: str = struct.field(pytree_node=False, default="exp")  #: Functional form: ``'exp'`` or ``'pade'``.
    structure_data: Structure_data = struct.field(
        pytree_node=True, default_factory=Structure_data
    )  #: Nuclear structure data providing positions and atomic numbers.
    core_electrons: list[float] | tuple[float] = struct.field(
        pytree_node=False, default_factory=tuple
    )  #: Effective core-electron counts aligned with ``structure_data``.

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        if self.jastrow_1b_param < 0.0:
            raise ValueError(f"jastrow_1b_param = {self.jastrow_1b_param} must be non-negative.")
        if self.jastrow_1b_type not in ("exp", "pade"):
            raise ValueError(f"jastrow_1b_type = '{self.jastrow_1b_type}' must be 'exp' or 'pade'.")
        if len(self.core_electrons) != len(self.structure_data.positions):
            raise ValueError(
                f"len(core_electrons) = {len(self.core_electrons)} must be the same as len(structure_data.positions) = {len(self.structure_data.positions)}."
            )
        if not isinstance(self.core_electrons, (list, tuple)):
            raise ValueError(f"core_electrons = {type(self.core_electrons)} must be a list or tuple.")
        self.structure_data.sanity_check()

    def _get_info(self) -> list[str]:
        """Return a list of strings representing the logged information."""
        info_lines = []
        info_lines.append("**" + self.__class__.__name__)
        info_lines.append(f"  Jastrow 1b param = {self.jastrow_1b_param}")
        info_lines.append(f"  1b Jastrow functional form is the {self.jastrow_1b_type} type.")
        return info_lines

    def _logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self._get_info():
            logger.info(line)

    @classmethod
    def init_jastrow_one_body_data(cls, jastrow_1b_param, structure_data, core_electrons, jastrow_1b_type="exp"):
        """Initialization."""
        dtype_np = get_dtype_np("jastrow_eval")
        jastrow_one_body_data = cls(
            jastrow_1b_param=np.asarray(jastrow_1b_param, dtype=dtype_np).reshape(()),
            jastrow_1b_type=jastrow_1b_type,
            structure_data=structure_data,
            core_electrons=core_electrons,
        )
        return jastrow_one_body_data


@jit
def compute_Jastrow_one_body(
    jastrow_one_body_data: Jastrow_one_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float:
    r"""Evaluate the one-body Jastrow :math:`J_1` (without ``exp``) for given coordinates.

    The original exponential form and usage remain unchanged: this routine
    returns the scalar ``J`` value; callers attach ``exp(J)`` to the wavefunction.

    Args:
        jastrow_one_body_data: One-body Jastrow parameters and structure data.
        r_up_carts: Spin-up electron coordinates with shape ``(N_up, 3)``.
        r_dn_carts: Spin-down electron coordinates with shape ``(N_dn, 3)``.

    Returns:
        float: One-body Jastrow value (before exponentiation).
    """
    # NOTE: do not pre-cast r_*_carts. ``one_body_jastrow_kernel`` reconstructs
    # ``r - R`` in float64 internally to avoid catastrophic cancellation;
    # a wrapper-level downcast would defeat that guard.
    dtype_jnp = get_dtype_jnp("jastrow_eval")
    # Retrieve structure data and convert to JAX arrays
    R_carts = jastrow_one_body_data.structure_data._positions_cart_jnp.astype(dtype_jnp)
    atomic_numbers = jnp.array(jastrow_one_body_data.structure_data.atomic_numbers, dtype=dtype_jnp)
    core_electrons = jnp.array(jastrow_one_body_data.core_electrons, dtype=dtype_jnp)
    effective_charges = atomic_numbers - core_electrons
    j1b = jnp.asarray(jastrow_one_body_data.jastrow_1b_param, dtype=dtype_jnp)

    j1b_type = jastrow_one_body_data.jastrow_1b_type

    if j1b_type == "exp":

        def one_body_jastrow_kernel(
            param: float,
            coeff: float,
            r_cart: jnpt.ArrayLike,
            R_cart: jnpt.ArrayLike,
        ) -> float:
            """Exponential form of J1."""
            # Reconstruct r - R in caller-supplied precision (fp64 from MCMC walker
            # state) via JAX promotion when one operand is fp64, then downcast to
            # the jastrow_eval zone. Avoids catastrophic cancellation without
            # hardcoding fp64.
            diff = (r_cart - R_cart).astype(dtype_jnp)
            return 1.0 / (2.0 * param) * (1.0 - jnp.exp(-param * coeff * jnp.linalg.norm(diff)))

        def atom_contrib(r_cart, R_cart, Z_eff):
            coeff = (2.0 * Z_eff) ** (1.0 / 4.0)
            return -((2.0 * Z_eff) ** (3.0 / 4.0)) * one_body_jastrow_kernel(j1b, coeff, r_cart, R_cart)

    elif j1b_type == "pade":

        def atom_contrib(r_cart, R_cart, Z_eff):
            """Pade form of J1: -Z_eff^{3/4} * r_eN / (2*(1 + a * Z_eff^{1/4} * r_eN))."""
            # Reconstruct r - R in caller-supplied precision (fp64 from MCMC walker
            # state) via JAX promotion when one operand is fp64, then downcast to
            # the jastrow_eval zone. Avoids catastrophic cancellation without
            # hardcoding fp64.
            diff = (r_cart - R_cart).astype(dtype_jnp)
            r_eN = jnp.linalg.norm(diff)
            coeff = (2.0 * Z_eff) ** (1.0 / 4.0)
            return -((2.0 * Z_eff) ** (3.0 / 4.0)) * r_eN / (2.0 * (1.0 + j1b * coeff * r_eN))

    else:
        raise ValueError(f"Unknown jastrow_1b_type: {j1b_type}")

    # Sum the contributions from all atoms for a single electron
    def electron_contrib(r_cart, R_carts, effective_charges):
        # Apply vmap over positions and effective_charges
        return jnp.sum(jax.vmap(atom_contrib, in_axes=(None, 0, 0))(r_cart, R_carts, effective_charges))

    # Sum contributions for all spin-up electrons
    J1_up = jnp.sum(jax.vmap(electron_contrib, in_axes=(0, None, None))(r_up_carts, R_carts, effective_charges))
    # Sum contributions for all spin-down electrons
    J1_dn = jnp.sum(jax.vmap(electron_contrib, in_axes=(0, None, None))(r_dn_carts, R_carts, effective_charges))

    return J1_up + J1_dn


def _compute_Jastrow_one_body_debug(
    jastrow_one_body_data: Jastrow_one_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """See compute_Jastrow_one_body_api."""
    dtype_np = get_dtype_np("jastrow_eval")
    positions = jastrow_one_body_data.structure_data.positions
    atomic_numbers = jastrow_one_body_data.structure_data.atomic_numbers
    core_electrons = jastrow_one_body_data.core_electrons
    effective_charges = np.array(atomic_numbers, dtype=dtype_np) - np.array(core_electrons, dtype=dtype_np)

    j1b_type = jastrow_one_body_data.jastrow_1b_type

    def one_body_jastrow_exp(
        param: float, coeff: float, r_cart: npt.NDArray[np.float64], R_cart: npt.NDArray[np.float64]
    ) -> float:
        """Exponential form of J1."""
        return 1.0 / (2.0 * param) * (1.0 - np.exp(-param * coeff * np.linalg.norm(r_cart - R_cart)))

    def one_body_jastrow_pade(
        param: float, coeff: float, r_cart: npt.NDArray[np.float64], R_cart: npt.NDArray[np.float64]
    ) -> float:
        """Pade form of J1."""
        r_eN = np.linalg.norm(r_cart - R_cart)
        return r_eN / (2.0 * (1.0 + param * coeff * r_eN))

    if j1b_type == "exp":
        _j1_kernel = one_body_jastrow_exp
    elif j1b_type == "pade":
        _j1_kernel = one_body_jastrow_pade
    else:
        raise ValueError(f"Unknown jastrow_1b_type: {j1b_type}")

    J1_up = 0.0
    for r_up in r_up_carts:
        for R_cart, Z_eff in zip(positions, effective_charges, strict=True):
            coeff = (2.0 * Z_eff) ** (1.0 / 4.0)
            J1_up += -((2.0 * Z_eff) ** (3.0 / 4.0)) * _j1_kernel(jastrow_one_body_data.jastrow_1b_param, coeff, r_up, R_cart)

    J1_dn = 0.0
    for r_up in r_dn_carts:
        for R_cart, Z_eff in zip(positions, effective_charges, strict=True):
            coeff = (2.0 * Z_eff) ** (1.0 / 4.0)
            J1_dn += -((2.0 * Z_eff) ** (3.0 / 4.0)) * _j1_kernel(jastrow_one_body_data.jastrow_1b_param, coeff, r_up, R_cart)

    J1 = J1_up + J1_dn

    return J1


def _compute_grads_and_laplacian_Jastrow_one_body_debug(
    jastrow_one_body_data: Jastrow_one_body_data,
    r_up_carts: np.ndarray,
    r_dn_carts: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Numerical gradients and Laplacian for one-body Jastrow (debug)."""
    dtype_np = get_dtype_np("jastrow_grad_lap")
    diff_h = 1.0e-5
    r_up_carts = np.array(r_up_carts, dtype=dtype_np)
    r_dn_carts = np.array(r_dn_carts, dtype=dtype_np)

    # grad up
    grad_x_up = []
    grad_y_up = []
    grad_z_up = []
    for r_i, _ in enumerate(r_up_carts):
        diff_p_x_r_up_carts = r_up_carts.copy()
        diff_p_y_r_up_carts = r_up_carts.copy()
        diff_p_z_r_up_carts = r_up_carts.copy()
        diff_p_x_r_up_carts[r_i][0] += diff_h
        diff_p_y_r_up_carts[r_i][1] += diff_h
        diff_p_z_r_up_carts[r_i][2] += diff_h

        J_p_x_up = _compute_Jastrow_one_body_debug(jastrow_one_body_data, diff_p_x_r_up_carts, r_dn_carts)
        J_p_y_up = _compute_Jastrow_one_body_debug(jastrow_one_body_data, diff_p_y_r_up_carts, r_dn_carts)
        J_p_z_up = _compute_Jastrow_one_body_debug(jastrow_one_body_data, diff_p_z_r_up_carts, r_dn_carts)

        diff_m_x_r_up_carts = r_up_carts.copy()
        diff_m_y_r_up_carts = r_up_carts.copy()
        diff_m_z_r_up_carts = r_up_carts.copy()
        diff_m_x_r_up_carts[r_i][0] -= diff_h
        diff_m_y_r_up_carts[r_i][1] -= diff_h
        diff_m_z_r_up_carts[r_i][2] -= diff_h

        J_m_x_up = _compute_Jastrow_one_body_debug(jastrow_one_body_data, diff_m_x_r_up_carts, r_dn_carts)
        J_m_y_up = _compute_Jastrow_one_body_debug(jastrow_one_body_data, diff_m_y_r_up_carts, r_dn_carts)
        J_m_z_up = _compute_Jastrow_one_body_debug(jastrow_one_body_data, diff_m_z_r_up_carts, r_dn_carts)

        grad_x_up.append((J_p_x_up - J_m_x_up) / (2.0 * diff_h))
        grad_y_up.append((J_p_y_up - J_m_y_up) / (2.0 * diff_h))
        grad_z_up.append((J_p_z_up - J_m_z_up) / (2.0 * diff_h))

    # grad dn
    grad_x_dn = []
    grad_y_dn = []
    grad_z_dn = []
    for r_i, _ in enumerate(r_dn_carts):
        diff_p_x_r_dn_carts = r_dn_carts.copy()
        diff_p_y_r_dn_carts = r_dn_carts.copy()
        diff_p_z_r_dn_carts = r_dn_carts.copy()
        diff_p_x_r_dn_carts[r_i][0] += diff_h
        diff_p_y_r_dn_carts[r_i][1] += diff_h
        diff_p_z_r_dn_carts[r_i][2] += diff_h

        J_p_x_dn = _compute_Jastrow_one_body_debug(jastrow_one_body_data, r_up_carts, diff_p_x_r_dn_carts)
        J_p_y_dn = _compute_Jastrow_one_body_debug(jastrow_one_body_data, r_up_carts, diff_p_y_r_dn_carts)
        J_p_z_dn = _compute_Jastrow_one_body_debug(jastrow_one_body_data, r_up_carts, diff_p_z_r_dn_carts)

        diff_m_x_r_dn_carts = r_dn_carts.copy()
        diff_m_y_r_dn_carts = r_dn_carts.copy()
        diff_m_z_r_dn_carts = r_dn_carts.copy()
        diff_m_x_r_dn_carts[r_i][0] -= diff_h
        diff_m_y_r_dn_carts[r_i][1] -= diff_h
        diff_m_z_r_dn_carts[r_i][2] -= diff_h

        J_m_x_dn = _compute_Jastrow_one_body_debug(jastrow_one_body_data, r_up_carts, diff_m_x_r_dn_carts)
        J_m_y_dn = _compute_Jastrow_one_body_debug(jastrow_one_body_data, r_up_carts, diff_m_y_r_dn_carts)
        J_m_z_dn = _compute_Jastrow_one_body_debug(jastrow_one_body_data, r_up_carts, diff_m_z_r_dn_carts)

        grad_x_dn.append((J_p_x_dn - J_m_x_dn) / (2.0 * diff_h))
        grad_y_dn.append((J_p_y_dn - J_m_y_dn) / (2.0 * diff_h))
        grad_z_dn.append((J_p_z_dn - J_m_z_dn) / (2.0 * diff_h))

    grad_J1_up = np.array([grad_x_up, grad_y_up, grad_z_up], dtype=dtype_np).T
    grad_J1_dn = np.array([grad_x_dn, grad_y_dn, grad_z_dn], dtype=dtype_np).T

    # laplacian
    diff_h2 = 1.0e-3
    J_ref = _compute_Jastrow_one_body_debug(jastrow_one_body_data, r_up_carts, r_dn_carts)

    lap_J1_up = np.zeros(len(r_up_carts), dtype=dtype_np)

    # laplacians up
    for r_i, _ in enumerate(r_up_carts):
        diff_p_x_r_up2_carts = r_up_carts.copy()
        diff_p_y_r_up2_carts = r_up_carts.copy()
        diff_p_z_r_up2_carts = r_up_carts.copy()
        diff_p_x_r_up2_carts[r_i][0] += diff_h2
        diff_p_y_r_up2_carts[r_i][1] += diff_h2
        diff_p_z_r_up2_carts[r_i][2] += diff_h2

        J_p_x_up2 = _compute_Jastrow_one_body_debug(jastrow_one_body_data, diff_p_x_r_up2_carts, r_dn_carts)
        J_p_y_up2 = _compute_Jastrow_one_body_debug(jastrow_one_body_data, diff_p_y_r_up2_carts, r_dn_carts)
        J_p_z_up2 = _compute_Jastrow_one_body_debug(jastrow_one_body_data, diff_p_z_r_up2_carts, r_dn_carts)

        diff_m_x_r_up2_carts = r_up_carts.copy()
        diff_m_y_r_up2_carts = r_up_carts.copy()
        diff_m_z_r_up2_carts = r_up_carts.copy()
        diff_m_x_r_up2_carts[r_i][0] -= diff_h2
        diff_m_y_r_up2_carts[r_i][1] -= diff_h2
        diff_m_z_r_up2_carts[r_i][2] -= diff_h2

        J_m_x_up2 = _compute_Jastrow_one_body_debug(jastrow_one_body_data, diff_m_x_r_up2_carts, r_dn_carts)
        J_m_y_up2 = _compute_Jastrow_one_body_debug(jastrow_one_body_data, diff_m_y_r_up2_carts, r_dn_carts)
        J_m_z_up2 = _compute_Jastrow_one_body_debug(jastrow_one_body_data, diff_m_z_r_up2_carts, r_dn_carts)

        gradgrad_x_up = (J_p_x_up2 + J_m_x_up2 - 2 * J_ref) / (diff_h2**2)
        gradgrad_y_up = (J_p_y_up2 + J_m_y_up2 - 2 * J_ref) / (diff_h2**2)
        gradgrad_z_up = (J_p_z_up2 + J_m_z_up2 - 2 * J_ref) / (diff_h2**2)

        lap_J1_up[r_i] = gradgrad_x_up + gradgrad_y_up + gradgrad_z_up

    lap_J1_dn = np.zeros(len(r_dn_carts), dtype=dtype_np)

    # laplacians dn
    for r_i, _ in enumerate(r_dn_carts):
        diff_p_x_r_dn2_carts = r_dn_carts.copy()
        diff_p_y_r_dn2_carts = r_dn_carts.copy()
        diff_p_z_r_dn2_carts = r_dn_carts.copy()
        diff_p_x_r_dn2_carts[r_i][0] += diff_h2
        diff_p_y_r_dn2_carts[r_i][1] += diff_h2
        diff_p_z_r_dn2_carts[r_i][2] += diff_h2

        J_p_x_dn2 = _compute_Jastrow_one_body_debug(jastrow_one_body_data, r_up_carts, diff_p_x_r_dn2_carts)
        J_p_y_dn2 = _compute_Jastrow_one_body_debug(jastrow_one_body_data, r_up_carts, diff_p_y_r_dn2_carts)
        J_p_z_dn2 = _compute_Jastrow_one_body_debug(jastrow_one_body_data, r_up_carts, diff_p_z_r_dn2_carts)

        diff_m_x_r_dn2_carts = r_dn_carts.copy()
        diff_m_y_r_dn2_carts = r_dn_carts.copy()
        diff_m_z_r_dn2_carts = r_dn_carts.copy()
        diff_m_x_r_dn2_carts[r_i][0] -= diff_h2
        diff_m_y_r_dn2_carts[r_i][1] -= diff_h2
        diff_m_z_r_dn2_carts[r_i][2] -= diff_h2

        J_m_x_dn2 = _compute_Jastrow_one_body_debug(jastrow_one_body_data, r_up_carts, diff_m_x_r_dn2_carts)
        J_m_y_dn2 = _compute_Jastrow_one_body_debug(jastrow_one_body_data, r_up_carts, diff_m_y_r_dn2_carts)
        J_m_z_dn2 = _compute_Jastrow_one_body_debug(jastrow_one_body_data, r_up_carts, diff_m_z_r_dn2_carts)

        gradgrad_x_dn = (J_p_x_dn2 + J_m_x_dn2 - 2 * J_ref) / (diff_h2**2)
        gradgrad_y_dn = (J_p_y_dn2 + J_m_y_dn2 - 2 * J_ref) / (diff_h2**2)
        gradgrad_z_dn = (J_p_z_dn2 + J_m_z_dn2 - 2 * J_ref) / (diff_h2**2)

        lap_J1_dn[r_i] = gradgrad_x_dn + gradgrad_y_dn + gradgrad_z_dn

    return grad_J1_up, grad_J1_dn, lap_J1_up, lap_J1_dn


@jit
def _compute_grads_and_laplacian_Jastrow_one_body_auto(
    jastrow_one_body_data: Jastrow_one_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Auto-diff gradients and Laplacian for one-body Jastrow."""
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    r_up_carts = jnp.array(r_up_carts, dtype=dtype_jnp)
    r_dn_carts = jnp.array(r_dn_carts, dtype=dtype_jnp)

    grad_J1_up = grad(compute_Jastrow_one_body, argnums=1)(jastrow_one_body_data, r_up_carts, r_dn_carts)
    grad_J1_dn = grad(compute_Jastrow_one_body, argnums=2)(jastrow_one_body_data, r_up_carts, r_dn_carts)

    hessian_J1_up = hessian(compute_Jastrow_one_body, argnums=1)(jastrow_one_body_data, r_up_carts, r_dn_carts)
    laplacian_J1_up = jnp.einsum("ijij->i", hessian_J1_up)

    hessian_J1_dn = hessian(compute_Jastrow_one_body, argnums=2)(jastrow_one_body_data, r_up_carts, r_dn_carts)
    laplacian_J1_dn = jnp.einsum("ijij->i", hessian_J1_dn)

    return grad_J1_up, grad_J1_dn, laplacian_J1_up, laplacian_J1_dn


@jit
def compute_grads_and_laplacian_Jastrow_one_body(
    jastrow_one_body_data: Jastrow_one_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Analytic gradients and per-electron Laplacians for the one-body Jastrow.

    Args:
        jastrow_one_body_data: One-body Jastrow parameters and structure data.
        r_up_carts: Spin-up electron coordinates with shape ``(N_up, 3)``.
        r_dn_carts: Spin-down electron coordinates with shape ``(N_dn, 3)``.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            Gradients for up/down electrons with shapes ``(N_up, 3)`` and ``(N_dn, 3)``,
            Laplacians for up/down electrons with shapes ``(N_up,)`` and ``(N_dn,)``.
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    positions = jastrow_one_body_data.structure_data._positions_cart_jnp.astype(dtype_jnp)
    atomic_numbers = jnp.asarray(jastrow_one_body_data.structure_data.atomic_numbers, dtype=dtype_jnp)
    core_electrons = jnp.asarray(jastrow_one_body_data.core_electrons, dtype=dtype_jnp)
    z_eff = atomic_numbers - core_electrons

    a = jastrow_one_body_data.jastrow_1b_param
    c = (2.0 * z_eff) ** (1.0 / 4.0)
    A = (2.0 * z_eff) ** (3.0 / 4.0)

    eps = EPS_safe_distance

    j1b_type = jastrow_one_body_data.jastrow_1b_type

    if j1b_type == "exp":

        def _grad_lap_one_spin(r_carts):
            diff = r_carts[:, None, :] - positions[None, :, :]
            r = jnp.linalg.norm(diff, axis=-1)
            r_safe = jnp.maximum(r, eps)
            exp_term = jnp.exp(-a * c[None, :] * r_safe)

            fprime = -A[None, :] * (c[None, :] / 2.0) * exp_term
            grad = jnp.sum((fprime[..., None] * diff) / r_safe[..., None], axis=1)

            fsecond = A[None, :] * (a * c[None, :] * c[None, :] / 2.0) * exp_term
            lap = fsecond - A[None, :] * c[None, :] * exp_term / r_safe
            lap_e = jnp.sum(lap, axis=1)
            return grad, lap_e

    elif j1b_type == "pade":

        def _grad_lap_one_spin(r_carts):
            # f(r) = -A * r / (2*(1 + a*c*r))
            # f'(r) = -A / (2*(1+a*c*r)^2)
            # f''(r) = A*a*c / ((1+a*c*r)^3)
            # lap contribution per atom: f''(r) + 2*f'(r)/r
            diff = r_carts[:, None, :] - positions[None, :, :]  # (n_e, n_nuc, 3)
            r = jnp.linalg.norm(diff, axis=-1)  # (n_e, n_nuc)
            r_safe = jnp.maximum(r, eps)
            denom = 1.0 + a * c[None, :] * r_safe  # (n_e, n_nuc)

            fprime = -A[None, :] / (2.0 * denom * denom)  # f'(r)
            grad = jnp.sum((fprime[..., None] * diff) / r_safe[..., None], axis=1)  # (n_e, 3)

            fsecond = A[None, :] * a * c[None, :] / (denom * denom * denom)  # f''(r)
            lap = fsecond + 2.0 * fprime / r_safe
            lap_e = jnp.sum(lap, axis=1)
            return grad, lap_e

    else:
        raise ValueError(f"Unknown jastrow_1b_type: {j1b_type}")

    grad_up, lap_up = _grad_lap_one_spin(jnp.asarray(r_up_carts, dtype=dtype_jnp))
    grad_dn, lap_dn = _grad_lap_one_spin(jnp.asarray(r_dn_carts, dtype=dtype_jnp))

    return grad_up, grad_dn, lap_up, lap_dn


# ---------------------------------------------------------------------------
# J1 streaming state (PR3).
#
# J1 is a per-electron sum over atoms with no inter-electron coupling, so a
# single-electron move only affects one row of the per-electron grad/lap.
# The state simply caches those rows; advance recomputes the moved row in
# ``O(n_atom)`` rather than the fresh O(N_e * n_atom).
# ---------------------------------------------------------------------------


@struct.dataclass
class Jastrow_one_body_streaming_state:
    """Cached per-electron J1 grad/lap consistent with current ``(r_up, r_dn)``."""

    grad_J1_up: jax.Array  # (N_up, 3)
    grad_J1_dn: jax.Array  # (N_dn, 3)
    lap_J1_up: jax.Array  # (N_up,)
    lap_J1_dn: jax.Array  # (N_dn,)


@jit
def _init_grads_laplacian_Jastrow_one_body_streaming_state(
    jastrow_one_body_data: Jastrow_one_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> Jastrow_one_body_streaming_state:
    """One-shot init equivalent in cost to ``compute_grads_and_laplacian_Jastrow_one_body``."""
    g_up, g_dn, l_up, l_dn = compute_grads_and_laplacian_Jastrow_one_body(jastrow_one_body_data, r_up_carts, r_dn_carts)
    return Jastrow_one_body_streaming_state(grad_J1_up=g_up, grad_J1_dn=g_dn, lap_J1_up=l_up, lap_J1_dn=l_dn)


@jit
def _advance_grads_laplacian_Jastrow_one_body_streaming_state(
    jastrow_one_body_data: Jastrow_one_body_data,
    state: Jastrow_one_body_streaming_state,
    moved_spin_is_up: jax.Array,
    moved_index: jax.Array,
    r_up_carts_new: jax.Array,
    r_dn_carts_new: jax.Array,
) -> Jastrow_one_body_streaming_state:
    """Advance J1 state after a single-electron move.

    Recomputes one electron's row by re-running the existing one-spin kernel
    on a single-row slice. Cost: ``O(n_atom)``.
    """
    num_up = state.grad_J1_up.shape[0]
    num_dn = state.grad_J1_dn.shape[0]

    def _branch_up(_):
        # Reuse the full-batch kernel on a length-1 slice — gives one row that
        # we slot back into the cached state.
        r_slice = jnp.expand_dims(r_up_carts_new[moved_index], axis=0)  # (1, 3)
        g_row, _, l_row, _ = compute_grads_and_laplacian_Jastrow_one_body(jastrow_one_body_data, r_slice, r_slice[:0])
        new_grad = state.grad_J1_up.at[moved_index].set(g_row[0])
        new_lap = state.lap_J1_up.at[moved_index].set(l_row[0])
        return state.replace(grad_J1_up=new_grad, lap_J1_up=new_lap)

    def _branch_dn(_):
        r_slice = jnp.expand_dims(r_dn_carts_new[moved_index], axis=0)
        # Pass empty up so only the dn branch contributes (J1 is per-spin
        # independent — feeding empty up has zero effect on dn output).
        _, g_row, _, l_row = compute_grads_and_laplacian_Jastrow_one_body(jastrow_one_body_data, r_slice[:0], r_slice)
        new_grad = state.grad_J1_dn.at[moved_index].set(g_row[0])
        new_lap = state.lap_J1_dn.at[moved_index].set(l_row[0])
        return state.replace(grad_J1_dn=new_grad, lap_J1_dn=new_lap)

    if num_up == 0:
        return _branch_dn(None)
    if num_dn == 0:
        return _branch_up(None)
    return jax.lax.cond(moved_spin_is_up, _branch_up, _branch_dn, operand=None)


@struct.dataclass
class Jastrow_two_body_data:
    r"""Two-body Jastrow parameter container.

    The two-body term models electron–electron correlations.  Two functional
    forms are available, selected by ``jastrow_2b_type``:

    * ``'pade'`` (default) — Padé form:

      .. math::

         f(r_{ee}) = \frac{r_{ee}}{2\,(1 + a\,r_{ee})}

    * ``'exp'`` — exponential form:

      .. math::

         f(r_{ee}) = \frac{1}{2a}\bigl(1 - e^{-a\,r_{ee}}\bigr)

    where :math:`a` is ``jastrow_2b_param``.

    Values are returned without exponentiation; callers use ``exp(J)``
    when constructing the wavefunction.

    Args:
        jastrow_2b_param (float): Parameter *a* for the two-body Jastrow part.
        jastrow_2b_type (str): Functional form — ``'pade'`` or ``'exp'``.
            Stored as a compile-time constant (``pytree_node=False``).
    """

    jastrow_2b_param: float = struct.field(pytree_node=True, default=1.0)  #: Two-body Jastrow parameter.
    jastrow_2b_type: str = struct.field(pytree_node=False, default="pade")  #: Functional form: ``'pade'`` or ``'exp'``.

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        if self.jastrow_2b_param < 0.0:
            raise ValueError(f"jastrow_2b_param = {self.jastrow_2b_param} must be non-negative.")
        if self.jastrow_2b_type not in ("exp", "pade"):
            raise ValueError(f"jastrow_2b_type = '{self.jastrow_2b_type}' must be 'exp' or 'pade'.")

    def _get_info(self) -> list[str]:
        """Return a list of strings representing the logged information."""
        info_lines = []
        info_lines.append("**" + self.__class__.__name__)
        info_lines.append(f"  Jastrow 2b param = {self.jastrow_2b_param}")
        info_lines.append(f"  2b Jastrow functional form is the {self.jastrow_2b_type} type.")
        return info_lines

    def _logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self._get_info():
            logger.info(line)

    @classmethod
    def init_jastrow_two_body_data(cls, jastrow_2b_param=1.0, jastrow_2b_type="pade"):
        """Initialization."""
        dtype_np = get_dtype_np("jastrow_eval")
        jastrow_two_body_data = cls(
            jastrow_2b_param=np.asarray(jastrow_2b_param, dtype=dtype_np).reshape(()),
            jastrow_2b_type=jastrow_2b_type,
        )
        return jastrow_two_body_data


@jit
def compute_Jastrow_two_body(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float:
    r"""Evaluate the two-body Jastrow :math:`J_2` (Pade form) without exponentiation.

    The functional form and usage remain identical to the original docstring;
    this returns ``J`` and callers attach ``exp(J)`` to the wavefunction.

    Args:
        jastrow_two_body_data: Two-body Jastrow parameter container.
        r_up_carts: Spin-up electron coordinates with shape ``(N_up, 3)``.
        r_dn_carts: Spin-down electron coordinates with shape ``(N_dn, 3)``.

    Returns:
        float: Two-body Jastrow value (before exponentiation).
    """
    # NOTE: do not pre-cast r_*_carts. ``two_body_jastrow_pade``/``_exp``
    # reconstruct ``r_i - r_j`` in float64 internally to avoid catastrophic
    # cancellation; a wrapper-level downcast would defeat that guard.
    dtype_jnp = get_dtype_jnp("jastrow_eval")
    j2b_param = jnp.asarray(jastrow_two_body_data.jastrow_2b_param, dtype=dtype_jnp)
    j2b_type = jastrow_two_body_data.jastrow_2b_type

    def two_body_jastrow_exp(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> float:
        """Exponential form of J2."""
        # Reconstruct r_i - r_j in caller-supplied precision (fp64 from MCMC walker
        # state) via JAX promotion when one operand is fp64, then downcast to the
        # jastrow_eval zone. Avoids catastrophic cancellation without hardcoding fp64.
        diff = (r_cart_i - r_cart_j).astype(dtype_jnp)
        return 1.0 / (2.0 * param) * (1.0 - jnp.exp(-param * jnp.linalg.norm(diff)))

    def two_body_jastrow_pade(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> float:
        """Pade form of J2."""
        # Reconstruct r_i - r_j in caller-supplied precision (fp64 from MCMC walker
        # state) via JAX promotion when one operand is fp64, then downcast to the
        # jastrow_eval zone. Avoids catastrophic cancellation without hardcoding fp64.
        diff = (r_cart_i - r_cart_j).astype(dtype_jnp)
        r_ij = jnp.linalg.norm(diff)
        return r_ij / 2.0 * (1.0 + param * r_ij) ** (-1.0)

    if j2b_type == "pade":
        two_body_jastrow_anti_parallel = two_body_jastrow_pade
        two_body_jastrow_parallel = two_body_jastrow_pade
    elif j2b_type == "exp":
        two_body_jastrow_anti_parallel = two_body_jastrow_exp
        two_body_jastrow_parallel = two_body_jastrow_exp
    else:
        raise ValueError(f"Unknown jastrow_2b_type: {j2b_type}")

    vmap_two_body_jastrow_anti_parallel_spins = vmap(
        vmap(two_body_jastrow_anti_parallel, in_axes=(None, None, 0)), in_axes=(None, 0, None)
    )

    two_body_jastrow_anti_parallel_val = jnp.sum(vmap_two_body_jastrow_anti_parallel_spins(j2b_param, r_up_carts, r_dn_carts))

    def compute_parallel_sum(r_carts):
        num_particles = r_carts.shape[0]
        idx_i, idx_j = jnp.triu_indices(num_particles, k=1)
        r_i = r_carts[idx_i]
        r_j = r_carts[idx_j]
        vmap_two_body_jastrow_parallel_spins = vmap(two_body_jastrow_parallel, in_axes=(None, 0, 0))(j2b_param, r_i, r_j)
        return jnp.sum(vmap_two_body_jastrow_parallel_spins)

    two_body_jastrow_parallel_up = compute_parallel_sum(r_up_carts)
    two_body_jastrow_parallel_dn = compute_parallel_sum(r_dn_carts)

    two_body_jastrow = two_body_jastrow_anti_parallel_val + two_body_jastrow_parallel_up + two_body_jastrow_parallel_dn

    return two_body_jastrow


def _compute_Jastrow_two_body_debug(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """See _api method."""
    j2b_type = jastrow_two_body_data.jastrow_2b_type

    def two_body_jastrow_exp(param: float, r_cart_i: npt.NDArray[np.float64], r_cart_j: npt.NDArray[np.float64]) -> float:
        """Exponential form of J2."""
        return 1.0 / (2.0 * param) * (1.0 - np.exp(-param * np.linalg.norm(r_cart_i - r_cart_j)))

    def two_body_jastrow_pade(param: float, r_cart_i: npt.NDArray[np.float64], r_cart_j: npt.NDArray[np.float64]) -> float:
        """Pade form of J2."""
        return np.linalg.norm(r_cart_i - r_cart_j) / 2.0 * (1.0 + param * np.linalg.norm(r_cart_i - r_cart_j)) ** (-1.0)

    if j2b_type == "pade":
        _j2_anti = two_body_jastrow_pade
        _j2_para = two_body_jastrow_pade
    elif j2b_type == "exp":
        _j2_anti = two_body_jastrow_exp
        _j2_para = two_body_jastrow_exp
    else:
        raise ValueError(f"Unknown jastrow_2b_type: {j2b_type}")

    two_body_jastrow = (
        np.sum(
            [
                _j2_anti(
                    param=jastrow_two_body_data.jastrow_2b_param,
                    r_cart_i=r_up_cart,
                    r_cart_j=r_dn_cart,
                )
                for (r_up_cart, r_dn_cart) in itertools.product(r_up_carts, r_dn_carts)
            ]
        )
        + np.sum(
            [
                _j2_para(
                    param=jastrow_two_body_data.jastrow_2b_param,
                    r_cart_i=r_up_cart_i,
                    r_cart_j=r_up_cart_j,
                )
                for (r_up_cart_i, r_up_cart_j) in itertools.combinations(r_up_carts, 2)
            ]
        )
        + np.sum(
            [
                _j2_para(
                    param=jastrow_two_body_data.jastrow_2b_param,
                    r_cart_i=r_dn_cart_i,
                    r_cart_j=r_dn_cart_j,
                )
                for (r_dn_cart_i, r_dn_cart_j) in itertools.combinations(r_dn_carts, 2)
            ]
        )
    )

    return two_body_jastrow


# @dataclass
@struct.dataclass
class Jastrow_three_body_data:
    """Three-body Jastrow parameters and orbital references.

    The three-body term uses the original matrix layout (square J3 block plus
    last-column J1-like vector). Values are returned without exponentiation;
    callers attach ``exp(J)`` to the wavefunction. All existing functional
    details from the prior docstring are preserved.

    Args:
        orb_data (AOs_sphe_data | AOs_cart_data | MOs_data): Basis/orbital data used for both spins.
        j_matrix (npt.NDArray[np.float64]): J matrix with shape ``(orb_num, orb_num + 1)``. dtype: float64.
    """

    orb_data: AOs_sphe_data | AOs_cart_data | MOs_data = struct.field(
        pytree_node=True, default_factory=AOs_sphe_data
    )  #: Orbital basis (AOs or MOs) shared across spins.
    j_matrix: npt.NDArray[np.float64] = struct.field(
        pytree_node=True, default_factory=lambda: np.array([], dtype=np.float64)
    )  #: J3/J1 matrix; square block plus final column. dtype: float64.

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        if self.j_matrix.shape != (
            self.orb_num,
            self.orb_num + 1,
        ):
            raise ValueError(
                f"dim. of j_matrix = {self.j_matrix.shape} is imcompatible with the expected one "
                + f"= ({self.orb_num}, {self.orb_num + 1}).",
            )

    def _get_info(self) -> list[str]:
        """Return a list of strings containing the information stored in the attributes."""
        info_lines = []
        info_lines.append("**" + self.__class__.__name__)
        info_lines.append(f"  dim. of jastrow_3b_matrix = {self.j_matrix.shape}")
        info_lines.append(
            f"  j3 part of the jastrow_3b_matrix is symmetric? = {np.allclose(self.j_matrix[:, :-1], self.j_matrix[:, :-1].T)}"
        )
        # Replace orb_data.logger_info() with orb_data.get_info() output.
        info_lines.extend(self.orb_data._get_info())
        return info_lines

    def _logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self._get_info():
            logger.info(line)

    @property
    def orb_num(self) -> int:
        """Get number of atomic orbitals.

        Returns:
            int: get number of atomic orbitals.
        """
        return self.orb_data._num_orb

    @property
    def compute_orb_api(self) -> Callable[..., npt.NDArray[np.float64]]:
        """Function for computing AOs or MOs.

        The api method to compute AOs or MOs corresponding to instances
        stored in self.orb_data

        Return:
            Callable: The api method to compute AOs or MOs.

        Raises:
            NotImplementedError:
                If the instances of orb_data is neither AOs_data nor MOs_data.
        """
        if isinstance(self.orb_data, AOs_sphe_data):
            return compute_AOs
        elif isinstance(self.orb_data, AOs_cart_data):
            return compute_AOs
        elif isinstance(self.orb_data, MOs_data):
            return compute_MOs
        else:
            raise NotImplementedError

    @property
    def _j_matrix_jnp(self) -> jax.Array:
        """Return j_matrix as a jax.Array (jnp view of the underlying numpy storage)."""
        # Lift-only fp64 basis-data storage accessor (see _precision.py exemption);
        # consumer casts to its own zone at use site.
        return jnp.asarray(self.j_matrix, dtype=jnp.float64)

    @property
    def ao_exponents(self) -> jax.Array:
        """AO Gaussian exponents (jnp view of underlying numpy storage)."""
        if isinstance(self.orb_data, (AOs_sphe_data, AOs_cart_data)):
            return self.orb_data._exponents_jnp
        elif isinstance(self.orb_data, MOs_data):
            return self.orb_data.aos_data._exponents_jnp
        else:
            raise NotImplementedError(f"Unsupported orb_data type: {type(self.orb_data)}")

    @property
    def ao_coefficients(self) -> jax.Array:
        """AO contraction coefficients (jnp view of underlying numpy storage)."""
        if isinstance(self.orb_data, (AOs_sphe_data, AOs_cart_data)):
            return self.orb_data._coefficients_jnp
        elif isinstance(self.orb_data, MOs_data):
            return self.orb_data.aos_data._coefficients_jnp
        else:
            raise NotImplementedError(f"Unsupported orb_data type: {type(self.orb_data)}")

    def with_updated_ao_exponents(self, new_exp: npt.NDArray[np.float64]) -> "Jastrow_three_body_data":
        """Return a new instance with updated AO exponents."""
        if isinstance(self.orb_data, (AOs_sphe_data, AOs_cart_data)):
            return self.replace(orb_data=self.orb_data.replace(exponents=new_exp))
        elif isinstance(self.orb_data, MOs_data):
            new_aos = self.orb_data.aos_data.replace(exponents=new_exp)
            return self.replace(orb_data=self.orb_data.replace(aos_data=new_aos))
        else:
            raise NotImplementedError(f"Unsupported orb_data type: {type(self.orb_data)}")

    def with_updated_ao_coefficients(self, new_coeff: npt.NDArray[np.float64]) -> "Jastrow_three_body_data":
        """Return a new instance with updated AO contraction coefficients."""
        if isinstance(self.orb_data, (AOs_sphe_data, AOs_cart_data)):
            return self.replace(orb_data=self.orb_data.replace(coefficients=new_coeff))
        elif isinstance(self.orb_data, MOs_data):
            new_aos = self.orb_data.aos_data.replace(coefficients=new_coeff)
            return self.replace(orb_data=self.orb_data.replace(aos_data=new_aos))
        else:
            raise NotImplementedError(f"Unsupported orb_data type: {type(self.orb_data)}")

    @classmethod
    def init_jastrow_three_body_data(
        cls,
        orb_data: AOs_sphe_data | AOs_cart_data | MOs_data,
        random_init: bool = False,
        random_scale: float = 0.01,
        seed: int | None = None,
    ):
        """Initialization.

        Args:
            orb_data: Orbital container (AOs or MOs) used to size the J-matrix.
            random_init: If True, initialize with small random values instead of zeros (for tests).
            random_scale: Upper bound of uniform sampler when random_init is True (default 0.01).
            seed: Optional seed for deterministic initialization when random_init is True.
        """
        dtype_np = get_dtype_np("jastrow_eval")
        if random_init:
            rng = np.random.default_rng(seed)
            j_matrix = rng.uniform(0.0, random_scale, size=(orb_data._num_orb, orb_data._num_orb + 1)).astype(dtype_np)
        else:
            j_matrix = np.zeros((orb_data._num_orb, orb_data._num_orb + 1), dtype=dtype_np)

        jastrow_three_body_data = cls(
            orb_data=orb_data,
            j_matrix=j_matrix,
        )
        return jastrow_three_body_data

    def to_cartesian(self) -> "Jastrow_three_body_data":
        """Convert spherical orbitals to Cartesian and transform the J-matrix.

        If the underlying orbitals are MOs, defer to ``MOs_data.to_cartesian``
        (the J-matrix size is unchanged). For Cartesian inputs, return self.
        """
        if isinstance(self.orb_data, AOs_cart_data):
            return self
        if isinstance(self.orb_data, MOs_data):
            return Jastrow_three_body_data(orb_data=self.orb_data.to_cartesian(), j_matrix=self.j_matrix)
        if not isinstance(self.orb_data, AOs_sphe_data):
            raise ValueError("Cartesian conversion is only available from spherical AOs or MOs.")

        aos_cart, transform_matrix = _aos_sphe_to_cart(self.orb_data)

        dtype_np = get_dtype_np("jastrow_eval")
        square_sph = np.asarray(self.j_matrix[:, :-1], dtype=dtype_np)
        j1_sph = np.asarray(self.j_matrix[:, -1], dtype=dtype_np)
        square_cart = transform_matrix.T @ square_sph @ transform_matrix
        j1_cart = transform_matrix.T @ j1_sph

        j_matrix_cart = np.zeros((aos_cart.num_ao, aos_cart.num_ao + 1), dtype=np.asarray(self.j_matrix).dtype)
        j_matrix_cart[:, :-1] = square_cart
        j_matrix_cart[:, -1] = j1_cart

        return Jastrow_three_body_data(orb_data=aos_cart, j_matrix=j_matrix_cart)

    def to_spherical(self) -> "Jastrow_three_body_data":
        """Convert Cartesian orbitals to spherical and transform the J-matrix.

        If the underlying orbitals are MOs, defer to ``MOs_data.to_spherical``
        (the J-matrix size is unchanged). For spherical inputs, return self.
        """
        if isinstance(self.orb_data, AOs_sphe_data):
            return self
        if isinstance(self.orb_data, MOs_data):
            return Jastrow_three_body_data(orb_data=self.orb_data.to_spherical(), j_matrix=self.j_matrix)
        if not isinstance(self.orb_data, AOs_cart_data):
            raise ValueError("Spherical conversion is only available from Cartesian AOs or MOs.")

        aos_sphe, transform_pinv = _aos_cart_to_sphe(self.orb_data)

        dtype_np = get_dtype_np("jastrow_eval")
        square_cart = np.asarray(self.j_matrix[:, :-1], dtype=dtype_np)
        j1_cart = np.asarray(self.j_matrix[:, -1], dtype=dtype_np)
        square_sph = transform_pinv.T @ square_cart @ transform_pinv
        j1_sph = transform_pinv.T @ j1_cart

        j_matrix_sph = np.zeros((aos_sphe.num_ao, aos_sphe.num_ao + 1), dtype=np.asarray(self.j_matrix).dtype)
        j_matrix_sph[:, :-1] = square_sph
        j_matrix_sph[:, -1] = j1_sph

        return Jastrow_three_body_data(orb_data=aos_sphe, j_matrix=j_matrix_sph)


@struct.dataclass
class Jastrow_NN_data:
    """Container for NN-based Jastrow factor.

    This dataclass stores both the neural network definition and its
    parameters, together with helper functions that integrate the NN
    Jastrow term into the variational-parameter block machinery.

    The intended usage is:

    * ``nn_def`` holds a Flax/SchNet-like module (e.g. NNJastrow).
    * ``params`` holds the corresponding PyTree of parameters.
    * ``flatten_fn`` / ``unflatten_fn`` convert between the PyTree and a
        1D parameter vector for SR/MCMC.
    * If this dataclass is set to ``None`` inside :class:`Jastrow_data`,
        the NN contribution is simply turned off. If it is not ``None``,
        its contribution is evaluated and added on top of the analytic
        three-body Jastrow (if present).
    """

    # Flax module definition (e.g. NNJastrow); not a pytree node.
    nn_def: Any = struct.field(pytree_node=False, default=None)  #: Flax module definition (e.g., NNJastrow).

    # Flax parameters PyTree (typically a FrozenDict); this is the actual
    # variational parameter set.
    params: Any = struct.field(pytree_node=True, default=None)  #: Parameter PyTree for ``nn_def``.

    # Utilities to flatten/unflatten params for VariationalParameterBlock.
    # NOTE: We do *not* store these function objects directly as dataclass
    # fields because they are not reliably picklable. Instead we store only
    # simple metadata (treedef, shapes) and reconstruct the functions on the
    # fly via properties below.
    flat_shape: tuple[int, ...] = struct.field(pytree_node=False, default=())  #: Shape of flattened params.
    num_params: int = struct.field(pytree_node=False, default=0)  #: Total number of parameters.

    # Metadata needed to reconstruct flatten_fn/unflatten_fn.
    treedef: Any = struct.field(pytree_node=False, default=None)  #: PyTree treedef for params.
    shapes: list[tuple[int, ...]] = struct.field(pytree_node=False, default_factory=list)  #: Per-leaf shapes.

    # Optional architecture/hyperparameters for logging and reproducibility.
    hidden_dim: int = struct.field(pytree_node=False, default=64)  #: Hidden width used in NNJastrow.
    num_layers: int = struct.field(pytree_node=False, default=3)  #: Number of PauliNet blocks.
    num_rbf: int = struct.field(pytree_node=False, default=16)  #: PhysNet radial basis size.
    cutoff: float = struct.field(pytree_node=False, default=5.0)  #: Radial cutoff for features.
    num_species: int = struct.field(pytree_node=False, default=0)  #: Count of unique nuclear species.
    species_lookup: tuple[int, ...] = struct.field(pytree_node=False, default=(0,))  #: Lookup table mapping Z to species ids.
    species_values: tuple[int, ...] = struct.field(
        pytree_node=False, default_factory=tuple
    )  #: Sorted unique atomic numbers used.

    # Structure information required to evaluate the NN J3 term.
    # This is a pytree node so that gradients with respect to nuclear positions
    # (atomic forces) can propagate into structure_data.positions, consistent
    # with the rest of the codebase.
    structure_data: Structure_data | None = struct.field(
        pytree_node=True, default=None
    )  #: Structure info required for NN evaluation.

    def __post_init__(self):
        """Populate flat_shape/num_params/treedef/shapes from params if needed.

        We *do not* attach flatten/unflatten functions here; instead they are
        exposed as properties that reconstruct the closures on demand so that
        this dataclass remains pickle-friendly (only pure data is serialized).
        """
        if self.params is None:
            return

        # If treedef/shapes are missing, infer them from params.
        if self.treedef is None or not self.shapes:
            flat, treedef, shapes = _flatten_params_with_treedef(self.params)
            object.__setattr__(self, "flat_shape", tuple(flat.shape))
            object.__setattr__(self, "num_params", int(flat.size))
            object.__setattr__(self, "treedef", treedef)
            object.__setattr__(self, "shapes", list(shapes))

    # --- Lazy, non-serialised helpers for SR/MCMC ---

    @property
    def flatten_fn(self) -> Callable[[Any], jnp.ndarray]:
        """Return a flatten function built from ``treedef``.

        This is constructed on each access and is not part of the
        serialized state (so it will not cause pickle errors).
        """
        if self.treedef is None:
            # Fallback: infer treedef/shapes from current params.
            flat, treedef, shapes = _flatten_params_with_treedef(self.params)
            object.__setattr__(self, "flat_shape", tuple(flat.shape))
            object.__setattr__(self, "num_params", int(flat.size))
            object.__setattr__(self, "treedef", treedef)
            object.__setattr__(self, "shapes", list(shapes))
        return _make_flatten_fn(self.treedef)

    @property
    def unflatten_fn(self) -> Callable[[jnp.ndarray], Any]:
        """Return an unflatten function built from ``treedef`` and ``shapes``.

        As with :py:meth:`flatten_fn`, this is constructed on each access and
        not stored inside the pickled state.
        """
        if self.treedef is None or not self.shapes:
            flat, treedef, shapes = _flatten_params_with_treedef(self.params)
            object.__setattr__(self, "flat_shape", tuple(flat.shape))
            object.__setattr__(self, "num_params", int(flat.size))
            object.__setattr__(self, "treedef", treedef)
            object.__setattr__(self, "shapes", list(shapes))
        return _make_unflatten_fn(self.treedef, self.shapes)

    @classmethod
    def init_from_structure(
        cls,
        structure_data: "Structure_data",
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_rbf: int = 16,
        cutoff: float = 5.0,
        key=None,
    ) -> "Jastrow_NN_data":
        """Initialize NN Jastrow from structure information.

        This creates a PauliNet-style NNJastrow module, initializes its
        parameters with a dummy electron configuration, and prepares
        flatten/unflatten utilities for SR/MCMC.
        """
        _ensure_flax_trace_level_compat()
        if key is None:
            key = jax.random.PRNGKey(0)

        _ensure_flax_trace_level_compat()

        atomic_numbers = np.asarray(structure_data.atomic_numbers, dtype=np.int32)
        species_values = np.unique(np.concatenate([atomic_numbers, np.array([0], dtype=np.int32)]))
        num_species = int(species_values.shape[0])
        max_species = int(species_values.max())
        species_lookup = np.zeros(max_species + 1, dtype=np.int32)
        for idx, species in enumerate(species_values):
            species_lookup[species] = idx

        species_lookup_tuple = tuple(int(x) for x in species_lookup)

        nn_def = NNJastrow(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_rbf=num_rbf,
            cutoff=cutoff,
            species_lookup=species_lookup_tuple,
            num_species=num_species,
        )

        # Dummy electron positions for parameter initialization:
        # use one spin-up and one spin-down electron at the origin so that
        # both PauliNet channels are initialized with valid shapes.
        dtype_jnp = get_dtype_jnp("jastrow_eval")
        r_up_init = jnp.zeros((1, 3), dtype=dtype_jnp)
        r_dn_init = jnp.zeros((1, 3), dtype=dtype_jnp)
        R_n = structure_data._positions_cart_jnp.astype(dtype_jnp)  # (n_nuc, 3)
        Z_n = jnp.asarray(structure_data.atomic_numbers)  # (n_nuc,)

        rngs = {"params": key}
        variables = nn_def.init(rngs, r_up_init, r_dn_init, R_n, Z_n)
        params = variables["params"]
        # Initialize the NN parameters with small random values so that the
        # NN J3 contribution starts near zero but still has gradient signal.

        leaves, treedef = tree_flatten(params)
        noise_keys = jax.random.split(key, len(leaves))
        scale = 1e-10
        noisy_leaves = [leaf + scale * jax.random.normal(k, leaf.shape) for leaf, k in zip(leaves, noise_keys, strict=True)]
        params = tree_unflatten(treedef, noisy_leaves)

        # Build metadata needed to reconstruct flatten / unflatten
        # utilities. The actual callables are created lazily in
        # __post_init__ to keep this dataclass pickle-friendly.
        flat, treedef, shapes = _flatten_params_with_treedef(params)

        return cls(
            nn_def=nn_def,
            params=params,
            flat_shape=flat.shape,
            num_params=int(flat.size),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_rbf=num_rbf,
            cutoff=cutoff,
            num_species=num_species,
            species_lookup=species_lookup_tuple,
            species_values=tuple(int(x) for x in species_values.tolist()),
            structure_data=structure_data,
            treedef=treedef,
            shapes=list(shapes),
        )

    def _get_info(self) -> list[str]:
        """Return a list of human-readable strings describing this NN Jastrow."""
        info = []
        info.append("**Jastrow_NN_data")
        info.append(f"  hidden_dim = {self.hidden_dim}")
        info.append(f"  num_layers = {self.num_layers}")
        info.append(f"  num_rbf = {self.num_rbf}")
        info.append(f"  cutoff = {self.cutoff}")
        info.append(f"  num_species = {self.num_species}")
        if self.species_values:
            info.append(f"  species_values = {self.species_values}")
        info.append(f"  num_params = {self.num_params}")
        if self.params is None:
            info.append("  params = None (Neural-Network Jastrow disabled)")
        return info


def compute_Jastrow_three_body(
    jastrow_three_body_data: Jastrow_three_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float:
    r"""Evaluate the three-body Jastrow :math:`J_3` (analytic) without exponentiation.

    This preserves the original functional form: the square J3 block couples
    electron pairs and the last column acts as a J1-like vector. Returned value
    is ``J``; attach ``exp(J)`` externally.

    Args:
        jastrow_three_body_data: Three-body Jastrow parameters and orbitals.
        r_up_carts: Spin-up electron coordinates with shape ``(N_up, 3)``.
        r_dn_carts: Spin-down electron coordinates with shape ``(N_dn, 3)``.

    Returns:
        float: Three-body Jastrow value (before exponentiation).
    """
    # r_*_carts forwarded unchanged to ``compute_orb_api`` (the AO/MO kernels
    # reconstruct ``r - R`` in float64 internally); do not pre-cast here.
    dtype_jnp = get_dtype_jnp("jastrow_eval")
    j_matrix = jastrow_three_body_data._j_matrix_jnp.astype(dtype_jnp)
    num_electron_up = len(r_up_carts)
    num_electron_dn = len(r_dn_carts)

    aos_up = jnp.array(jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, r_up_carts), dtype=dtype_jnp)
    aos_dn = jnp.array(jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, r_dn_carts), dtype=dtype_jnp)

    K_up = jnp.tril(jnp.ones((num_electron_up, num_electron_up), dtype=dtype_jnp), k=-1)
    K_dn = jnp.tril(jnp.ones((num_electron_dn, num_electron_dn), dtype=dtype_jnp), k=-1)

    j1_matrix_up = j_matrix[:, -1]
    j1_matrix_dn = j_matrix[:, -1]
    j3_matrix_up_up = j_matrix[:, :-1]
    j3_matrix_dn_dn = j_matrix[:, :-1]
    j3_matrix_up_dn = j_matrix[:, :-1]

    e_up = jnp.ones(num_electron_up, dtype=dtype_jnp).T
    e_dn = jnp.ones(num_electron_dn, dtype=dtype_jnp).T

    # print(f"aos_up.shape={aos_up.shape}")
    # print(f"aos_dn.shape={aos_dn.shape}")
    # print(f"e_up.shape={e_up.shape}")
    # print(f"e_dn.shape={e_dn.shape}")
    # print(f"j3_matrix_up_up.shape={j3_matrix_up_up.shape}")
    # print(f"j3_matrix_dn_dn.shape={j3_matrix_dn_dn.shape}")
    # print(f"j3_matrix_up_dn.shape={j3_matrix_up_dn.shape}")

    J3 = (
        j1_matrix_up @ aos_up @ e_up
        + j1_matrix_dn @ aos_dn @ e_dn
        + jnp.trace(aos_up.T @ j3_matrix_up_up @ aos_up @ K_up)
        + jnp.trace(aos_dn.T @ j3_matrix_dn_dn @ aos_dn @ K_dn)
        + e_up.T @ aos_up.T @ j3_matrix_up_dn @ aos_dn @ e_dn
    )

    return J3


def _compute_Jastrow_three_body_debug(
    jastrow_three_body_data: Jastrow_three_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """See _api method."""
    aos_up = jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, r_up_carts)
    aos_dn = jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, r_dn_carts)

    # compute one body
    J_1_up = 0.0
    j1_vector_up = jastrow_three_body_data.j_matrix[:, -1]
    for i in range(len(r_up_carts)):
        ao_up = aos_up[:, i]
        for al in range(len(ao_up)):
            J_1_up += j1_vector_up[al] * ao_up[al]

    J_1_dn = 0.0
    j1_vector_dn = jastrow_three_body_data.j_matrix[:, -1]
    for i in range(len(r_dn_carts)):
        ao_dn = aos_dn[:, i]
        for al in range(len(ao_dn)):
            J_1_dn += j1_vector_dn[al] * ao_dn[al]

    # compute three-body
    J_3_up_up = 0.0
    j3_matrix_up_up = jastrow_three_body_data.j_matrix[:, :-1]
    for i in range(len(r_up_carts)):
        for j in range(i + 1, len(r_up_carts)):
            ao_up_i = aos_up[:, i]
            ao_up_j = aos_up[:, j]
            for al in range(len(ao_up_i)):
                for bm in range(len(ao_up_j)):
                    J_3_up_up += j3_matrix_up_up[al, bm] * ao_up_i[al] * ao_up_j[bm]

    J_3_dn_dn = 0.0
    j3_matrix_dn_dn = jastrow_three_body_data.j_matrix[:, :-1]
    for i in range(len(r_dn_carts)):
        for j in range(i + 1, len(r_dn_carts)):
            ao_dn_i = aos_dn[:, i]
            ao_dn_j = aos_dn[:, j]
            for al in range(len(ao_dn_i)):
                for bm in range(len(ao_dn_j)):
                    J_3_dn_dn += j3_matrix_dn_dn[al, bm] * ao_dn_i[al] * ao_dn_j[bm]

    J_3_up_dn = 0.0
    j3_matrix_up_dn = jastrow_three_body_data.j_matrix[:, :]
    for i in range(len(r_up_carts)):
        for j in range(len(r_dn_carts)):
            ao_up_i = aos_up[:, i]
            ao_dn_j = aos_dn[:, j]
            for al in range(len(ao_up_i)):
                for bm in range(len(ao_dn_j)):
                    J_3_up_dn += j3_matrix_up_dn[al, bm] * ao_up_i[al] * ao_dn_j[bm]

    J3 = J_1_up + J_1_dn + J_3_up_up + J_3_dn_dn + J_3_up_dn

    return J3


@struct.dataclass
class Jastrow_data:
    """Jastrow dataclass.

    The class contains data for evaluating a Jastrow function.

    Args:
        jastrow_one_body_data (Jastrow_one_body_data):
            An instance of Jastrow_one_body_data. If None, the one-body Jastrow is turned off.
        jastrow_two_body_data (Jastrow_two_body_data):
            An instance of Jastrow_two_body_data. If None, the two-body Jastrow is turned off.
        jastrow_three_body_data (Jastrow_three_body_data):
            An instance of Jastrow_three_body_data. if None, the three-body Jastrow is turned off.
        jastrow_nn_data (Jastrow_NN_data | None):
            Optional container for a NN-based three-body Jastrow term. If None,
            the Jastrow NN contribution is turned off.
    """

    jastrow_one_body_data: Jastrow_one_body_data | None = struct.field(
        pytree_node=True, default=None
    )  #: Optional one-body Jastrow component.
    jastrow_two_body_data: Jastrow_two_body_data | None = struct.field(
        pytree_node=True, default=None
    )  #: Optional two-body Jastrow component.
    jastrow_three_body_data: Jastrow_three_body_data | None = struct.field(
        pytree_node=True, default=None
    )  #: Optional analytic three-body Jastrow component.
    jastrow_nn_data: Jastrow_NN_data | None = struct.field(
        pytree_node=True, default=None
    )  #: Optional NN-based three-body Jastrow component.

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        if self.jastrow_one_body_data is not None:
            self.jastrow_one_body_data.sanity_check()
        if self.jastrow_two_body_data is not None:
            self.jastrow_two_body_data.sanity_check()
        if self.jastrow_three_body_data is not None:
            self.jastrow_three_body_data.sanity_check()

    def to_cartesian(self) -> "Jastrow_data":
        """Return a copy with the three-body term converted to Cartesian AOs/MOs.

        Only the analytic J3 block is transformed; one- and two-body parts are
        unchanged. If no three-body data is present, the current instance is
        returned.
        """
        if self.jastrow_three_body_data is None:
            return self

        return Jastrow_data(
            jastrow_one_body_data=self.jastrow_one_body_data,
            jastrow_two_body_data=self.jastrow_two_body_data,
            jastrow_three_body_data=self.jastrow_three_body_data.to_cartesian(),
            jastrow_nn_data=self.jastrow_nn_data,
        )

    def to_spherical(self) -> "Jastrow_data":
        """Return a copy with the three-body term converted to spherical AOs/MOs.

        Only the analytic J3 block is transformed; one- and two-body parts are
        unchanged. If no three-body data is present, the current instance is
        returned.
        """
        if self.jastrow_three_body_data is None:
            return self

        return Jastrow_data(
            jastrow_one_body_data=self.jastrow_one_body_data,
            jastrow_two_body_data=self.jastrow_two_body_data,
            jastrow_three_body_data=self.jastrow_three_body_data.to_spherical(),
            jastrow_nn_data=self.jastrow_nn_data,
        )

    def _get_info(self) -> list[str]:
        """Return a list of strings representing the logged information from Jastrow data attributes."""
        info_lines = []
        # Replace jastrow_one_body_data.logger_info() with its get_info() output if available.
        if self.jastrow_one_body_data is not None:
            info_lines.extend(self.jastrow_one_body_data._get_info())
        # Replace jastrow_two_body_data.logger_info() with its get_info() output if available.
        if self.jastrow_two_body_data is not None:
            info_lines.extend(self.jastrow_two_body_data._get_info())
        # Replace jastrow_three_body_data.logger_info() with its get_info() output if available.
        if self.jastrow_three_body_data is not None:
            info_lines.extend(self.jastrow_three_body_data._get_info())
        if self.jastrow_nn_data is not None:
            info_lines.extend(self.jastrow_nn_data._get_info())
        return info_lines

    def _logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self._get_info():
            logger.info(line)

    def apply_block_update(self, block: "VariationalParameterBlock") -> "Jastrow_data":
        """Apply a single variational-parameter block update to this Jastrow object.

        This method is the Jastrow-specific counterpart of
        :meth:`Wavefunction_data.apply_block_updates`.  It receives a generic
        :class:`VariationalParameterBlock` whose ``values`` have already been
        updated (typically by ``block.apply_update`` inside the SR/MCMC driver),
        and interprets that block according to Jastrow semantics.

        Responsibilities of this method are:

        * Map the block name (e.g. ``"j1_param"``, ``"j2_param"``,
          ``"j3_matrix"``) to the corresponding internal Jastrow field(s).
        * Enforce Jastrow-specific structural constraints when copying the
          block values into the internal arrays.  In particular, for the
          three-body Jastrow term (J3) this includes:

          - Handling the case where only the last column is variational and the
            rest of the matrix is constrained.
          - Handling the fully square J3 matrix case.
          - Enforcing the required symmetry of the square J3 block.

        By keeping all J1/J2/J3 interpretation and constraints in this method
        (and in the surrounding ``Jastrow_data`` class), the optimizer and
        :class:`VariationalParameterBlock` remain completely structure-agnostic.
        To introduce a new Jastrow parameter, extend the block construction
        in ``Wavefunction_data.get_variational_blocks`` and add the
        corresponding handling here, without touching the SR/MCMC driver.
        """
        dtype_jnp = get_dtype_jnp("jastrow_eval")
        dtype_np = get_dtype_np("jastrow_eval")
        j1 = self.jastrow_one_body_data
        j2 = self.jastrow_two_body_data
        j3 = self.jastrow_three_body_data
        nn3 = self.jastrow_nn_data

        if block.name == "j1_param" and j1 is not None:
            new_param = np.asarray(block.values, dtype=dtype_np).reshape(())
            j1 = Jastrow_one_body_data(
                jastrow_1b_param=new_param,
                structure_data=j1.structure_data,
                core_electrons=j1.core_electrons,
                jastrow_1b_type=j1.jastrow_1b_type,
            )
        elif block.name == "j2_param" and j2 is not None:
            new_param = np.asarray(block.values, dtype=dtype_np).reshape(())
            j2 = Jastrow_two_body_data(jastrow_2b_param=new_param, jastrow_2b_type=j2.jastrow_2b_type)
        elif block.name == "j3_matrix" and j3 is not None:
            j3_new = np.array(block.values, dtype=dtype_np)

            # Symmetrize unconditionally — the method is a no-op for non-symmetric matrices.
            j3_new = self.symmetrize_j3(j3_new)

            j3 = Jastrow_three_body_data(orb_data=j3.orb_data, j_matrix=j3_new)
        elif block.name == "j3_basis_exp" and j3 is not None:
            new_exp = np.asarray(block.values, dtype=dtype_np)
            new_exp = self._symmetrize_ao_basis(j3.orb_data, new_exp)
            j3 = j3.with_updated_ao_exponents(new_exp)
        elif block.name == "j3_basis_coeff" and j3 is not None:
            new_coeff = np.asarray(block.values, dtype=dtype_np)
            new_coeff = self._symmetrize_ao_basis(j3.orb_data, new_coeff)
            j3 = j3.with_updated_ao_coefficients(new_coeff)
        elif block.name == "jastrow_nn_params" and nn3 is not None:
            # Update NN Jastrow parameters: block.values is the flattened parameter vector.
            flat = jnp.asarray(block.values, dtype=dtype_jnp).reshape(-1)
            params_new = nn3.unflatten_fn(flat)
            nn3 = nn3.replace(params=params_new)

        return Jastrow_data(
            jastrow_one_body_data=j1,
            jastrow_two_body_data=j2,
            jastrow_three_body_data=j3,
            jastrow_nn_data=nn3,
        )

    def symmetrize_j3(self, mat):
        """Symmetrize a j3 matrix and return it, or return it unchanged.

        If the square sub-block ``j_matrix[:, :-1]`` of the current
        three-body Jastrow matrix is symmetric within
        ``atol_consistency``, this method enforces ``0.5*(A+A.T)`` on
        that sub-block (leaving the last column untouched) and returns
        the full 2-D matrix.  Otherwise the input matrix is returned
        unchanged.

        This method is the **single source of truth** for all symmetry
        operations on the j3 matrix.  Both :meth:`apply_block_update`
        (parameter enforcement) and the MCMC driver (SN-metric
        symmetrization, selection-mask expansion) use this method,
        so the symmetry logic is never duplicated.  (The MCMC driver
        wraps it with flatten/unflatten in
        :meth:`~Wavefunction_data.get_variational_blocks`.)

        .. note::

           When molecular or crystal spatial symmetry is incorporated in
           the future, **only this method** needs to be extended (e.g. to
           average over a symmetry-group orbit) — every call site
           automatically follows.

        Args:
            mat: 2-D j3 matrix to symmetrize.

        Returns:
            Symmetrized matrix, or the input unchanged if no symmetry applies.
        """
        j3 = self.jastrow_three_body_data
        if j3 is None:
            return mat
        j3_arr = np.asarray(j3.j_matrix)
        if j3_arr.ndim != 2 or j3_arr.shape[1] < 2:
            return mat
        sq = j3_arr[:, :-1]
        if sq.shape[0] != sq.shape[1]:
            return mat
        if np.allclose(sq, sq.T, atol=atol_consistency):
            out = mat.copy()
            sq_new = out[:, :-1]
            out[:, :-1] = 0.5 * (sq_new + sq_new.T)
            return out
        return mat

    @staticmethod
    def _symmetrize_ao_basis(orb_data, arr: np.ndarray) -> np.ndarray:
        """Average within same-atom same-shell primitive groups.

        This is the single source of truth for shell-sharing constraints
        on AO basis exponents/coefficients in the Jastrow factor.
        """
        from .wavefunction import _get_aos_data

        spm = ShellPrimMap.from_aos_data(_get_aos_data(orb_data))
        return spm.symmetrize(arr)

    def accumulate_position_grad(self, grad_jastrow: "Jastrow_data"):
        """Aggregate position gradients from all active Jastrow components."""
        grad = 0.0
        if grad_jastrow.jastrow_one_body_data is not None:
            grad += grad_jastrow.jastrow_one_body_data.structure_data.positions
        if grad_jastrow.jastrow_three_body_data is not None:
            grad += grad_jastrow.jastrow_three_body_data.orb_data.structure_data.positions
        if grad_jastrow.jastrow_nn_data is not None:
            grad += grad_jastrow.jastrow_nn_data.structure_data.positions
        return grad

    def collect_param_grads(self, grad_jastrow: "Jastrow_data") -> dict[str, object]:
        """Collect parameter gradients into a flat dict keyed by block name."""
        grads: dict[str, object] = {}
        if grad_jastrow.jastrow_one_body_data is not None:
            grads["j1_param"] = grad_jastrow.jastrow_one_body_data.jastrow_1b_param
        if grad_jastrow.jastrow_two_body_data is not None:
            grads["j2_param"] = grad_jastrow.jastrow_two_body_data.jastrow_2b_param
        if grad_jastrow.jastrow_three_body_data is not None:
            grads["j3_matrix"] = grad_jastrow.jastrow_three_body_data.j_matrix
            # AO basis gradients
            grads["j3_basis_exp"] = grad_jastrow.jastrow_three_body_data.ao_exponents
            grads["j3_basis_coeff"] = grad_jastrow.jastrow_three_body_data.ao_coefficients
        if grad_jastrow.jastrow_nn_data is not None and grad_jastrow.jastrow_nn_data.params is not None:
            grads["jastrow_nn_params"] = grad_jastrow.jastrow_nn_data.params
        return grads


def compute_Jastrow_part(jastrow_data: Jastrow_data, r_up_carts: jax.Array, r_dn_carts: jax.Array) -> float:
    """Evaluate the total Jastrow ``J = J1 + J2 + J3`` (without exponentiation).

    This preserves the original behavior: the returned scalar ``J`` excludes
    the ``exp`` factor; callers apply ``exp(J)`` to the wavefunction. Both the
    analytic three-body and optional NN three-body contributions are included.

    Args:
        jastrow_data: Collection of active Jastrow components (J1/J2/J3/NN).
        r_up_carts: Spin-up electron coordinates with shape ``(N_up, 3)``.
        r_dn_carts: Spin-down electron coordinates with shape ``(N_dn, 3)``.

    Returns:
        float: Total Jastrow value before exponentiation.
    """
    # r_*_carts forwarded unchanged to the sub-Jastrow kernels (each handles
    # its own zone management). Do not pre-cast.
    dtype_jnp = get_dtype_jnp("jastrow_eval")

    J1 = 0.0
    J2 = 0.0
    J3 = 0.0

    # one-body
    if jastrow_data.jastrow_one_body_data is not None:
        J1 += compute_Jastrow_one_body(jastrow_data.jastrow_one_body_data, r_up_carts, r_dn_carts)

    # two-body
    if jastrow_data.jastrow_two_body_data is not None:
        J2 += compute_Jastrow_two_body(jastrow_data.jastrow_two_body_data, r_up_carts, r_dn_carts)

    # three-body (analytic)
    if jastrow_data.jastrow_three_body_data is not None:
        J3 += compute_Jastrow_three_body(jastrow_data.jastrow_three_body_data, r_up_carts, r_dn_carts)

    # three-body (NN)
    if jastrow_data.jastrow_nn_data is not None:
        nn3 = jastrow_data.jastrow_nn_data
        if nn3.structure_data is None:
            raise ValueError("NN_Jastrow_data.structure_data must be set to evaluate NN J3.")

        R_n = nn3.structure_data._positions_cart_jnp.astype(dtype_jnp)
        Z_n = jnp.asarray(nn3.structure_data.atomic_numbers, dtype=dtype_jnp)
        nn_params = jax.tree_util.tree_map(
            lambda x: x.astype(dtype_jnp) if hasattr(x, "dtype") and x.dtype.kind == "f" else x, nn3.params
        )
        J3_nn = nn3.nn_def.apply({"params": nn_params}, r_up_carts, r_dn_carts, R_n, Z_n)
        J3 = J3 + J3_nn

    J = J1 + J2 + J3

    return J


def _compute_Jastrow_part_debug(
    jastrow_data: Jastrow_data, r_up_carts: npt.NDArray[np.float64], r_dn_carts: npt.NDArray[np.float64]
) -> float:
    """See compute_Jastrow_part_jax for more details."""
    dtype_jnp = get_dtype_jnp("jastrow_eval")
    dtype_np = get_dtype_np("jastrow_eval")
    J1 = 0.0
    J2 = 0.0
    J3 = 0.0

    # one-body
    if jastrow_data.jastrow_one_body_data is not None:
        J1 += _compute_Jastrow_one_body_debug(jastrow_data.jastrow_one_body_data, r_up_carts, r_dn_carts)

    # two-body
    if jastrow_data.jastrow_two_body_data is not None:
        J2 += _compute_Jastrow_two_body_debug(jastrow_data.jastrow_two_body_data, r_up_carts, r_dn_carts)

    # three-body (analytic)
    if jastrow_data.jastrow_three_body_data is not None:
        J3 += _compute_Jastrow_three_body_debug(jastrow_data.jastrow_three_body_data, r_up_carts, r_dn_carts)

    # three-body (NN)
    if jastrow_data.jastrow_nn_data is not None:
        nn3 = jastrow_data.jastrow_nn_data
        if nn3.structure_data is None:
            raise ValueError("NN_Jastrow_data.structure_data must be set to evaluate NN J3 (debug).")

        R_n = np.asarray(nn3.structure_data.positions, dtype=dtype_np)
        Z_n = np.asarray(nn3.structure_data.atomic_numbers, dtype=dtype_np)

        # Use JAX NN for debug as well; convert inputs to jnp and back to float
        nn_params = jax.tree_util.tree_map(
            lambda x: x.astype(dtype_jnp) if hasattr(x, "dtype") and x.dtype.kind == "f" else x, nn3.params
        )
        J3_nn = nn3.nn_def.apply(
            {"params": nn_params},
            jnp.asarray(r_up_carts, dtype=dtype_jnp),
            jnp.asarray(r_dn_carts, dtype=dtype_jnp),
            jnp.asarray(R_n, dtype=dtype_jnp),
            jnp.asarray(Z_n, dtype=dtype_jnp),
        )
        J3 += float(J3_nn)

    J = J1 + J2 + J3

    return J


def _compute_ratio_Jastrow_part_rank1_update(
    jastrow_data: Jastrow_data,
    old_r_up_carts: jax.Array,
    old_r_dn_carts: jax.Array,
    new_r_up_carts_arr: jax.Array,
    new_r_dn_carts_arr: jax.Array,
    j3_state: "Jastrow_three_body_streaming_state | None" = None,
) -> jax.Array:
    r"""Compute :math:`\exp(J(\mathbf r'))/\exp(J(\mathbf r))` for batched moves.

    This follows the original ratio logic (including exp) while updating types
    to use ``jax.Array`` inputs. The return is one ratio per proposed grid
    configuration.

    Args:
        jastrow_data: Active Jastrow components.
        old_r_up_carts: Reference spin-up coordinates with shape ``(N_up, 3)``.
        old_r_dn_carts: Reference spin-down coordinates with shape ``(N_dn, 3)``.
        new_r_up_carts_arr: Proposed spin-up coordinates with shape ``(N_grid, N_up, 3)``.
        new_r_dn_carts_arr: Proposed spin-down coordinates with shape ``(N_grid, N_dn, 3)``.
        j3_state: Optional cached J3 auxiliaries consistent with
            ``(old_r_up_carts, old_r_dn_carts)``. When provided, the J3 block
            reuses ``aos_*``, ``j3_mat @ aos_*``, ``j3_mat.T @ aos_*`` from the
            state instead of recomputing them — saves per-call ``O(n_ao^2 * N_e)``
            in matmul work. Pass ``None`` (default) to recompute from scratch
            (the original 1-shot path used outside the projection loop).

    Returns:
        jax.Array: Jastrow ratios per grid with shape ``(N_grid,)`` (includes ``exp``).

    Warning:
        Each proposed configuration in ``new_r_up_carts_arr`` / ``new_r_dn_carts_arr``
        must differ from ``old_r_up_carts`` / ``old_r_dn_carts`` in **exactly one
        electron**.  The moved electron is identified via ``jnp.argmax`` on the
        change mask; if two or more electrons differ in the same config, only the
        first changed electron is detected and the ratio is silently incorrect.
        This function is intended exclusively for the non-local ECP integration
        grid generated by the MCMC loop, where exactly one electron is displaced
        per grid point by construction.
    """
    # Forward old/new r_up/dn_carts as-is (Principle 3a — no parameter rebind).
    # Module-level forwards (compute_Jastrow_part, compute_Jastrow_one_body,
    # compute_orb_api, NN_Jastrow.apply) handle their own use-site casts.
    # Inline arithmetic in the local J1/J2/J3 closures below casts at the diff
    # site (Principle 3b) — for r-r differences the operand is reconstructed in
    # caller-supplied precision (fp64 from MCMC walker state) before downcast.
    dtype_jnp = get_dtype_jnp("jastrow_ratio")

    num_up = old_r_up_carts.shape[0]
    num_dn = old_r_dn_carts.shape[0]
    n_grid = new_r_up_carts_arr.shape[0]
    if num_up == 0 or num_dn == 0:
        jastrow_x = compute_Jastrow_part(jastrow_data, old_r_up_carts, old_r_dn_carts)
        jastrow_xp = vmap(compute_Jastrow_part, in_axes=(None, 0, 0))(jastrow_data, new_r_up_carts_arr, new_r_dn_carts_arr)
        return jnp.exp(jastrow_xp - jastrow_x)

    J_ratio = jnp.ones(n_grid, dtype=dtype_jnp)

    # J1 part
    if jastrow_data.jastrow_one_body_data is not None:
        j1_data = jastrow_data.jastrow_one_body_data

        if num_up == 0:

            def compute_one_grid_J1(j1_data, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts):
                delta_dn = new_r_dn_carts - old_r_dn_carts
                nonzero_dn = jnp.any(delta_dn != 0, axis=1)
                idx_dn = jnp.argmax(nonzero_dn)
                r_dn_new = new_r_dn_carts[idx_dn]
                r_dn_old = old_r_dn_carts[idx_dn]
                j1_new = compute_Jastrow_one_body(
                    j1_data, jnp.zeros((0, 3), dtype=dtype_jnp), jnp.expand_dims(r_dn_new, axis=0)
                )
                j1_old = compute_Jastrow_one_body(
                    j1_data, jnp.zeros((0, 3), dtype=dtype_jnp), jnp.expand_dims(r_dn_old, axis=0)
                )
                return jnp.exp(j1_new - j1_old)

        elif num_dn == 0:

            def compute_one_grid_J1(j1_data, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts):
                delta_up = new_r_up_carts - old_r_up_carts
                nonzero_up = jnp.any(delta_up != 0, axis=1)
                idx_up = jnp.argmax(nonzero_up)
                r_up_new = new_r_up_carts[idx_up]
                r_up_old = old_r_up_carts[idx_up]
                j1_new = compute_Jastrow_one_body(
                    j1_data, jnp.expand_dims(r_up_new, axis=0), jnp.zeros((0, 3), dtype=dtype_jnp)
                )
                j1_old = compute_Jastrow_one_body(
                    j1_data, jnp.expand_dims(r_up_old, axis=0), jnp.zeros((0, 3), dtype=dtype_jnp)
                )
                return jnp.exp(j1_new - j1_old)

        else:

            def compute_one_grid_J1(j1_data, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts):
                delta_up = new_r_up_carts - old_r_up_carts
                delta_dn = new_r_dn_carts - old_r_dn_carts
                up_moved = jnp.any(delta_up != 0)

                nonzero_up = jnp.any(delta_up != 0, axis=1)
                nonzero_dn = jnp.any(delta_dn != 0, axis=1)
                idx_up = jnp.argmax(nonzero_up)
                idx_dn = jnp.argmax(nonzero_dn)

                def up_case(args):
                    j1_data, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts = args
                    r_up_new = new_r_up_carts[idx_up]
                    r_up_old = old_r_up_carts[idx_up]
                    j1_new = compute_Jastrow_one_body(
                        j1_data, jnp.expand_dims(r_up_new, axis=0), jnp.zeros((0, 3), dtype=dtype_jnp)
                    )
                    j1_old = compute_Jastrow_one_body(
                        j1_data, jnp.expand_dims(r_up_old, axis=0), jnp.zeros((0, 3), dtype=dtype_jnp)
                    )
                    return jnp.exp(j1_new - j1_old)

                def dn_case(args):
                    j1_data, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts = args
                    r_dn_new = new_r_dn_carts[idx_dn]
                    r_dn_old = old_r_dn_carts[idx_dn]
                    j1_new = compute_Jastrow_one_body(
                        j1_data, jnp.zeros((0, 3), dtype=dtype_jnp), jnp.expand_dims(r_dn_new, axis=0)
                    )
                    j1_old = compute_Jastrow_one_body(
                        j1_data, jnp.zeros((0, 3), dtype=dtype_jnp), jnp.expand_dims(r_dn_old, axis=0)
                    )
                    return jnp.exp(j1_new - j1_old)

                return jax.lax.cond(
                    up_moved,
                    up_case,
                    dn_case,
                    (j1_data, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts),
                )

        J1_ratio = vmap(compute_one_grid_J1, in_axes=(None, 0, 0, None, None))(
            j1_data,
            new_r_up_carts_arr,
            new_r_dn_carts_arr,
            old_r_up_carts,
            old_r_dn_carts,
        )
        J_ratio *= jnp.ravel(J1_ratio)

    def _two_body_jastrow_exp(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> float:
        """Exponential form of J2."""
        # Reconstruct diff in caller-supplied precision then downcast (Principle 3b).
        diff = (r_cart_i - r_cart_j).astype(dtype_jnp)
        return 1.0 / (2.0 * param) * (1.0 - jnp.exp(-param * jnp.linalg.norm(diff)))

    def _two_body_jastrow_pade(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> float:
        """Pade form of J2."""
        # Reconstruct diff in caller-supplied precision then downcast (Principle 3b).
        diff = (r_cart_i - r_cart_j).astype(dtype_jnp)
        return jnp.linalg.norm(diff) / 2.0 * (1.0 + param * jnp.linalg.norm(diff)) ** (-1.0)

    # Select the functional form based on type
    if jastrow_data.jastrow_two_body_data is not None:
        _j2b_type = jastrow_data.jastrow_two_body_data.jastrow_2b_type
    else:
        _j2b_type = "pade"

    if _j2b_type == "pade":
        two_body_jastrow_anti_parallel_spins = _two_body_jastrow_pade
        two_body_jastrow_parallel_spins = _two_body_jastrow_pade
    else:
        two_body_jastrow_anti_parallel_spins = _two_body_jastrow_exp
        two_body_jastrow_parallel_spins = _two_body_jastrow_exp

    def compute_one_grid_J2(jastrow_2b_param, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts):
        delta_up = new_r_up_carts - old_r_up_carts
        delta_dn = new_r_dn_carts - old_r_dn_carts
        up_moved = jnp.any(delta_up != 0)
        if num_up == 0:
            nonzero_dn = jnp.any(delta_dn != 0, axis=1)
            idx = jnp.argmax(nonzero_dn)
            up_moved = False
        elif num_dn == 0:
            nonzero_up = jnp.any(delta_up != 0, axis=1)
            idx = jnp.argmax(nonzero_up)
            up_moved = True
        else:
            nonzero_up = jnp.any(delta_up != 0, axis=1)
            nonzero_dn = jnp.any(delta_dn != 0, axis=1)
            idx_up = jnp.argmax(nonzero_up)
            idx_dn = jnp.argmax(nonzero_dn)
            idx = jax.lax.cond(up_moved, lambda _: idx_up, lambda _: idx_dn, operand=None)

        def up_case(jastrow_2b_param, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts):
            new_r_up_carts_extracted = jnp.expand_dims(new_r_up_carts[idx], axis=0)  # shape=(1,3)
            old_r_up_carts_extracted = jnp.expand_dims(old_r_up_carts[idx], axis=0)  # shape=(1,3)
            J2_up_up_new = jnp.sum(
                vmap(two_body_jastrow_parallel_spins, in_axes=(None, None, 0))(
                    jastrow_2b_param, new_r_up_carts_extracted, new_r_up_carts
                )
            )
            J2_up_up_old = jnp.sum(
                vmap(two_body_jastrow_parallel_spins, in_axes=(None, None, 0))(
                    jastrow_2b_param, old_r_up_carts_extracted, old_r_up_carts
                )
            )
            J2_up_dn_new = jnp.sum(
                vmap(two_body_jastrow_anti_parallel_spins, in_axes=(None, None, 0))(
                    jastrow_2b_param, new_r_up_carts_extracted, old_r_dn_carts
                )
            )
            J2_up_dn_old = jnp.sum(
                vmap(two_body_jastrow_anti_parallel_spins, in_axes=(None, None, 0))(
                    jastrow_2b_param, old_r_up_carts_extracted, old_r_dn_carts
                )
            )
            return jnp.exp(J2_up_dn_new - J2_up_dn_old + J2_up_up_new - J2_up_up_old)

        def dn_case(jastrow_2b_param, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts):
            new_r_dn_carts_extracted = jnp.expand_dims(new_r_dn_carts[idx], axis=0)  # shape=(1,3)
            old_r_dn_carts_extracted = jnp.expand_dims(old_r_dn_carts[idx], axis=0)  # shape=(1,3)
            J2_dn_dn_new = jnp.sum(
                vmap(two_body_jastrow_parallel_spins, in_axes=(None, None, 0))(
                    jastrow_2b_param, new_r_dn_carts_extracted, new_r_dn_carts
                )
            )
            J2_dn_dn_old = jnp.sum(
                vmap(two_body_jastrow_parallel_spins, in_axes=(None, None, 0))(
                    jastrow_2b_param, old_r_dn_carts_extracted, old_r_dn_carts
                )
            )
            J2_up_dn_new = jnp.sum(
                vmap(two_body_jastrow_anti_parallel_spins, in_axes=(None, 0, None))(
                    jastrow_2b_param, old_r_up_carts, new_r_dn_carts_extracted
                )
            )
            J2_up_dn_old = jnp.sum(
                vmap(two_body_jastrow_anti_parallel_spins, in_axes=(None, 0, None))(
                    jastrow_2b_param, old_r_up_carts, old_r_dn_carts_extracted
                )
            )

            return jnp.exp(J2_up_dn_new - J2_up_dn_old + J2_dn_dn_new - J2_dn_dn_old)

        if num_up == 0:
            return dn_case(jastrow_2b_param, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts)
        if num_dn == 0:
            return up_case(jastrow_2b_param, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts)

        return jax.lax.cond(
            up_moved,
            up_case,
            dn_case,
            *(jastrow_2b_param, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts),
        )

    # J2 part
    if jastrow_data.jastrow_two_body_data is not None:
        j2_param = jastrow_data.jastrow_two_body_data.jastrow_2b_param
        _j2_type = jastrow_data.jastrow_two_body_data.jastrow_2b_type

        def _safe_norm(diff):
            sq = jnp.sum(diff**2, axis=-1)
            return jnp.where(sq > 0, jnp.sqrt(jnp.where(sq > 0, sq, jnp.ones_like(sq))), jnp.zeros_like(sq))

        if _j2_type == "pade":

            def _j2_from_dist(dist, param):
                return dist / 2.0 * (1.0 + param * dist) ** (-1.0)
        else:

            def _j2_from_dist(dist, param):
                return 1.0 / (2.0 * param) * (1.0 - jnp.exp(-param * dist))

        def compute_pairwise_sums(pos1, pos2):
            if pos1.shape[0] == 0 or pos2.shape[0] == 0:
                return jnp.zeros(pos1.shape[0], dtype=dtype_jnp)
            # Reconstruct diff in caller-supplied precision then downcast (Principle 3b).
            diff = (pos1[:, None, :] - pos2[None, :, :]).astype(dtype_jnp)
            dists = _safe_norm(diff)
            vals = _j2_from_dist(dists, j2_param)
            return jnp.sum(vals, axis=1)

        J2_sum_up_up = compute_pairwise_sums(old_r_up_carts, old_r_up_carts)
        J2_sum_up_dn = compute_pairwise_sums(old_r_up_carts, old_r_dn_carts)
        J2_sum_dn_dn = compute_pairwise_sums(old_r_dn_carts, old_r_dn_carts)
        J2_sum_dn_up = compute_pairwise_sums(old_r_dn_carts, old_r_up_carts)

        delta_up_all = new_r_up_carts_arr - old_r_up_carts
        delta_dn_all = new_r_dn_carts_arr - old_r_dn_carts
        moved_up_mask = jnp.any(delta_up_all != 0.0, axis=2)
        moved_dn_mask = jnp.any(delta_dn_all != 0.0, axis=2)
        moved_up_exists = jnp.any(moved_up_mask, axis=1)
        idx_up = jnp.argmax(moved_up_mask.astype(jnp.int32), axis=1)
        idx_dn = jnp.argmax(moved_dn_mask.astype(jnp.int32), axis=1)

        r_up_new = jnp.take_along_axis(new_r_up_carts_arr, idx_up[:, None, None], axis=1).reshape(-1, 3)
        r_dn_new = jnp.take_along_axis(new_r_dn_carts_arr, idx_dn[:, None, None], axis=1).reshape(-1, 3)
        r_up_old = jnp.take(old_r_up_carts, idx_up, axis=0)
        r_dn_old = jnp.take(old_r_dn_carts, idx_dn, axis=0)

        def _batch_pairwise_sum(points_a, points_b, param):
            # Cast operands to the jastrow_ratio zone at the arithmetic use site
            # (Principle 3b). Inputs may arrive in caller-supplied precision; cast
            # before consuming as norm/dot operands to keep the function in-zone.
            pa = points_a.astype(dtype_jnp)
            pb = points_b.astype(dtype_jnp)
            norm_a2 = jnp.sum(pa * pa, axis=1, keepdims=True)
            norm_b2 = jnp.sum(pb * pb, axis=1, keepdims=True).T
            dots = jnp.dot(pa, pb.T)
            d2 = jnp.maximum(norm_a2 + norm_b2 - 2.0 * dots, 0.0)
            safe_d2 = jnp.where(d2 > 0, d2, jnp.ones_like(d2))
            d = jnp.where(d2 > 0, jnp.sqrt(safe_d2), jnp.zeros_like(d2))
            vals = _j2_from_dist(d, param)
            return jnp.sum(vals, axis=1)

        # Up-move branch contributions (all grids in batch)
        up_up_new_raw = _batch_pairwise_sum(r_up_new, old_r_up_carts, j2_param)
        # Reconstruct diff in caller-supplied precision then downcast (Principle 3b).
        up_up_self = _j2_from_dist(jnp.linalg.norm((r_up_new - r_up_old).astype(dtype_jnp), axis=1), j2_param)
        up_up_new = up_up_new_raw - up_up_self
        up_up_old = jnp.take(J2_sum_up_up, idx_up, axis=0)

        up_dn_new = _batch_pairwise_sum(r_up_new, old_r_dn_carts, j2_param)
        up_dn_old = jnp.take(J2_sum_up_dn, idx_up, axis=0)
        J2_ratio_up = jnp.exp((up_up_new - up_up_old) + (up_dn_new - up_dn_old))

        # Down-move branch contributions (all grids in batch)
        dn_dn_new_raw = _batch_pairwise_sum(r_dn_new, old_r_dn_carts, j2_param)
        # Reconstruct diff in caller-supplied precision then downcast (Principle 3b).
        dn_dn_self = _j2_from_dist(jnp.linalg.norm((r_dn_new - r_dn_old).astype(dtype_jnp), axis=1), j2_param)
        dn_dn_new = dn_dn_new_raw - dn_dn_self
        dn_dn_old = jnp.take(J2_sum_dn_dn, idx_dn, axis=0)

        dn_up_new = _batch_pairwise_sum(r_dn_new, old_r_up_carts, j2_param)
        dn_up_old = jnp.take(J2_sum_dn_up, idx_dn, axis=0)
        J2_ratio_dn = jnp.exp((dn_dn_new - dn_dn_old) + (dn_up_new - dn_up_old))

        J2_ratio = jnp.where(moved_up_exists, J2_ratio_up, J2_ratio_dn)

        J_ratio *= jnp.ravel(J2_ratio)

    # J3 part  (batched AO evaluation — avoids per-config compute_orb_api inside vmap)
    if jastrow_data.jastrow_three_body_data is not None:
        j3d = jastrow_data.jastrow_three_body_data
        j3_mat = j3d._j_matrix_jnp[:, :-1]  # (n_ao, n_ao)  shared for up-up / dn-dn / up-dn
        j1_vec = j3d._j_matrix_jnp[:, -1]  # (n_ao,)

        # Old AOs evaluated once.
        # When ``j3_state`` is supplied, the cached AOs/W/U/cross_vec from the
        # streaming state are consistent with ``(old_r_up_carts, old_r_dn_carts)``
        # by contract — we just dtype-cast into the jastrow_ratio zone and skip
        # the recomputation. Python-static dispatch (j3_state is None vs not).
        if j3_state is None:
            aos_up_old = jnp.array(j3d.compute_orb_api(j3d.orb_data, old_r_up_carts), dtype=dtype_jnp)  # (n_ao, N_up)
            aos_dn_old = jnp.array(j3d.compute_orb_api(j3d.orb_data, old_r_dn_carts), dtype=dtype_jnp)  # (n_ao, N_dn)
        else:
            aos_up_old = j3_state.aos_up.astype(dtype_jnp)
            aos_dn_old = j3_state.aos_dn.astype(dtype_jnp)

        N_batch = new_r_up_carts_arr.shape[0]

        # Detect which spin moved and which electron index per config
        delta_up_batch = new_r_up_carts_arr - old_r_up_carts[None]  # (N, N_up, 3)
        delta_dn_batch = new_r_dn_carts_arr - old_r_dn_carts[None]  # (N, N_dn, 3)
        up_moved_batch = jnp.any(delta_up_batch != 0.0, axis=(1, 2))  # (N,) bool
        idx_up = jnp.argmax(jnp.any(delta_up_batch != 0.0, axis=2).astype(jnp.int32), axis=1)  # (N,)
        idx_dn = jnp.argmax(jnp.any(delta_dn_batch != 0.0, axis=2).astype(jnp.int32), axis=1)  # (N,)

        # New position of the moved electron per config (select spin block)
        r_new_up_moved = jnp.take_along_axis(new_r_up_carts_arr, idx_up[:, None, None], axis=1).reshape(N_batch, 3)
        r_new_dn_moved = jnp.take_along_axis(new_r_dn_carts_arr, idx_dn[:, None, None], axis=1).reshape(N_batch, 3)
        r_old_up_moved = old_r_up_carts[idx_up]  # (N, 3)
        r_old_dn_moved = old_r_dn_carts[idx_dn]  # (N, 3)
        r_new_moved = jnp.where(up_moved_batch[:, None], r_new_up_moved, r_new_dn_moved)  # (N, 3)
        r_old_moved = jnp.where(up_moved_batch[:, None], r_old_up_moved, r_old_dn_moved)  # (N, 3)

        # Single batched AO evaluation for all N configs (replaces N per-config calls inside vmap)
        aos_new_batch = jnp.array(j3d.compute_orb_api(j3d.orb_data, r_new_moved), dtype=dtype_jnp)  # (n_ao, N)
        aos_old_batch = jnp.array(j3d.compute_orb_api(j3d.orb_data, r_old_moved), dtype=dtype_jnp)  # (n_ao, N)
        aos_p_batch = aos_new_batch - aos_old_batch  # (n_ao, N)

        # Precompute constant products (independent of config). With a
        # streaming state, all four matmuls are read directly from the cache
        # — that's the main per-step ``O(n_ao^2 * N_e)`` saving. Note that
        # ``j3_state.j3_mat_T_aos_*`` stores ``j3_mat.T @ aos_*`` of shape
        # ``(n_ao, N_*)``, while we want ``U_* = aos_*.T @ j3_mat`` of shape
        # ``(N_*, n_ao)`` — these are transposes of each other.
        if j3_state is None:
            W_up = jnp.dot(j3_mat, aos_up_old)  # (n_ao, N_up)  = j3_mat @ A_up
            U_up = jnp.dot(aos_up_old.T, j3_mat)  # (N_up, n_ao)  = A_up.T @ j3_mat
            W_dn = jnp.dot(j3_mat, aos_dn_old)  # (n_ao, N_dn)
            U_dn = jnp.dot(aos_dn_old.T, j3_mat)  # (N_dn, n_ao)
            dn_cross_vec = j3_mat @ jnp.sum(aos_dn_old, axis=1)  # (n_ao,): UP cross term constant
            up_cross_vec = jnp.sum(aos_up_old, axis=1) @ j3_mat  # (n_ao,): DN cross term constant
        else:
            W_up = j3_state.j3_mat_aos_up.astype(dtype_jnp)
            W_dn = j3_state.j3_mat_aos_dn.astype(dtype_jnp)
            U_up = j3_state.j3_mat_T_aos_up.astype(dtype_jnp).T
            U_dn = j3_state.j3_mat_T_aos_dn.astype(dtype_jnp).T
            # cross_vec equivalences:
            #   j3_mat @ sum(aos_dn, axis=1) = sum(j3_mat @ aos_dn, axis=1) = sum(W_dn, axis=1)
            #   sum(aos_up, axis=1) @ j3_mat = sum(j3_mat.T @ aos_up, axis=1) = sum(j3_mat_T_aos_up, axis=1)
            dn_cross_vec = jnp.sum(W_dn, axis=1)
            up_cross_vec = jnp.sum(j3_state.j3_mat_T_aos_up.astype(dtype_jnp), axis=1)

        # Q index: idx_up for UP configs, idx_dn for DN configs
        idx_for_Q = jnp.where(up_moved_batch, idx_up, idx_dn)  # (N,)

        # term1: J1-like contribution (identical formula for UP and DN)
        term1 = j1_vec @ aos_p_batch  # (N,)

        # UP formula  -----------------------------------------------------------
        V_up = jnp.dot(aos_p_batch.T, W_up)  # (N, N_up)
        P_up = jnp.dot(U_up, aos_p_batch)  # (N_up, N)
        Q_up_c = (idx_for_Q[:, None] < jnp.arange(num_up)[None, :]).astype(dtype_jnp)  # (N, N_up)
        Q_up_r = (idx_for_Q[:, None] > jnp.arange(num_up)[None, :]).astype(dtype_jnp)  # (N, N_up)
        term2_up = jnp.sum(V_up * Q_up_c, axis=1)  # (N,)
        term3_up = jnp.sum(P_up.T * Q_up_r, axis=1)  # (N,)
        term4_up = dn_cross_vec @ aos_p_batch  # (N,)
        J3_log_up = term1 + term2_up + term3_up + term4_up

        # DN formula  -----------------------------------------------------------
        V_dn = jnp.dot(aos_p_batch.T, W_dn)  # (N, N_dn)
        P_dn = jnp.dot(U_dn, aos_p_batch)  # (N_dn, N)
        Q_dn_c = (idx_for_Q[:, None] < jnp.arange(num_dn)[None, :]).astype(dtype_jnp)  # (N, N_dn)
        Q_dn_r = (idx_for_Q[:, None] > jnp.arange(num_dn)[None, :]).astype(dtype_jnp)  # (N, N_dn)
        term2_dn = jnp.sum(V_dn * Q_dn_c, axis=1)  # (N,)
        term3_dn = jnp.sum(P_dn.T * Q_dn_r, axis=1)  # (N,)
        term4_dn = up_cross_vec @ aos_p_batch  # (N,)
        J3_log_dn = term1 + term2_dn + term3_dn + term4_dn

        # Select UP or DN formula per config
        if num_up == 0:
            J3_ratio = jnp.exp(J3_log_dn)
        elif num_dn == 0:
            J3_ratio = jnp.exp(J3_log_up)
        else:
            J3_ratio = jnp.exp(jnp.where(up_moved_batch, J3_log_up, J3_log_dn))

        J_ratio *= J3_ratio

    # JNN part
    if jastrow_data.jastrow_nn_data is not None:
        nn = jastrow_data.jastrow_nn_data
        if nn.structure_data is None:
            raise ValueError("NN_Jastrow_data.structure_data must be set to evaluate NN J3.")

        R_n = nn.structure_data._positions_cart_jnp.astype(dtype_jnp)
        Z_n = jnp.asarray(nn.structure_data.atomic_numbers)

        def compute_one_grid_JNN(new_r_up_carts, new_r_dn_carts):
            return nn.nn_def.apply({"params": nn.params}, new_r_up_carts, new_r_dn_carts, R_n, Z_n)

        JNN_old = compute_one_grid_JNN(old_r_up_carts, old_r_dn_carts)
        JNN_new = vmap(compute_one_grid_JNN, in_axes=(0, 0))(new_r_up_carts_arr, new_r_dn_carts_arr)
        JNN_ratio = jnp.exp(JNN_new - JNN_old)
        J_ratio *= jnp.ravel(JNN_ratio)

    return J_ratio


def _compute_ratio_Jastrow_part_split_spin(
    jastrow_data: Jastrow_data,
    old_r_up_carts: jax.Array,
    old_r_dn_carts: jax.Array,
    new_r_up_shifted: jax.Array,
    new_r_dn_shifted: jax.Array,
    j3_state: "Jastrow_three_body_streaming_state | None" = None,
) -> jax.Array:
    r"""Jastrow ratio for a block-structured mesh where up and dn electrons move separately.

    Avoids computing spin-block contributions for the unchanged spin.  Compared
    with ``_compute_ratio_Jastrow_part_rank1_update`` called on the concatenated
    (G_up + G_dn) array, this evaluates only the up-spin J3 formula for the
    ``G_up`` configs and only the dn-spin J3 formula for the ``G_dn`` configs,
    replacing the ``jnp.where`` merge with separate block computations.  For
    J3 the old AOs of the moved electron are obtained by column-slicing the
    already-computed ``aos_up_old`` / ``aos_dn_old`` matrices, avoiding two
    extra ``compute_orb_api`` calls.

    When called from the projection streaming path, the caller may pass
    ``j3_state`` to skip recomputing ``aos_*_old`` and the ``W``/``U``/cross_vec
    products — see ``_compute_ratio_Jastrow_part_rank1_update`` for the exact
    correspondence.

    Args:
        jastrow_data: Active Jastrow components.
        old_r_up_carts: Reference up-spin coordinates ``(N_up, 3)``.
        old_r_dn_carts: Reference dn-spin coordinates ``(N_dn, 3)``.
        new_r_up_shifted: Up-block proposed coords ``(G_up, N_up, 3)``.  Exactly one
            up electron differs from ``old_r_up_carts`` per config.
        new_r_dn_shifted: Dn-block proposed coords ``(G_dn, N_dn, 3)``.  Exactly one
            dn electron differs from ``old_r_dn_carts`` per config.

    Returns:
        jax.Array: Concatenated Jastrow ratios ``(G_up + G_dn,)`` (includes ``exp``).

    Warning:
        Each config in ``new_r_up_shifted`` must differ from ``old_r_up_carts``
        in **exactly one up electron**, and each config in ``new_r_dn_shifted``
        must differ from ``old_r_dn_carts`` in **exactly one dn electron**.
        The moved electron is located via ``jnp.argmax`` on the change mask;
        if two or more electrons differ in the same config, only the first is
        detected and the ratio is silently incorrect.  This function is intended
        exclusively for the block-structured non-local ECP grids produced by
        the MCMC loop.
    """
    # Forward old/new r_up/dn_carts as-is (Principle 3a — no parameter rebind).
    # Module-level forwards (compute_Jastrow_one_body, compute_orb_api,
    # _compute_ratio_Jastrow_part_rank1_update, NN_Jastrow.apply) handle their
    # own use-site casts. Inline diffs in the local J2 _safe_norm closure cast
    # operands to the jastrow_ratio zone at the use site (Principle 3b).
    dtype_jnp = get_dtype_jnp("jastrow_ratio")

    num_up = old_r_up_carts.shape[0]
    num_dn = old_r_dn_carts.shape[0]

    # Degenerate cases fall back to the general function on empty slices.
    if num_up == 0 or num_dn == 0:
        g_up = new_r_up_shifted.shape[0]
        g_dn = new_r_dn_shifted.shape[0]
        combined_up = jnp.concatenate(
            [new_r_up_shifted, jnp.broadcast_to(old_r_up_carts[None], (g_dn, num_up, 3))],
            axis=0,
        )
        combined_dn = jnp.concatenate(
            [jnp.broadcast_to(old_r_dn_carts[None], (g_up, num_dn, 3)), new_r_dn_shifted],
            axis=0,
        )
        return _compute_ratio_Jastrow_part_rank1_update(
            jastrow_data,
            old_r_up_carts,
            old_r_dn_carts,
            combined_up,
            combined_dn,
            j3_state=j3_state,
        )

    g_up = new_r_up_shifted.shape[0]
    g_dn = new_r_dn_shifted.shape[0]

    # Precompute moved-electron indices for both blocks.
    delta_up_block = new_r_up_shifted - old_r_up_carts[None]  # (G_up, N_up, 3)
    delta_dn_block = new_r_dn_shifted - old_r_dn_carts[None]  # (G_dn, N_dn, 3)
    idx_up_block = jnp.argmax(jnp.any(delta_up_block != 0.0, axis=2).astype(jnp.int32), axis=1)  # (G_up,)
    idx_dn_block = jnp.argmax(jnp.any(delta_dn_block != 0.0, axis=2).astype(jnp.int32), axis=1)  # (G_dn,)

    # New position of the moved electron in each block.
    r_up_moved = jnp.take_along_axis(new_r_up_shifted, idx_up_block[:, None, None], axis=1).reshape(g_up, 3)  # (G_up, 3)
    r_dn_moved = jnp.take_along_axis(new_r_dn_shifted, idx_dn_block[:, None, None], axis=1).reshape(g_dn, 3)  # (G_dn, 3)

    # Old position of the moved electron in each block.
    r_up_old_moved = old_r_up_carts[idx_up_block]  # (G_up, 3)
    r_dn_old_moved = old_r_dn_carts[idx_dn_block]  # (G_dn, 3)

    J_up = jnp.ones(g_up, dtype=dtype_jnp)
    J_dn = jnp.ones(g_dn, dtype=dtype_jnp)

    # ── J1 part ──────────────────────────────────────────────────────────────
    if jastrow_data.jastrow_one_body_data is not None:
        j1_data = jastrow_data.jastrow_one_body_data

        # UP block: only the moved up electron contributes to the J1 change.
        def compute_J1_up_one(r_up_new: jax.Array, r_up_old: jax.Array) -> jax.Array:
            j1_new = compute_Jastrow_one_body(j1_data, jnp.expand_dims(r_up_new, axis=0), jnp.zeros((0, 3), dtype=dtype_jnp))
            j1_old = compute_Jastrow_one_body(j1_data, jnp.expand_dims(r_up_old, axis=0), jnp.zeros((0, 3), dtype=dtype_jnp))
            return jnp.exp(j1_new - j1_old)

        J1_up_block = vmap(compute_J1_up_one)(r_up_moved, r_up_old_moved)  # (G_up,)
        J_up = J_up * jnp.ravel(J1_up_block)

        # DN block: only the moved dn electron contributes.
        def compute_J1_dn_one(r_dn_new: jax.Array, r_dn_old: jax.Array) -> jax.Array:
            j1_new = compute_Jastrow_one_body(j1_data, jnp.zeros((0, 3), dtype=dtype_jnp), jnp.expand_dims(r_dn_new, axis=0))
            j1_old = compute_Jastrow_one_body(j1_data, jnp.zeros((0, 3), dtype=dtype_jnp), jnp.expand_dims(r_dn_old, axis=0))
            return jnp.exp(j1_new - j1_old)

        J1_dn_block = vmap(compute_J1_dn_one)(r_dn_moved, r_dn_old_moved)  # (G_dn,)
        J_dn = J_dn * jnp.ravel(J1_dn_block)

    # ── J2 part ──────────────────────────────────────────────────────────────
    if jastrow_data.jastrow_two_body_data is not None:
        j2_param = jastrow_data.jastrow_two_body_data.jastrow_2b_param
        _j2_type_split = jastrow_data.jastrow_two_body_data.jastrow_2b_type

        def _safe_norm(diff):
            # Cast diff (reconstructed in caller-supplied precision by the
            # caller, e.g. `pos1 - pos2`) to the jastrow_ratio zone at the use
            # site (Principle 3b). New variable name `d` keeps the parameter
            # `diff` itself frozen (Principle 3a).
            d = diff.astype(dtype_jnp)
            sq = jnp.sum(d**2, axis=-1)
            return jnp.where(sq > 0, jnp.sqrt(jnp.where(sq > 0, sq, jnp.ones_like(sq))), jnp.zeros_like(sq))

        if _j2_type_split == "pade":

            def _j2_from_dist_split(dist, param):
                return dist / 2.0 * (1.0 + param * dist) ** (-1.0)
        else:

            def _j2_from_dist_split(dist, param):
                return 1.0 / (2.0 * param) * (1.0 - jnp.exp(-param * dist))

        def _pairwise_sums(pos1: jax.Array, pos2: jax.Array) -> jax.Array:
            """Row-wise sum of two-body terms: pos1 -> all of pos2."""
            dists = _safe_norm(pos1[:, None, :] - pos2[None, :, :])
            return jnp.sum(_j2_from_dist_split(dists, j2_param), axis=1)

        J2_sum_up_up = _pairwise_sums(old_r_up_carts, old_r_up_carts)  # (N_up,)
        J2_sum_up_dn = _pairwise_sums(old_r_up_carts, old_r_dn_carts)  # (N_up,)
        J2_sum_dn_dn = _pairwise_sums(old_r_dn_carts, old_r_dn_carts)  # (N_dn,)
        J2_sum_dn_up = _pairwise_sums(old_r_dn_carts, old_r_up_carts)  # (N_dn,)

        # UP block: moved up electron interacts with all old up and dn electrons.
        dists_up_up_new = _safe_norm(r_up_moved[:, None, :] - old_r_up_carts[None, :, :])  # (G_up, N_up)
        J2_up_up_new = jnp.sum(_j2_from_dist_split(dists_up_up_new, j2_param), axis=1)  # (G_up,)
        # The sum above includes the self-term f(r_new, r_old[idx]) but the correct self-term is
        # f(r_new, r_new) = 0.  Subtract the spurious contribution.
        dists_self_up = _safe_norm(r_up_moved - r_up_old_moved)  # (G_up,)
        J2_up_up_new = J2_up_up_new - _j2_from_dist_split(dists_self_up, j2_param)
        J2_up_up_old = J2_sum_up_up[idx_up_block]  # (G_up,)  self-term is f(r_old,r_old)=0 already
        dists_up_dn_new = _safe_norm(r_up_moved[:, None, :] - old_r_dn_carts[None, :, :])  # (G_up, N_dn)
        J2_up_dn_new = jnp.sum(_j2_from_dist_split(dists_up_dn_new, j2_param), axis=1)  # (G_up,)
        J2_up_dn_old = J2_sum_up_dn[idx_up_block]  # (G_up,)
        J_up = J_up * jnp.exp(J2_up_dn_new - J2_up_dn_old + J2_up_up_new - J2_up_up_old)

        # DN block: moved dn electron interacts with all old dn and up electrons.
        dists_dn_dn_new = _safe_norm(r_dn_moved[:, None, :] - old_r_dn_carts[None, :, :])  # (G_dn, N_dn)
        J2_dn_dn_new = jnp.sum(_j2_from_dist_split(dists_dn_dn_new, j2_param), axis=1)  # (G_dn,)
        # Same self-term correction for down block.
        dists_self_dn = _safe_norm(r_dn_moved - r_dn_old_moved)  # (G_dn,)
        J2_dn_dn_new = J2_dn_dn_new - _j2_from_dist_split(dists_self_dn, j2_param)
        J2_dn_dn_old = J2_sum_dn_dn[idx_dn_block]  # (G_dn,)
        dists_dn_up_new = _safe_norm(r_dn_moved[:, None, :] - old_r_up_carts[None, :, :])  # (G_dn, N_up)
        J2_dn_up_new = jnp.sum(_j2_from_dist_split(dists_dn_up_new, j2_param), axis=1)  # (G_dn,)
        J2_dn_up_old = J2_sum_dn_up[idx_dn_block]  # (G_dn,)
        J_dn = J_dn * jnp.exp(J2_dn_up_new - J2_dn_up_old + J2_dn_dn_new - J2_dn_dn_old)

    # ── J3 part ──────────────────────────────────────────────────────────────
    if jastrow_data.jastrow_three_body_data is not None:
        j3d = jastrow_data.jastrow_three_body_data
        j3_mat = j3d._j_matrix_jnp[:, :-1]  # (n_ao, n_ao)
        j1_vec = j3d._j_matrix_jnp[:, -1]  # (n_ao,)

        # Old AOs evaluated once; column slices give old AO at each moved position.
        # When the streaming state is provided, all four matmuls and both
        # cross_vec products come from the cache (Python-static dispatch).
        if j3_state is None:
            aos_up_old = jnp.array(j3d.compute_orb_api(j3d.orb_data, old_r_up_carts), dtype=dtype_jnp)  # (n_ao, N_up)
            aos_dn_old = jnp.array(j3d.compute_orb_api(j3d.orb_data, old_r_dn_carts), dtype=dtype_jnp)  # (n_ao, N_dn)
            # Precompute constant products (shared between blocks).
            W_up = jnp.dot(j3_mat, aos_up_old)  # (n_ao, N_up)
            U_up = jnp.dot(aos_up_old.T, j3_mat)  # (N_up, n_ao)
            W_dn = jnp.dot(j3_mat, aos_dn_old)  # (n_ao, N_dn)
            U_dn = jnp.dot(aos_dn_old.T, j3_mat)  # (N_dn, n_ao)
            dn_cross_vec = j3_mat @ jnp.sum(aos_dn_old, axis=1)  # (n_ao,): UP cross term constant
            up_cross_vec = jnp.sum(aos_up_old, axis=1) @ j3_mat  # (n_ao,): DN cross term constant
        else:
            aos_up_old = j3_state.aos_up.astype(dtype_jnp)
            aos_dn_old = j3_state.aos_dn.astype(dtype_jnp)
            W_up = j3_state.j3_mat_aos_up.astype(dtype_jnp)
            W_dn = j3_state.j3_mat_aos_dn.astype(dtype_jnp)
            U_up = j3_state.j3_mat_T_aos_up.astype(dtype_jnp).T
            U_dn = j3_state.j3_mat_T_aos_dn.astype(dtype_jnp).T
            dn_cross_vec = jnp.sum(W_dn, axis=1)
            up_cross_vec = jnp.sum(j3_state.j3_mat_T_aos_up.astype(dtype_jnp), axis=1)

        # ── UP BLOCK ─────────────────────────────────────────────────────────
        # New AOs at the moved up-electron positions; old AOs by column-slice.
        aos_up_new_moved = jnp.array(j3d.compute_orb_api(j3d.orb_data, r_up_moved), dtype=dtype_jnp)  # (n_ao, G_up)
        aos_up_old_moved = aos_up_old[:, idx_up_block]  # (n_ao, G_up)
        aos_p_up = aos_up_new_moved - aos_up_old_moved  # (n_ao, G_up)

        term1_up = j1_vec @ aos_p_up  # (G_up,)
        V_up_block = jnp.dot(aos_p_up.T, W_up)  # (G_up, N_up)
        P_up_block = jnp.dot(U_up, aos_p_up)  # (N_up, G_up)
        Q_up_c = (idx_up_block[:, None] < jnp.arange(num_up)[None, :]).astype(dtype_jnp)  # (G_up, N_up)
        Q_up_r = (idx_up_block[:, None] > jnp.arange(num_up)[None, :]).astype(dtype_jnp)  # (G_up, N_up)
        term2_up = jnp.sum(V_up_block * Q_up_c, axis=1)  # (G_up,)
        term3_up = jnp.sum(P_up_block.T * Q_up_r, axis=1)  # (G_up,)
        term4_up = dn_cross_vec @ aos_p_up  # (G_up,)
        J_up = J_up * jnp.exp(term1_up + term2_up + term3_up + term4_up)

        # ── DN BLOCK ─────────────────────────────────────────────────────────
        # New AOs at the moved dn-electron positions; old AOs by column-slice.
        aos_dn_new_moved = jnp.array(j3d.compute_orb_api(j3d.orb_data, r_dn_moved), dtype=dtype_jnp)  # (n_ao, G_dn)
        aos_dn_old_moved = aos_dn_old[:, idx_dn_block]  # (n_ao, G_dn)
        aos_p_dn = aos_dn_new_moved - aos_dn_old_moved  # (n_ao, G_dn)

        term1_dn = j1_vec @ aos_p_dn  # (G_dn,)
        V_dn_block = jnp.dot(aos_p_dn.T, W_dn)  # (G_dn, N_dn)
        P_dn_block = jnp.dot(U_dn, aos_p_dn)  # (N_dn, G_dn)
        Q_dn_c = (idx_dn_block[:, None] < jnp.arange(num_dn)[None, :]).astype(dtype_jnp)  # (G_dn, N_dn)
        Q_dn_r = (idx_dn_block[:, None] > jnp.arange(num_dn)[None, :]).astype(dtype_jnp)  # (G_dn, N_dn)
        term2_dn = jnp.sum(V_dn_block * Q_dn_c, axis=1)  # (G_dn,)
        term3_dn = jnp.sum(P_dn_block.T * Q_dn_r, axis=1)  # (G_dn,)
        term4_dn = up_cross_vec @ aos_p_dn  # (G_dn,)
        J_dn = J_dn * jnp.exp(term1_dn + term2_dn + term3_dn + term4_dn)

    # ── JNN part ─────────────────────────────────────────────────────────────
    if jastrow_data.jastrow_nn_data is not None:
        nn = jastrow_data.jastrow_nn_data
        if nn.structure_data is None:
            raise ValueError("NN_Jastrow_data.structure_data must be set to evaluate NN J3.")

        R_n = nn.structure_data._positions_cart_jnp.astype(dtype_jnp)
        Z_n = jnp.asarray(nn.structure_data.atomic_numbers)

        def compute_one_grid_JNN_split(r_up: jax.Array, r_dn: jax.Array) -> jax.Array:
            return nn.nn_def.apply({"params": nn.params}, r_up, r_dn, R_n, Z_n)

        JNN_old = compute_one_grid_JNN_split(old_r_up_carts, old_r_dn_carts)

        # UP block: only r_up changes; dn stays at old positions.
        JNN_new_up = vmap(compute_one_grid_JNN_split, in_axes=(0, None))(new_r_up_shifted, old_r_dn_carts)
        J_up = J_up * jnp.exp(JNN_new_up - JNN_old)

        # DN block: only r_dn changes; up stays at old positions.
        JNN_new_dn = vmap(compute_one_grid_JNN_split, in_axes=(None, 0))(old_r_up_carts, new_r_dn_shifted)
        J_dn = J_dn * jnp.exp(JNN_new_dn - JNN_old)

    return jnp.concatenate([J_up, J_dn])


def _compute_ratio_Jastrow_part_debug(
    jastrow_data: Jastrow_data,
    old_r_up_carts: npt.NDArray[np.float64],
    old_r_dn_carts: npt.NDArray[np.float64],
    new_r_up_carts_arr: npt.NDArray[np.float64],
    new_r_dn_carts_arr: npt.NDArray[np.float64],
) -> npt.NDArray:
    """See _api method."""
    dtype_np = get_dtype_np("jastrow_ratio")
    return np.array(
        [
            np.exp(compute_Jastrow_part(jastrow_data, new_r_up_carts, new_r_dn_carts))
            / np.exp(compute_Jastrow_part(jastrow_data, old_r_up_carts, old_r_dn_carts))
            for new_r_up_carts, new_r_dn_carts in zip(new_r_up_carts_arr, new_r_dn_carts_arr, strict=True)
        ],
        dtype=dtype_np,
    )


@jit
def compute_grads_and_laplacian_Jastrow_part(
    jastrow_data: Jastrow_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    r"""Per-electron gradients and Laplacians of the full Jastrow :math:`J`.

    Analytic paths are used for J1/J2/J3 when available; the NN three-body
    term (if present) is handled via autodiff. Values are returned per electron
    (not summed) to match downstream kinetic-energy estimators.

    Args:
        jastrow_data: Active Jastrow components.
        r_up_carts: Spin-up electron coordinates with shape ``(N_up, 3)``.
        r_dn_carts: Spin-down electron coordinates with shape ``(N_dn, 3)``.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            Gradients for up/down electrons with shapes ``(N_up, 3)`` and ``(N_dn, 3)``
            and Laplacians for up/down electrons with shapes ``(N_up,)`` and ``(N_dn,)``.
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    r_up = jnp.asarray(r_up_carts, dtype=dtype_jnp)
    r_dn = jnp.asarray(r_dn_carts, dtype=dtype_jnp)

    grad_J_up = jnp.zeros_like(r_up)
    grad_J_dn = jnp.zeros_like(r_dn)
    lap_J_up = jnp.zeros((r_up.shape[0],), dtype=dtype_jnp)
    lap_J_dn = jnp.zeros((r_dn.shape[0],), dtype=dtype_jnp)

    # one-body (analytic)
    if jastrow_data.jastrow_one_body_data is not None:
        grad_J1_up, grad_J1_dn, lap_J1_up, lap_J1_dn = compute_grads_and_laplacian_Jastrow_one_body(
            jastrow_data.jastrow_one_body_data,
            r_up_carts,
            r_dn_carts,
        )
        grad_J_up = grad_J_up + grad_J1_up
        grad_J_dn = grad_J_dn + grad_J1_dn
        lap_J_up = lap_J_up + lap_J1_up
        lap_J_dn = lap_J_dn + lap_J1_dn

    # two-body (analytic)
    if jastrow_data.jastrow_two_body_data is not None:
        grad_J2_up, grad_J2_dn, lap_J2_up, lap_J2_dn = compute_grads_and_laplacian_Jastrow_two_body(
            jastrow_data.jastrow_two_body_data,
            r_up_carts,
            r_dn_carts,
        )
        grad_J_up = grad_J_up + grad_J2_up
        grad_J_dn = grad_J_dn + grad_J2_dn
        lap_J_up = lap_J_up + lap_J2_up
        lap_J_dn = lap_J_dn + lap_J2_dn

    # three-body (analytic)
    if jastrow_data.jastrow_three_body_data is not None:
        grad_J3_up, grad_J3_dn, lap_J3_up, lap_J3_dn = compute_grads_and_laplacian_Jastrow_three_body(
            jastrow_data.jastrow_three_body_data,
            r_up_carts,
            r_dn_carts,
        )
        grad_J_up = grad_J_up + grad_J3_up
        grad_J_dn = grad_J_dn + grad_J3_dn
        lap_J_up = lap_J_up + lap_J3_up
        lap_J_dn = lap_J_dn + lap_J3_dn

    # NN three-body (autodiff)
    if jastrow_data.jastrow_nn_data is not None:
        nn3 = jastrow_data.jastrow_nn_data
        if nn3.structure_data is None:
            raise ValueError("NN_Jastrow_data.structure_data must be set to evaluate NN J3.")

        r_up_carts_jnp = jnp.asarray(r_up_carts, dtype=dtype_jnp)
        r_dn_carts_jnp = jnp.asarray(r_dn_carts, dtype=dtype_jnp)
        R_n = nn3.structure_data._positions_cart_jnp.astype(dtype_jnp)
        Z_n = jnp.asarray(nn3.structure_data.atomic_numbers, dtype=dtype_jnp)

        def _compute_Jastrow_nn_only(r_up, r_dn):
            nn_params = jax.tree_util.tree_map(
                lambda x: x.astype(dtype_jnp) if hasattr(x, "dtype") and x.dtype.kind == "f" else x, nn3.params
            )
            return nn3.nn_def.apply({"params": nn_params}, r_up, r_dn, R_n, Z_n)

        grad_JNN_up = grad(_compute_Jastrow_nn_only, argnums=0)(r_up_carts_jnp, r_dn_carts_jnp)
        grad_JNN_dn = grad(_compute_Jastrow_nn_only, argnums=1)(r_up_carts_jnp, r_dn_carts_jnp)

        # Compute per-electron Laplacian via forward-over-reverse (diagonal Hessian only).
        # This produces an O(n) computation graph instead of O(n²) from hessian(),
        # which significantly reduces the XLA kernel size when grad(compute_local_energy)
        # differentiates through the kinetic energy (i.e. under 3rd-order AD).
        def _lap_jvp(f_r, r):
            """Sum of diagonal Hessian (Laplacian) per electron via jvp(grad)."""
            n_elec, n_coord = r.shape
            n = n_elec * n_coord
            g_r = grad(f_r)

            def diag_one(e_flat):
                e = e_flat.reshape(r.shape)
                _, jvp_val = jax.jvp(g_r, (r,), (e,))
                return jnp.sum(jvp_val * e)

            basis = jnp.eye(n, dtype=r.dtype)
            diags = jax.vmap(diag_one)(basis)
            return diags.reshape(n_elec, n_coord).sum(axis=-1)

        lap_JNN_up = _lap_jvp(lambda r: _compute_Jastrow_nn_only(r, r_dn_carts_jnp), r_up_carts_jnp)
        lap_JNN_dn = _lap_jvp(lambda r: _compute_Jastrow_nn_only(r_up_carts_jnp, r), r_dn_carts_jnp)

        grad_J_up = grad_J_up + grad_JNN_up
        grad_J_dn = grad_J_dn + grad_JNN_dn
        lap_J_up = lap_J_up + lap_JNN_up
        lap_J_dn = lap_J_dn + lap_JNN_dn

    return grad_J_up, grad_J_dn, lap_J_up, lap_J_dn


@jit
def _compute_grads_and_laplacian_Jastrow_part_auto(
    jastrow_data: Jastrow_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Function for computing grads and laplacians with a given Jastrow_data.

    The method is for computing the gradients and the sum of laplacians of J at (r_up_carts, r_dn_carts)
    with a given Jastrow_data.

    Args:
        jastrow_data (Jastrow_data): an instance of Jastrow_two_body_data class
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        the gradients(x,y,z) of J and the sum of laplacians of J at (r_up_carts, r_dn_carts).
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    r_up_carts_jnp = jnp.array(r_up_carts, dtype=dtype_jnp)
    r_dn_carts_jnp = jnp.array(r_dn_carts, dtype=dtype_jnp)

    grad_J_up = jnp.zeros_like(r_up_carts_jnp)
    grad_J_dn = jnp.zeros_like(r_dn_carts_jnp)
    lap_J_up = jnp.zeros((r_up_carts_jnp.shape[0],), dtype=dtype_jnp)
    lap_J_dn = jnp.zeros((r_dn_carts_jnp.shape[0],), dtype=dtype_jnp)

    # one-body
    if jastrow_data.jastrow_one_body_data is not None:
        grad_J1_up = grad(compute_Jastrow_one_body, argnums=1)(
            jastrow_data.jastrow_one_body_data, r_up_carts_jnp, r_dn_carts_jnp
        )
        grad_J1_dn = grad(compute_Jastrow_one_body, argnums=2)(
            jastrow_data.jastrow_one_body_data, r_up_carts_jnp, r_dn_carts_jnp
        )

        hessian_J1_up = hessian(compute_Jastrow_one_body, argnums=1)(
            jastrow_data.jastrow_one_body_data, r_up_carts_jnp, r_dn_carts_jnp
        )
        laplacian_J1_up = jnp.einsum("ijij->i", hessian_J1_up)

        hessian_J1_dn = hessian(compute_Jastrow_one_body, argnums=2)(
            jastrow_data.jastrow_one_body_data, r_up_carts_jnp, r_dn_carts_jnp
        )
        laplacian_J1_dn = jnp.einsum("ijij->i", hessian_J1_dn)

        grad_J_up = grad_J_up + grad_J1_up
        grad_J_dn = grad_J_dn + grad_J1_dn
        lap_J_up = lap_J_up + laplacian_J1_up
        lap_J_dn = lap_J_dn + laplacian_J1_dn

    # two-body
    if jastrow_data.jastrow_two_body_data is not None:
        grad_J2_up, grad_J2_dn, lap_J2_up, lap_J2_dn = _compute_grads_and_laplacian_Jastrow_two_body_auto(
            jastrow_data.jastrow_two_body_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
        )
        grad_J_up = grad_J_up + grad_J2_up
        grad_J_dn = grad_J_dn + grad_J2_dn
        lap_J_up = lap_J_up + lap_J2_up
        lap_J_dn = lap_J_dn + lap_J2_dn

    # three-body
    if jastrow_data.jastrow_three_body_data is not None:
        grad_J3_up_add, grad_J3_dn_add, lap_J3_up_add, lap_J3_dn_add = _compute_grads_and_laplacian_Jastrow_three_body_auto(
            jastrow_data.jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        grad_J_up = grad_J_up + grad_J3_up_add
        grad_J_dn = grad_J_dn + grad_J3_dn_add
        lap_J_up = lap_J_up + lap_J3_up_add
        lap_J_dn = lap_J_dn + lap_J3_dn_add

    # three-body (NN)
    if jastrow_data.jastrow_nn_data is not None:
        nn3 = jastrow_data.jastrow_nn_data
        if nn3.structure_data is None:
            raise ValueError("NN_Jastrow_data.structure_data must be set to evaluate NN J3.")

        R_n = nn3.structure_data._positions_cart_jnp.astype(dtype_jnp)
        Z_n = jnp.asarray(nn3.structure_data.atomic_numbers, dtype=dtype_jnp)

        def _compute_Jastrow_nn_only(r_up, r_dn):
            nn_params = jax.tree_util.tree_map(
                lambda x: x.astype(dtype_jnp) if hasattr(x, "dtype") and x.dtype.kind == "f" else x, nn3.params
            )
            return nn3.nn_def.apply({"params": nn_params}, r_up, r_dn, R_n, Z_n)

        grad_JNN_up = grad(_compute_Jastrow_nn_only, argnums=0)(r_up_carts_jnp, r_dn_carts_jnp)
        grad_JNN_dn = grad(_compute_Jastrow_nn_only, argnums=1)(r_up_carts_jnp, r_dn_carts_jnp)

        # Compute per-electron Laplacian via forward-over-reverse (diagonal Hessian only).
        # See compute_grads_and_laplacian_Jastrow_part for rationale.
        def _lap_jvp(f_r, r):
            """Sum of diagonal Hessian (Laplacian) per electron via jvp(grad)."""
            n_elec, n_coord = r.shape
            n = n_elec * n_coord
            g_r = grad(f_r)

            def diag_one(e_flat):
                e = e_flat.reshape(r.shape)
                _, jvp_val = jax.jvp(g_r, (r,), (e,))
                return jnp.sum(jvp_val * e)

            basis = jnp.eye(n, dtype=r.dtype)
            diags = jax.vmap(diag_one)(basis)
            return diags.reshape(n_elec, n_coord).sum(axis=-1)

        lap_JNN_up = _lap_jvp(lambda r: _compute_Jastrow_nn_only(r, r_dn_carts_jnp), r_up_carts_jnp)
        lap_JNN_dn = _lap_jvp(lambda r: _compute_Jastrow_nn_only(r_up_carts_jnp, r), r_dn_carts_jnp)

        grad_J_up = grad_J_up + grad_JNN_up
        grad_J_dn = grad_J_dn + grad_JNN_dn
        lap_J_up = lap_J_up + lap_JNN_up
        lap_J_dn = lap_J_dn + lap_JNN_dn

    return grad_J_up, grad_J_dn, lap_J_up, lap_J_dn


def _compute_grads_and_laplacian_Jastrow_part_debug(
    jastrow_data: Jastrow_data,
    r_up_carts: np.ndarray,
    r_dn_carts: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Numerical gradients and Laplacian for the full Jastrow factor.

    Uses central finite differences to approximate gradients and the
    sum of Laplacians of J at (r_up_carts, r_dn_carts).
    """
    dtype_np = get_dtype_np("jastrow_grad_lap")
    diff_h = 1.0e-5

    r_up_carts = np.array(r_up_carts, dtype=dtype_np)
    r_dn_carts = np.array(r_dn_carts, dtype=dtype_np)

    # grad up
    grad_x_up = []
    grad_y_up = []
    grad_z_up = []
    for r_i, _ in enumerate(r_up_carts):
        diff_p_x_r_up_carts = r_up_carts.copy()
        diff_p_y_r_up_carts = r_up_carts.copy()
        diff_p_z_r_up_carts = r_up_carts.copy()
        diff_p_x_r_up_carts[r_i][0] += diff_h
        diff_p_y_r_up_carts[r_i][1] += diff_h
        diff_p_z_r_up_carts[r_i][2] += diff_h

        J_p_x_up = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=diff_p_x_r_up_carts, r_dn_carts=r_dn_carts)
        J_p_y_up = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=diff_p_y_r_up_carts, r_dn_carts=r_dn_carts)
        J_p_z_up = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=diff_p_z_r_up_carts, r_dn_carts=r_dn_carts)

        diff_m_x_r_up_carts = r_up_carts.copy()
        diff_m_y_r_up_carts = r_up_carts.copy()
        diff_m_z_r_up_carts = r_up_carts.copy()
        diff_m_x_r_up_carts[r_i][0] -= diff_h
        diff_m_y_r_up_carts[r_i][1] -= diff_h
        diff_m_z_r_up_carts[r_i][2] -= diff_h

        J_m_x_up = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=diff_m_x_r_up_carts, r_dn_carts=r_dn_carts)
        J_m_y_up = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=diff_m_y_r_up_carts, r_dn_carts=r_dn_carts)
        J_m_z_up = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=diff_m_z_r_up_carts, r_dn_carts=r_dn_carts)

        grad_x_up.append((J_p_x_up - J_m_x_up) / (2.0 * diff_h))
        grad_y_up.append((J_p_y_up - J_m_y_up) / (2.0 * diff_h))
        grad_z_up.append((J_p_z_up - J_m_z_up) / (2.0 * diff_h))

    # grad dn
    grad_x_dn = []
    grad_y_dn = []
    grad_z_dn = []
    for r_i, _ in enumerate(r_dn_carts):
        diff_p_x_r_dn_carts = r_dn_carts.copy()
        diff_p_y_r_dn_carts = r_dn_carts.copy()
        diff_p_z_r_dn_carts = r_dn_carts.copy()
        diff_p_x_r_dn_carts[r_i][0] += diff_h
        diff_p_y_r_dn_carts[r_i][1] += diff_h
        diff_p_z_r_dn_carts[r_i][2] += diff_h

        J_p_x_dn = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up_carts, r_dn_carts=diff_p_x_r_dn_carts)
        J_p_y_dn = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up_carts, r_dn_carts=diff_p_y_r_dn_carts)
        J_p_z_dn = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up_carts, r_dn_carts=diff_p_z_r_dn_carts)

        diff_m_x_r_dn_carts = r_dn_carts.copy()
        diff_m_y_r_dn_carts = r_dn_carts.copy()
        diff_m_z_r_dn_carts = r_dn_carts.copy()
        diff_m_x_r_dn_carts[r_i][0] -= diff_h
        diff_m_y_r_dn_carts[r_i][1] -= diff_h
        diff_m_z_r_dn_carts[r_i][2] -= diff_h

        J_m_x_dn = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up_carts, r_dn_carts=diff_m_x_r_dn_carts)
        J_m_y_dn = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up_carts, r_dn_carts=diff_m_y_r_dn_carts)
        J_m_z_dn = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up_carts, r_dn_carts=diff_m_z_r_dn_carts)

        grad_x_dn.append((J_p_x_dn - J_m_x_dn) / (2.0 * diff_h))
        grad_y_dn.append((J_p_y_dn - J_m_y_dn) / (2.0 * diff_h))
        grad_z_dn.append((J_p_z_dn - J_m_z_dn) / (2.0 * diff_h))

    grad_J_up = np.array([grad_x_up, grad_y_up, grad_z_up], dtype=dtype_np).T
    grad_J_dn = np.array([grad_x_dn, grad_y_dn, grad_z_dn], dtype=dtype_np).T

    # laplacian
    diff_h2 = 1.0e-3
    J_ref = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)

    lap_J_up = np.zeros(len(r_up_carts), dtype=dtype_np)

    # laplacians up
    for r_i, _ in enumerate(r_up_carts):
        diff_p_x_r_up2_carts = r_up_carts.copy()
        diff_p_y_r_up2_carts = r_up_carts.copy()
        diff_p_z_r_up2_carts = r_up_carts.copy()
        diff_p_x_r_up2_carts[r_i][0] += diff_h2
        diff_p_y_r_up2_carts[r_i][1] += diff_h2
        diff_p_z_r_up2_carts[r_i][2] += diff_h2

        J_p_x_up2 = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=diff_p_x_r_up2_carts, r_dn_carts=r_dn_carts)
        J_p_y_up2 = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=diff_p_y_r_up2_carts, r_dn_carts=r_dn_carts)
        J_p_z_up2 = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=diff_p_z_r_up2_carts, r_dn_carts=r_dn_carts)

        diff_m_x_r_up2_carts = r_up_carts.copy()
        diff_m_y_r_up2_carts = r_up_carts.copy()
        diff_m_z_r_up2_carts = r_up_carts.copy()
        diff_m_x_r_up2_carts[r_i][0] -= diff_h2
        diff_m_y_r_up2_carts[r_i][1] -= diff_h2
        diff_m_z_r_up2_carts[r_i][2] -= diff_h2

        J_m_x_up2 = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=diff_m_x_r_up2_carts, r_dn_carts=r_dn_carts)
        J_m_y_up2 = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=diff_m_y_r_up2_carts, r_dn_carts=r_dn_carts)
        J_m_z_up2 = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=diff_m_z_r_up2_carts, r_dn_carts=r_dn_carts)

        gradgrad_x_up = (J_p_x_up2 + J_m_x_up2 - 2 * J_ref) / (diff_h2**2)
        gradgrad_y_up = (J_p_y_up2 + J_m_y_up2 - 2 * J_ref) / (diff_h2**2)
        gradgrad_z_up = (J_p_z_up2 + J_m_z_up2 - 2 * J_ref) / (diff_h2**2)

        lap_J_up[r_i] = gradgrad_x_up + gradgrad_y_up + gradgrad_z_up

    lap_J_dn = np.zeros(len(r_dn_carts), dtype=dtype_np)

    # laplacians dn
    for r_i, _ in enumerate(r_dn_carts):
        diff_p_x_r_dn2_carts = r_dn_carts.copy()
        diff_p_y_r_dn2_carts = r_dn_carts.copy()
        diff_p_z_r_dn2_carts = r_dn_carts.copy()
        diff_p_x_r_dn2_carts[r_i][0] += diff_h2
        diff_p_y_r_dn2_carts[r_i][1] += diff_h2
        diff_p_z_r_dn2_carts[r_i][2] += diff_h2

        J_p_x_dn2 = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up_carts, r_dn_carts=diff_p_x_r_dn2_carts)
        J_p_y_dn2 = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up_carts, r_dn_carts=diff_p_y_r_dn2_carts)
        J_p_z_dn2 = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up_carts, r_dn_carts=diff_p_z_r_dn2_carts)

        diff_m_x_r_dn2_carts = r_dn_carts.copy()
        diff_m_y_r_dn2_carts = r_dn_carts.copy()
        diff_m_z_r_dn2_carts = r_dn_carts.copy()
        diff_m_x_r_dn2_carts[r_i][0] -= diff_h2
        diff_m_y_r_dn2_carts[r_i][1] -= diff_h2
        diff_m_z_r_dn2_carts[r_i][2] -= diff_h2

        J_m_x_dn2 = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up_carts, r_dn_carts=diff_m_x_r_dn2_carts)
        J_m_y_dn2 = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up_carts, r_dn_carts=diff_m_y_r_dn2_carts)
        J_m_z_dn2 = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up_carts, r_dn_carts=diff_m_z_r_dn2_carts)

        gradgrad_x_dn = (J_p_x_dn2 + J_m_x_dn2 - 2 * J_ref) / (diff_h2**2)
        gradgrad_y_dn = (J_p_y_dn2 + J_m_y_dn2 - 2 * J_ref) / (diff_h2**2)
        gradgrad_z_dn = (J_p_z_dn2 + J_m_z_dn2 - 2 * J_ref) / (diff_h2**2)

        lap_J_dn[r_i] = gradgrad_x_dn + gradgrad_y_dn + gradgrad_z_dn

    return grad_J_up, grad_J_dn, lap_J_up, lap_J_dn


@jit
def _compute_grads_and_laplacian_Jastrow_two_body_auto(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Function for computing grads and laplacians with a given Jastrow_two_body_data.

    The method is for computing the gradients and the sum of laplacians of J at (r_up_carts, r_dn_carts)
    with a given Jastrow_two_body_data.

    Args:
        jastrow_two_body_data (Jastrow_two_body_data): an instance of Jastrow_two_body_data class
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        the gradients(x,y,z) of J(twobody) and the sum of laplacians of J(twobody) at (r_up_carts, r_dn_carts).
    """
    # grad_J2_up, grad_J2_dn, sum_laplacian_J2 = (
    #    compute_grads_and_laplacian_Jastrow_two_body_debug(
    #        jastrow_two_body_data, r_up_carts, r_dn_carts
    #    )
    # )
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    r_up_carts = jnp.array(r_up_carts, dtype=dtype_jnp)
    r_dn_carts = jnp.array(r_dn_carts, dtype=dtype_jnp)

    # compute grad
    grad_J2_up = grad(compute_Jastrow_two_body, argnums=1)(jastrow_two_body_data, r_up_carts, r_dn_carts)

    grad_J2_dn = grad(compute_Jastrow_two_body, argnums=2)(jastrow_two_body_data, r_up_carts, r_dn_carts)

    # compute laplacians
    hessian_J2_up = hessian(compute_Jastrow_two_body, argnums=1)(jastrow_two_body_data, r_up_carts, r_dn_carts)
    laplacian_J2_up = jnp.einsum("ijij->i", hessian_J2_up)

    hessian_J2_dn = hessian(compute_Jastrow_two_body, argnums=2)(jastrow_two_body_data, r_up_carts, r_dn_carts)
    laplacian_J2_dn = jnp.einsum("ijij->i", hessian_J2_dn)

    return grad_J2_up, grad_J2_dn, laplacian_J2_up, laplacian_J2_dn


@jit
def compute_grads_and_laplacian_Jastrow_two_body(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Analytic gradients and Laplacians for the two-body Jastrow.

    Supports both ``'pade'`` and ``'exp'`` functional forms, selected via
    ``jastrow_two_body_data.jastrow_2b_type``. Returns per-electron quantities (not summed).

    Args:
        jastrow_two_body_data: Two-body Jastrow parameter container.
        r_up_carts: Spin-up electron coordinates with shape ``(N_up, 3)``.
        r_dn_carts: Spin-down electron coordinates with shape ``(N_dn, 3)``.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            Gradients for up/down electrons with shapes ``(N_up, 3)`` and ``(N_dn, 3)``,
            Laplacians for up/down electrons with shapes ``(N_up,)`` and ``(N_dn,)``.
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    a = jnp.asarray(jastrow_two_body_data.jastrow_2b_param, dtype=dtype_jnp)
    eps = jnp.asarray(EPS_safe_distance, dtype=dtype_jnp)

    r_up = jnp.asarray(r_up_carts, dtype=dtype_jnp)
    r_dn = jnp.asarray(r_dn_carts, dtype=dtype_jnp)

    num_up = r_up.shape[0]
    num_dn = r_dn.shape[0]

    grad_up = jnp.zeros_like(r_up)
    grad_dn = jnp.zeros_like(r_dn)
    lap_up = jnp.zeros((num_up,), dtype=dtype_jnp)
    lap_dn = jnp.zeros((num_dn,), dtype=dtype_jnp)

    j2b_type = jastrow_two_body_data.jastrow_2b_type

    if j2b_type == "pade":
        # f(r) = r / (2*(1 + a*r))
        # f'(r) = 1 / (2*(1+a*r)^2)
        # f''(r) = - a / ((1+a*r)^3)
        def pair_terms(diff):
            r = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
            r = jnp.maximum(r, eps)
            denom = 1.0 + a * r
            f_prime = 0.5 / (denom * denom)
            grad_coeff = f_prime / r
            lap = -a / (denom * denom * denom) + (2.0 * f_prime) / r
            return grad_coeff[..., None] * diff, lap

    elif j2b_type == "exp":
        # f(r) = 1/(2a) * (1 - exp(-a*r))
        # f'(r) = (1/2) * exp(-a*r)
        # f''(r) = -(a/2) * exp(-a*r)
        def pair_terms(diff):
            r = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
            r = jnp.maximum(r, eps)
            exp_term = jnp.exp(-a * r)
            f_prime = 0.5 * exp_term
            grad_coeff = f_prime / r
            lap = -(a / 2.0) * exp_term + (2.0 * f_prime) / r
            return grad_coeff[..., None] * diff, lap

    else:
        raise ValueError(f"Unknown jastrow_2b_type: {j2b_type}")

    # up-up pairs (i<j)
    if num_up > 1:
        idx_i, idx_j = jnp.triu_indices(num_up, k=1)
        diff_up = r_up[idx_i] - r_up[idx_j]
        grad_pair, lap_pair = pair_terms(diff_up)
        grad_up = grad_up.at[idx_i].add(grad_pair)
        grad_up = grad_up.at[idx_j].add(-grad_pair)
        lap_up = lap_up.at[idx_i].add(lap_pair)
        lap_up = lap_up.at[idx_j].add(lap_pair)

    # dn-dn pairs (i<j)
    if num_dn > 1:
        idx_i, idx_j = jnp.triu_indices(num_dn, k=1)
        diff_dn = r_dn[idx_i] - r_dn[idx_j]
        grad_pair, lap_pair = pair_terms(diff_dn)
        grad_dn = grad_dn.at[idx_i].add(grad_pair)
        grad_dn = grad_dn.at[idx_j].add(-grad_pair)
        lap_dn = lap_dn.at[idx_i].add(lap_pair)
        lap_dn = lap_dn.at[idx_j].add(lap_pair)

    # up-dn pairs (all combinations)
    if (num_up > 0) and (num_dn > 0):
        diff_ud = r_up[:, None, :] - r_dn[None, :, :]
        grad_pair, lap_pair = pair_terms(diff_ud)
        grad_up = grad_up + jnp.sum(grad_pair, axis=1)
        grad_dn = grad_dn - jnp.sum(grad_pair, axis=0)
        lap_up = lap_up + jnp.sum(lap_pair, axis=1)
        lap_dn = lap_dn + jnp.sum(lap_pair, axis=0)

    return grad_up, grad_dn, lap_up, lap_dn


# ---------------------------------------------------------------------------
# J2 streaming state (PR3).
#
# When a single electron k of spin σ moves, only pair contributions involving
# k change. The state caches per-electron grad/lap and the previous (r_up,
# r_dn) so the advance can compute the per-pair delta for that electron.
# Cost: O(N_e) per advance, vs O(N_e²) fresh.
# ---------------------------------------------------------------------------


@struct.dataclass
class Jastrow_two_body_streaming_state:
    """Cached J2 grad/lap and electron coordinates consistent with the state."""

    r_up_carts: jax.Array  # (N_up, 3) — config used for the cached J2 quantities
    r_dn_carts: jax.Array  # (N_dn, 3)
    grad_J2_up: jax.Array  # (N_up, 3)
    grad_J2_dn: jax.Array  # (N_dn, 3)
    lap_J2_up: jax.Array  # (N_up,)
    lap_J2_dn: jax.Array  # (N_dn,)


def _j2_pair_terms(j2b_type: str, a: jax.Array, eps: jax.Array, diff: jax.Array):
    """Single-pair grad / lap contributions for the two-body Jastrow.

    Mirrors the closures inside ``compute_grads_and_laplacian_Jastrow_two_body``
    so init and advance share the exact arithmetic. ``j2b_type`` is JIT-static
    (Jastrow_two_body_data marks it ``pytree_node=False``).

    Callers may construct ``diff`` in caller-supplied precision (e.g. fp64
    walker coords for ``r - r_new``); cast at the arithmetic use site here so
    pair-term outputs always live in this function's own zone
    (``jastrow_grad_lap``) regardless of input dtype (Principle 3b — and
    required for fori_loop carry-shape stability under mixed precision,
    where state.r_up_carts is stored in fp64 but state.grad_J2_up lives in
    the grad/lap zone). The cast target is fetched via
    ``get_dtype_jnp("jastrow_grad_lap")`` rather than reading ``a.dtype``,
    so the function declares its own zone explicitly instead of inheriting
    it from a caller-supplied argument.
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    d = diff.astype(dtype_jnp)
    r = jnp.sqrt(jnp.sum(d * d, axis=-1))
    r = jnp.maximum(r, eps)
    if j2b_type == "pade":
        denom = 1.0 + a * r
        f_prime = 0.5 / (denom * denom)
        grad_coeff = f_prime / r
        lap = -a / (denom * denom * denom) + (2.0 * f_prime) / r
    elif j2b_type == "exp":
        exp_term = jnp.exp(-a * r)
        f_prime = 0.5 * exp_term
        grad_coeff = f_prime / r
        lap = -(a / 2.0) * exp_term + (2.0 * f_prime) / r
    else:
        raise ValueError(f"Unknown jastrow_2b_type: {j2b_type}")
    return grad_coeff[..., None] * d, lap


@jit
def _init_grads_laplacian_Jastrow_two_body_streaming_state(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> Jastrow_two_body_streaming_state:
    """Build a J2 state at ``(r_up, r_dn)`` via the existing fresh kernel.

    Stores ``r_up_carts`` / ``r_dn_carts`` in caller-supplied precision
    (Principle 3a — no rebind). Under mixed precision the carry-shape
    must match what ``advance`` writes back via
    ``state.r_*.at[moved_index].set(r_up_carts_new[moved_index])``;
    ``r_up_carts_new`` arrives in fp64 (walker state), so the cached
    coords must be fp64 too. Diffs are downcast to the jastrow_grad_lap
    zone at the arithmetic use site inside ``_j2_pair_terms``
    (Principle 3b).
    """
    g_up, g_dn, l_up, l_dn = compute_grads_and_laplacian_Jastrow_two_body(jastrow_two_body_data, r_up_carts, r_dn_carts)
    return Jastrow_two_body_streaming_state(
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        grad_J2_up=g_up,
        grad_J2_dn=g_dn,
        lap_J2_up=l_up,
        lap_J2_dn=l_dn,
    )


@jit
def _advance_grads_laplacian_Jastrow_two_body_streaming_state(
    jastrow_two_body_data: Jastrow_two_body_data,
    state: Jastrow_two_body_streaming_state,
    moved_spin_is_up: jax.Array,
    moved_index: jax.Array,
    r_up_carts_new: jax.Array,
    r_dn_carts_new: jax.Array,
) -> Jastrow_two_body_streaming_state:
    """Advance J2 state after a single-electron move.

    Computes only the pair-contribution deltas that involve the moved
    electron: ``O(N_e)`` operations per call.
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    j2b_type = jastrow_two_body_data.jastrow_2b_type
    a = jnp.asarray(jastrow_two_body_data.jastrow_2b_param, dtype=dtype_jnp)
    eps = jnp.asarray(EPS_safe_distance, dtype=dtype_jnp)

    num_up = state.r_up_carts.shape[0]
    num_dn = state.r_dn_carts.shape[0]

    def _branch_up(_):
        r_old = state.r_up_carts[moved_index]
        r_new = r_up_carts_new[moved_index]

        # --- Same-spin (up-up) pairs (k, i) for i ≠ k --------------------
        # Old & new diffs both place 0 at i=k (state.r_up_carts[k]=r_old vs r_old,
        # r_up_carts_new[k]=r_new vs r_new), so masking is implicit at i=k for new
        # but not for old. Mask out the i=k row explicitly to avoid contaminating k
        # via the eps-clamped self-pair value.
        diff_old_uu = r_old - state.r_up_carts  # (N_up, 3); k-th = 0
        diff_new_uu = r_new - r_up_carts_new  # (N_up, 3); k-th = 0
        grad_old_uu, lap_old_uu = _j2_pair_terms(j2b_type, a, eps, diff_old_uu)
        grad_new_uu, lap_new_uu = _j2_pair_terms(j2b_type, a, eps, diff_new_uu)
        delta_grad_uu = grad_new_uu - grad_old_uu  # (N_up, 3)
        delta_lap_uu = lap_new_uu - lap_old_uu  # (N_up,)
        mask_uu = (jnp.arange(num_up) != moved_index).astype(dtype_jnp)
        delta_grad_uu = delta_grad_uu * mask_uu[:, None]
        delta_lap_uu = delta_lap_uu * mask_uu

        # i ≠ k: grad_up[i] -= delta_grad_uu[i], lap_up[i] += delta_lap_uu[i]
        new_grad_up = state.grad_J2_up - delta_grad_uu
        new_lap_up = state.lap_J2_up + delta_lap_uu
        # k:   grad_up[k] += sum delta_grad_uu, lap_up[k] += sum delta_lap_uu
        new_grad_up = new_grad_up.at[moved_index].add(jnp.sum(delta_grad_uu, axis=0))
        new_lap_up = new_lap_up.at[moved_index].add(jnp.sum(delta_lap_uu, axis=0))

        # --- Cross-spin (up-dn) pairs (k, j) for all j -------------------
        diff_old_ud = r_old[None, :] - state.r_dn_carts  # (N_dn, 3)
        diff_new_ud = r_new[None, :] - state.r_dn_carts  # (N_dn, 3) (r_dn unchanged)
        grad_old_ud, lap_old_ud = _j2_pair_terms(j2b_type, a, eps, diff_old_ud)
        grad_new_ud, lap_new_ud = _j2_pair_terms(j2b_type, a, eps, diff_new_ud)
        delta_grad_ud = grad_new_ud - grad_old_ud
        delta_lap_ud = lap_new_ud - lap_old_ud

        new_grad_up = new_grad_up.at[moved_index].add(jnp.sum(delta_grad_ud, axis=0))
        new_lap_up = new_lap_up.at[moved_index].add(jnp.sum(delta_lap_ud, axis=0))
        new_grad_dn = state.grad_J2_dn - delta_grad_ud
        new_lap_dn = state.lap_J2_dn + delta_lap_ud

        new_r_up = state.r_up_carts.at[moved_index].set(r_new)
        return state.replace(
            r_up_carts=new_r_up,
            grad_J2_up=new_grad_up,
            grad_J2_dn=new_grad_dn,
            lap_J2_up=new_lap_up,
            lap_J2_dn=new_lap_dn,
        )

    def _branch_dn(_):
        r_old = state.r_dn_carts[moved_index]
        r_new = r_dn_carts_new[moved_index]

        # --- Same-spin (dn-dn) -------------------------------------------
        diff_old_dd = r_old - state.r_dn_carts
        diff_new_dd = r_new - r_dn_carts_new
        grad_old_dd, lap_old_dd = _j2_pair_terms(j2b_type, a, eps, diff_old_dd)
        grad_new_dd, lap_new_dd = _j2_pair_terms(j2b_type, a, eps, diff_new_dd)
        delta_grad_dd = grad_new_dd - grad_old_dd
        delta_lap_dd = lap_new_dd - lap_old_dd
        mask_dd = (jnp.arange(num_dn) != moved_index).astype(dtype_jnp)
        delta_grad_dd = delta_grad_dd * mask_dd[:, None]
        delta_lap_dd = delta_lap_dd * mask_dd

        new_grad_dn = state.grad_J2_dn - delta_grad_dd
        new_lap_dn = state.lap_J2_dn + delta_lap_dd
        new_grad_dn = new_grad_dn.at[moved_index].add(jnp.sum(delta_grad_dd, axis=0))
        new_lap_dn = new_lap_dn.at[moved_index].add(jnp.sum(delta_lap_dd, axis=0))

        # --- Cross-spin (up-dn): grad_up[i] receives +grad_pair(r_up[i] - r_dn[k])
        # so for dn-k moving, the deltas flip signs vs the up branch:
        #   diff = r_up[i] - r_dn_*  →  diff_new for r_dn[k]=r_new is r_up[i] - r_new
        diff_old_du = state.r_up_carts - r_old[None, :]  # (N_up, 3)
        diff_new_du = state.r_up_carts - r_new[None, :]  # (N_up, 3)
        grad_old_du, lap_old_du = _j2_pair_terms(j2b_type, a, eps, diff_old_du)
        grad_new_du, lap_new_du = _j2_pair_terms(j2b_type, a, eps, diff_new_du)
        delta_grad_du = grad_new_du - grad_old_du
        delta_lap_du = lap_new_du - lap_old_du

        # grad_up[i] += delta_grad_du[i]  (sign +)
        new_grad_up = state.grad_J2_up + delta_grad_du
        new_lap_up = state.lap_J2_up + delta_lap_du
        # grad_dn[k] -= sum_i delta_grad_du[i]  (sign − accumulated at k)
        new_grad_dn = new_grad_dn.at[moved_index].add(-jnp.sum(delta_grad_du, axis=0))
        new_lap_dn = new_lap_dn.at[moved_index].add(jnp.sum(delta_lap_du, axis=0))

        new_r_dn = state.r_dn_carts.at[moved_index].set(r_new)
        return state.replace(
            r_dn_carts=new_r_dn,
            grad_J2_up=new_grad_up,
            grad_J2_dn=new_grad_dn,
            lap_J2_up=new_lap_up,
            lap_J2_dn=new_lap_dn,
        )

    if num_up == 0:
        return _branch_dn(None)
    if num_dn == 0:
        return _branch_up(None)
    return jax.lax.cond(moved_spin_is_up, _branch_up, _branch_dn, operand=None)


def _compute_grads_and_laplacian_Jastrow_two_body_debug(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: np.ndarray,
    r_dn_carts: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """See _api method."""
    dtype_np = get_dtype_np("jastrow_grad_lap")
    diff_h = 1.0e-5

    # grad up
    grad_x_up = []
    grad_y_up = []
    grad_z_up = []
    for r_i, _ in enumerate(r_up_carts):
        diff_p_x_r_up_carts = r_up_carts.copy()
        diff_p_y_r_up_carts = r_up_carts.copy()
        diff_p_z_r_up_carts = r_up_carts.copy()
        diff_p_x_r_up_carts[r_i][0] += diff_h
        diff_p_y_r_up_carts[r_i][1] += diff_h
        diff_p_z_r_up_carts[r_i][2] += diff_h

        J2_p_x_up = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_x_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_p_y_up = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_y_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_p_z_up = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_z_r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        diff_m_x_r_up_carts = r_up_carts.copy()
        diff_m_y_r_up_carts = r_up_carts.copy()
        diff_m_z_r_up_carts = r_up_carts.copy()
        diff_m_x_r_up_carts[r_i][0] -= diff_h
        diff_m_y_r_up_carts[r_i][1] -= diff_h
        diff_m_z_r_up_carts[r_i][2] -= diff_h

        J2_m_x_up = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_x_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_m_y_up = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_y_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_m_z_up = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_z_r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        grad_x_up.append((J2_p_x_up - J2_m_x_up) / (2.0 * diff_h))
        grad_y_up.append((J2_p_y_up - J2_m_y_up) / (2.0 * diff_h))
        grad_z_up.append((J2_p_z_up - J2_m_z_up) / (2.0 * diff_h))

    # grad dn
    grad_x_dn = []
    grad_y_dn = []
    grad_z_dn = []
    for r_i, _ in enumerate(r_dn_carts):
        diff_p_x_r_dn_carts = r_dn_carts.copy()
        diff_p_y_r_dn_carts = r_dn_carts.copy()
        diff_p_z_r_dn_carts = r_dn_carts.copy()
        diff_p_x_r_dn_carts[r_i][0] += diff_h
        diff_p_y_r_dn_carts[r_i][1] += diff_h
        diff_p_z_r_dn_carts[r_i][2] += diff_h

        J2_p_x_dn = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_x_r_dn_carts,
        )
        J2_p_y_dn = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_y_r_dn_carts,
        )
        J2_p_z_dn = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_z_r_dn_carts,
        )

        diff_m_x_r_dn_carts = r_dn_carts.copy()
        diff_m_y_r_dn_carts = r_dn_carts.copy()
        diff_m_z_r_dn_carts = r_dn_carts.copy()
        diff_m_x_r_dn_carts[r_i][0] -= diff_h
        diff_m_y_r_dn_carts[r_i][1] -= diff_h
        diff_m_z_r_dn_carts[r_i][2] -= diff_h

        J2_m_x_dn = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_x_r_dn_carts,
        )
        J2_m_y_dn = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_y_r_dn_carts,
        )
        J2_m_z_dn = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_z_r_dn_carts,
        )

        grad_x_dn.append((J2_p_x_dn - J2_m_x_dn) / (2.0 * diff_h))
        grad_y_dn.append((J2_p_y_dn - J2_m_y_dn) / (2.0 * diff_h))
        grad_z_dn.append((J2_p_z_dn - J2_m_z_dn) / (2.0 * diff_h))

    grad_J2_up = np.array([grad_x_up, grad_y_up, grad_z_up], dtype=dtype_np).T
    grad_J2_dn = np.array([grad_x_dn, grad_y_dn, grad_z_dn], dtype=dtype_np).T

    # laplacian
    diff_h2 = 1.0e-3  # for laplacian

    J2_ref = compute_Jastrow_two_body(
        jastrow_two_body_data=jastrow_two_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    lap_J2_up = np.zeros(len(r_up_carts), dtype=dtype_np)

    # laplacians up
    for r_i, _ in enumerate(r_up_carts):
        diff_p_x_r_up2_carts = r_up_carts.copy()
        diff_p_y_r_up2_carts = r_up_carts.copy()
        diff_p_z_r_up2_carts = r_up_carts.copy()
        diff_p_x_r_up2_carts[r_i][0] += diff_h2
        diff_p_y_r_up2_carts[r_i][1] += diff_h2
        diff_p_z_r_up2_carts[r_i][2] += diff_h2

        J2_p_x_up2 = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_x_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_p_y_up2 = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_y_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )

        J2_p_z_up2 = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_z_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )

        diff_m_x_r_up2_carts = r_up_carts.copy()
        diff_m_y_r_up2_carts = r_up_carts.copy()
        diff_m_z_r_up2_carts = r_up_carts.copy()
        diff_m_x_r_up2_carts[r_i][0] -= diff_h2
        diff_m_y_r_up2_carts[r_i][1] -= diff_h2
        diff_m_z_r_up2_carts[r_i][2] -= diff_h2

        J2_m_x_up2 = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_x_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_m_y_up2 = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_y_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_m_z_up2 = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_z_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )

        gradgrad_x_up = (J2_p_x_up2 + J2_m_x_up2 - 2 * J2_ref) / (diff_h2**2)
        gradgrad_y_up = (J2_p_y_up2 + J2_m_y_up2 - 2 * J2_ref) / (diff_h2**2)
        gradgrad_z_up = (J2_p_z_up2 + J2_m_z_up2 - 2 * J2_ref) / (diff_h2**2)

        lap_J2_up[r_i] = gradgrad_x_up + gradgrad_y_up + gradgrad_z_up

    lap_J2_dn = np.zeros(len(r_dn_carts), dtype=dtype_np)

    # laplacians dn
    for r_i, _ in enumerate(r_dn_carts):
        diff_p_x_r_dn2_carts = r_dn_carts.copy()
        diff_p_y_r_dn2_carts = r_dn_carts.copy()
        diff_p_z_r_dn2_carts = r_dn_carts.copy()
        diff_p_x_r_dn2_carts[r_i][0] += diff_h2
        diff_p_y_r_dn2_carts[r_i][1] += diff_h2
        diff_p_z_r_dn2_carts[r_i][2] += diff_h2

        J2_p_x_dn2 = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_x_r_dn2_carts,
        )
        J2_p_y_dn2 = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_y_r_dn2_carts,
        )

        J2_p_z_dn2 = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_z_r_dn2_carts,
        )

        diff_m_x_r_dn2_carts = r_dn_carts.copy()
        diff_m_y_r_dn2_carts = r_dn_carts.copy()
        diff_m_z_r_dn2_carts = r_dn_carts.copy()
        diff_m_x_r_dn2_carts[r_i][0] -= diff_h2
        diff_m_y_r_dn2_carts[r_i][1] -= diff_h2
        diff_m_z_r_dn2_carts[r_i][2] -= diff_h2

        J2_m_x_dn2 = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_x_r_dn2_carts,
        )
        J2_m_y_dn2 = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_y_r_dn2_carts,
        )
        J2_m_z_dn2 = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_z_r_dn2_carts,
        )

        gradgrad_x_dn = (J2_p_x_dn2 + J2_m_x_dn2 - 2 * J2_ref) / (diff_h2**2)
        gradgrad_y_dn = (J2_p_y_dn2 + J2_m_y_dn2 - 2 * J2_ref) / (diff_h2**2)
        gradgrad_z_dn = (J2_p_z_dn2 + J2_m_z_dn2 - 2 * J2_ref) / (diff_h2**2)

        lap_J2_dn[r_i] = gradgrad_x_dn + gradgrad_y_dn + gradgrad_z_dn

    return grad_J2_up, grad_J2_dn, lap_J2_up, lap_J2_dn


@jit
def _compute_grads_and_laplacian_Jastrow_three_body_auto(
    jastrow_three_body_data: Jastrow_three_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Function for computing grads and laplacians with a given Jastrow_three_body_data.

    The method is for computing the gradients and the sum of laplacians of J3 at (r_up_carts, r_dn_carts)
    with a given Jastrow_three_body_data.

    Args:
        jastrow_three_body_data (Jastrow_three_body_data): an instance of Jastrow_three_body_data class
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        the gradients(x,y,z) of J(threebody) and the sum of laplacians of J(threebody) at (r_up_carts, r_dn_carts).
    """
    # Forward r_up/dn_carts as-is (Principle 3a — no parameter rebind). Cast to
    # the jastrow_grad_lap zone at the use site (Principle 3b) before passing as
    # the differentiation operand to grad/hessian.
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")

    # compute grad
    grad_J3_up = grad(compute_Jastrow_three_body, argnums=1)(
        jastrow_three_body_data, r_up_carts.astype(dtype_jnp), r_dn_carts.astype(dtype_jnp)
    )

    grad_J3_dn = grad(compute_Jastrow_three_body, argnums=2)(
        jastrow_three_body_data, r_up_carts.astype(dtype_jnp), r_dn_carts.astype(dtype_jnp)
    )

    # compute laplacians
    hessian_J3_up = hessian(compute_Jastrow_three_body, argnums=1)(
        jastrow_three_body_data, r_up_carts.astype(dtype_jnp), r_dn_carts.astype(dtype_jnp)
    )
    laplacian_J3_up = jnp.einsum("ijij->i", hessian_J3_up)

    hessian_J3_dn = hessian(compute_Jastrow_three_body, argnums=2)(
        jastrow_three_body_data, r_up_carts.astype(dtype_jnp), r_dn_carts.astype(dtype_jnp)
    )
    laplacian_J3_dn = jnp.einsum("ijij->i", hessian_J3_dn)

    return grad_J3_up, grad_J3_dn, laplacian_J3_up, laplacian_J3_dn


@jit
def compute_grads_and_laplacian_Jastrow_three_body(
    jastrow_three_body_data: Jastrow_three_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Analytic gradients and Laplacians for the three-body Jastrow.

    The functional form is unchanged; this routine leverages analytic AO/MO
    gradients and Laplacians. Per-electron derivatives are returned (not
    summed), matching kinetic-energy estimator expectations.

    Args:
        jastrow_three_body_data: Three-body Jastrow parameters and orbitals.
        r_up_carts: Spin-up electron coordinates with shape ``(N_up, 3)``.
        r_dn_carts: Spin-down electron coordinates with shape ``(N_dn, 3)``.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            Gradients for up/down electrons with shapes ``(N_up, 3)`` and ``(N_dn, 3)``,
            Laplacians for up/down electrons with shapes ``(N_up,)`` and ``(N_dn,)``.
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    orb_data = jastrow_three_body_data.orb_data

    if isinstance(orb_data, MOs_data):
        compute_orb_vgl = compute_MOs_value_grad_lap
    elif isinstance(orb_data, (AOs_sphe_data, AOs_cart_data)):
        compute_orb_vgl = compute_AOs_value_grad_lap
    else:
        raise NotImplementedError

    # r_*_carts forwarded unchanged to ``compute_orb_vgl``; do not pre-cast
    # (the AO/MO kernels reconstruct ``r - R`` in float64 internally). Single
    # fused dispatch shares the heavy block (exp / poly / S_l_m) across
    # val/grad/lap.
    aos_up, grad_up_x, grad_up_y, grad_up_z, lap_up = compute_orb_vgl(orb_data, r_up_carts)
    aos_dn, grad_dn_x, grad_dn_y, grad_dn_z, lap_dn = compute_orb_vgl(orb_data, r_dn_carts)
    aos_up = jnp.asarray(aos_up, dtype=dtype_jnp)  # (n_orb, n_up)
    aos_dn = jnp.asarray(aos_dn, dtype=dtype_jnp)  # (n_orb, n_dn)

    grad_up = jnp.stack([grad_up_x, grad_up_y, grad_up_z], axis=-1)  # (n_orb, n_up, 3)
    grad_dn = jnp.stack([grad_dn_x, grad_dn_y, grad_dn_z], axis=-1)  # (n_orb, n_dn, 3)

    lap_up = jnp.asarray(lap_up, dtype=dtype_jnp)  # (n_orb, n_up)
    lap_dn = jnp.asarray(lap_dn, dtype=dtype_jnp)  # (n_orb, n_dn)

    j1_vec = jastrow_three_body_data._j_matrix_jnp[:, -1].astype(dtype_jnp)  # (n_orb,)
    j3_mat = jastrow_three_body_data._j_matrix_jnp[:, :-1].astype(dtype_jnp)  # (n_orb, n_orb)

    num_up = aos_up.shape[1]
    num_dn = aos_dn.shape[1]

    # Precompute pair-accumulation masks
    upper_up = jnp.triu(jnp.ones((num_up, num_up), dtype=dtype_jnp), k=1)
    lower_up = jnp.tril(jnp.ones((num_up, num_up), dtype=dtype_jnp), k=-1)
    upper_dn = jnp.triu(jnp.ones((num_dn, num_dn), dtype=dtype_jnp), k=1)
    lower_dn = jnp.tril(jnp.ones((num_dn, num_dn), dtype=dtype_jnp), k=-1)

    # dJ/dA for each electron (orbital-space coefficients)
    g_up = (
        j1_vec[:, None]
        + jnp.dot(j3_mat, aos_up) @ lower_up
        + jnp.dot(j3_mat.T, aos_up) @ upper_up
        + jnp.dot(j3_mat, aos_dn) @ jnp.ones((num_dn, 1), dtype=dtype_jnp)
    )  # (n_orb, n_up)

    g_dn = (
        j1_vec[:, None]
        + jnp.dot(j3_mat, aos_dn) @ lower_dn
        + jnp.dot(j3_mat.T, aos_dn) @ upper_dn
        + jnp.dot(j3_mat.T, aos_up) @ jnp.ones((num_up, 1), dtype=dtype_jnp)
    )  # (n_orb, n_dn)

    grad_J3_up = jnp.einsum("on,onj->nj", g_up, grad_up)
    grad_J3_dn = jnp.einsum("on,onj->nj", g_dn, grad_dn)

    lap_up_contrib = jnp.einsum("on,on->n", g_up, lap_up)
    lap_dn_contrib = jnp.einsum("on,on->n", g_dn, lap_dn)

    return grad_J3_up, grad_J3_dn, lap_up_contrib, lap_dn_contrib


# ---------------------------------------------------------------------------
# J3 streaming state (single-electron rank-1 advance for projection loops)
# ---------------------------------------------------------------------------
#
# The functions below maintain a per-walker auxiliary state that lets us
# advance the per-electron J3 gradients/Laplacians by O(n_ao^2 + n_ao*N_e)
# per single-electron move, instead of recomputing them from scratch at
# O(n_ao^2 * N_e + n_ao * N_e^2). Used by the GFMC projection inner loop
# (jqmc_gfmc.py:_body_fun_n_streaming).
#
# Design references: lrdmc_refactoring.md sections 1-1, 1-2, 1-4.
#
# Lifetime: the state is freshly initialized at each branching boundary
# (when _projection_n is re-entered) and advanced for at most
# `num_mcmc_per_measurement` steps inside the fori_loop, mirroring the
# Sherman-Morrison `A_old_inv`. No persistence across branchings.


@struct.dataclass
class Jastrow_three_body_streaming_state:
    """Auxiliary state for streaming J3 grad/Laplacian updates.

    All fields are evaluated at the current ``(r_up_carts, r_dn_carts)``.
    Advancing the state via :func:`_advance_grads_laplacian_Jastrow_three_body_streaming_state`
    after a single-electron move keeps every field consistent with the new
    configuration, with cost ``O(n_ao^2 + n_ao * N_e)`` per step.

    Fields (shapes use ``n_orb`` for the orbital dimension; for MO-based
    three-body the same ``n_orb`` is used, since orbitals are evaluated by
    ``compute_MOs`` to dimension ``orb_data._num_orb``):

    - ``aos_up`` / ``aos_dn``: ``(n_orb, N_up)`` / ``(n_orb, N_dn)`` orbital values.
    - ``grad_aos_up`` / ``grad_aos_dn``: ``(n_orb, N_up, 3)`` / ``(n_orb, N_dn, 3)``.
    - ``lap_aos_up`` / ``lap_aos_dn``: ``(n_orb, N_up)`` / ``(n_orb, N_dn)``.
    - ``j3_mat_aos_up`` / ``j3_mat_aos_dn``: ``j3_mat @ aos_*`` (shapes match aos_*).
    - ``j3_mat_T_aos_up`` / ``j3_mat_T_aos_dn``: ``j3_mat.T @ aos_*``.
    - ``g_up`` / ``g_dn``: ``(n_orb, N_up)`` / ``(n_orb, N_dn)`` ``dJ/dA`` per electron.
    - ``grad_J3_up`` / ``grad_J3_dn``: ``(N_up, 3)`` / ``(N_dn, 3)`` per-electron grad.
    - ``lap_J3_up`` / ``lap_J3_dn``: ``(N_up,)`` / ``(N_dn,)`` per-electron lap.
    """

    aos_up: jax.Array = struct.field(pytree_node=True)
    aos_dn: jax.Array = struct.field(pytree_node=True)
    grad_aos_up: jax.Array = struct.field(pytree_node=True)
    grad_aos_dn: jax.Array = struct.field(pytree_node=True)
    lap_aos_up: jax.Array = struct.field(pytree_node=True)
    lap_aos_dn: jax.Array = struct.field(pytree_node=True)
    j3_mat_aos_up: jax.Array = struct.field(pytree_node=True)
    j3_mat_aos_dn: jax.Array = struct.field(pytree_node=True)
    j3_mat_T_aos_up: jax.Array = struct.field(pytree_node=True)
    j3_mat_T_aos_dn: jax.Array = struct.field(pytree_node=True)
    g_up: jax.Array = struct.field(pytree_node=True)
    g_dn: jax.Array = struct.field(pytree_node=True)
    grad_J3_up: jax.Array = struct.field(pytree_node=True)
    grad_J3_dn: jax.Array = struct.field(pytree_node=True)
    lap_J3_up: jax.Array = struct.field(pytree_node=True)
    lap_J3_dn: jax.Array = struct.field(pytree_node=True)


def _three_body_orb_apis(jastrow_three_body_data: Jastrow_three_body_data):
    """Pick the correct orbital evaluation backends (AO or MO).

    Returned as a Python tuple so callers can dispatch statically (the
    orb_data type is JIT-static via the @struct.dataclass). Includes the
    fused ``(val, gx, gy, gz, lap)`` dispatcher used by the streaming advance
    hot path so the heavy block (exp / poly / S_l_m) is shared across
    val/grad/lap.
    """
    orb_data = jastrow_three_body_data.orb_data
    if isinstance(orb_data, MOs_data):
        return (
            compute_MOs,
            compute_MOs_grad,
            compute_MOs_laplacian,
            compute_MOs_value_grad_lap,
        )
    if isinstance(orb_data, (AOs_sphe_data, AOs_cart_data)):
        return (
            compute_AOs,
            compute_AOs_grad,
            compute_AOs_laplacian,
            compute_AOs_value_grad_lap,
        )
    raise NotImplementedError(f"Unsupported orb_data type: {type(orb_data)}")


@jit
def _init_grads_laplacian_Jastrow_three_body_streaming_state(
    jastrow_three_body_data: Jastrow_three_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> Jastrow_three_body_streaming_state:
    """Initialize the J3 streaming state from a configuration ``(r_up, r_dn)``.

    This is a one-shot evaluation equivalent in cost to
    :func:`compute_grads_and_laplacian_Jastrow_three_body`; it additionally
    materializes the auxiliary tables required by the rank-1 advance.
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    orb_data = jastrow_three_body_data.orb_data
    compute_orb, compute_orb_grad, compute_orb_lapl, compute_orb_vgl = _three_body_orb_apis(jastrow_three_body_data)

    # AO/MO tables (forward r_*_carts unchanged so the underlying kernels can
    # reconstruct r-R in float64 — Principle 3b). Single fused dispatch shares
    # the heavy block (exp / poly / S_l_m) across val/grad/lap.
    aos_up, grad_up_x, grad_up_y, grad_up_z, lap_aos_up = compute_orb_vgl(orb_data, r_up_carts)
    aos_dn, grad_dn_x, grad_dn_y, grad_dn_z, lap_aos_dn = compute_orb_vgl(orb_data, r_dn_carts)
    aos_up = jnp.asarray(aos_up, dtype=dtype_jnp)
    aos_dn = jnp.asarray(aos_dn, dtype=dtype_jnp)
    grad_aos_up = jnp.asarray(jnp.stack([grad_up_x, grad_up_y, grad_up_z], axis=-1), dtype=dtype_jnp)
    grad_aos_dn = jnp.asarray(jnp.stack([grad_dn_x, grad_dn_y, grad_dn_z], axis=-1), dtype=dtype_jnp)
    lap_aos_up = jnp.asarray(lap_aos_up, dtype=dtype_jnp)
    lap_aos_dn = jnp.asarray(lap_aos_dn, dtype=dtype_jnp)

    j_matrix = jastrow_three_body_data._j_matrix_jnp.astype(dtype_jnp)
    j1_vec = j_matrix[:, -1]
    j3_mat = j_matrix[:, :-1]

    num_up = aos_up.shape[1]
    num_dn = aos_dn.shape[1]

    j3_mat_aos_up = j3_mat @ aos_up
    j3_mat_T_aos_up = j3_mat.T @ aos_up
    j3_mat_aos_dn = j3_mat @ aos_dn
    j3_mat_T_aos_dn = j3_mat.T @ aos_dn

    upper_up = jnp.triu(jnp.ones((num_up, num_up), dtype=dtype_jnp), k=1)
    lower_up = jnp.tril(jnp.ones((num_up, num_up), dtype=dtype_jnp), k=-1)
    upper_dn = jnp.triu(jnp.ones((num_dn, num_dn), dtype=dtype_jnp), k=1)
    lower_dn = jnp.tril(jnp.ones((num_dn, num_dn), dtype=dtype_jnp), k=-1)

    g_up = (
        j1_vec[:, None]
        + j3_mat_aos_up @ lower_up
        + j3_mat_T_aos_up @ upper_up
        + j3_mat_aos_dn @ jnp.ones((num_dn, 1), dtype=dtype_jnp)
    )
    g_dn = (
        j1_vec[:, None]
        + j3_mat_aos_dn @ lower_dn
        + j3_mat_T_aos_dn @ upper_dn
        + j3_mat_T_aos_up @ jnp.ones((num_up, 1), dtype=dtype_jnp)
    )

    grad_J3_up = jnp.einsum("on,onj->nj", g_up, grad_aos_up)
    grad_J3_dn = jnp.einsum("on,onj->nj", g_dn, grad_aos_dn)
    lap_J3_up = jnp.einsum("on,on->n", g_up, lap_aos_up)
    lap_J3_dn = jnp.einsum("on,on->n", g_dn, lap_aos_dn)

    return Jastrow_three_body_streaming_state(
        aos_up=aos_up,
        aos_dn=aos_dn,
        grad_aos_up=grad_aos_up,
        grad_aos_dn=grad_aos_dn,
        lap_aos_up=lap_aos_up,
        lap_aos_dn=lap_aos_dn,
        j3_mat_aos_up=j3_mat_aos_up,
        j3_mat_aos_dn=j3_mat_aos_dn,
        j3_mat_T_aos_up=j3_mat_T_aos_up,
        j3_mat_T_aos_dn=j3_mat_T_aos_dn,
        g_up=g_up,
        g_dn=g_dn,
        grad_J3_up=grad_J3_up,
        grad_J3_dn=grad_J3_dn,
        lap_J3_up=lap_J3_up,
        lap_J3_dn=lap_J3_dn,
    )


@jit
def _advance_grads_laplacian_Jastrow_three_body_streaming_state(
    jastrow_three_body_data: Jastrow_three_body_data,
    state: Jastrow_three_body_streaming_state,
    moved_spin_is_up: jax.Array,
    moved_index: jax.Array,
    r_up_carts_new: jax.Array,
    r_dn_carts_new: jax.Array,
) -> Jastrow_three_body_streaming_state:
    """Advance the J3 streaming state after a single-electron move.

    The new ``(r_up_carts_new, r_dn_carts_new)`` differ from the configuration
    represented by ``state`` in *exactly one* electron position, identified by
    ``(moved_spin_is_up, moved_index)``. If neither spin actually moved (e.g. a
    no-op step), the state should still be passed through unchanged by the
    caller — this routine assumes a real one-electron displacement.

    Cost: ``O(n_ao^2 + n_ao * N_e)`` per call, dominated by two ``n_ao``-sized
    matvecs ``j3_mat @ delta_aos`` and one full einsum over ``g``.
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    orb_data = jastrow_three_body_data.orb_data
    compute_orb, compute_orb_grad, compute_orb_lapl, compute_orb_vgl = _three_body_orb_apis(jastrow_three_body_data)

    j_matrix = jastrow_three_body_data._j_matrix_jnp.astype(dtype_jnp)
    j3_mat = j_matrix[:, :-1]

    num_up = state.aos_up.shape[1]
    num_dn = state.aos_dn.shape[1]

    def _branch_up(_):
        # Single-point AO eval at the moved electron's new position.
        # NB: forward r_up_carts_new unchanged (Principle 3b — fp64 r-R
        # reconstruction inside the kernels). Single fused dispatch shares
        # the heavy block (exp / poly / S_l_m) across val/grad/lap.
        r_new = jnp.expand_dims(r_up_carts_new[moved_index], axis=0)  # (1, 3)
        ao_v, gx, gy, gz, ao_lap = compute_orb_vgl(orb_data, r_new)
        aos_new_col = jnp.asarray(ao_v[:, 0], dtype=dtype_jnp)
        grad_aos_new_col = jnp.asarray(jnp.stack([gx[:, 0], gy[:, 0], gz[:, 0]], axis=-1), dtype=dtype_jnp)
        lap_aos_new_col = jnp.asarray(ao_lap[:, 0], dtype=dtype_jnp)

        delta_aos = aos_new_col - state.aos_up[:, moved_index]
        d_J = j3_mat @ delta_aos  # (n_orb,)
        d_JT = j3_mat.T @ delta_aos  # (n_orb,)

        # Update auxiliary matmuls at the moved column.
        new_j3_mat_aos_up = state.j3_mat_aos_up.at[:, moved_index].add(d_J)
        new_j3_mat_T_aos_up = state.j3_mat_T_aos_up.at[:, moved_index].add(d_JT)
        # j3_mat_aos_dn / j3_mat_T_aos_dn unchanged (depend on aos_dn).

        # g_up update:
        #   term A (j3_mat_aos_up @ lower_up): col j gets +d_J for j < k.
        #   term B (j3_mat_T_aos_up @ upper_up): col j gets +d_JT for j > k.
        #   term C (cross-spin via aos_dn): unchanged.
        #   col k itself: unchanged (strict triangulars set k-th col to 0).
        col_idx_up = jnp.arange(num_up)
        mask_lt = (col_idx_up < moved_index).astype(dtype_jnp)
        mask_gt = (col_idx_up > moved_index).astype(dtype_jnp)
        new_g_up = state.g_up + d_J[:, None] * mask_lt[None, :] + d_JT[:, None] * mask_gt[None, :]

        # g_dn update: term C is (j3_mat.T @ aos_up) @ ones_up, so the change
        # is sum_k Δ(j3_mat.T @ aos_up)[:, k] = d_JT (single column changed).
        # Same vector added to every dn column.
        new_g_dn = state.g_dn + d_JT[:, None]

        # Update aos/grad_aos/lap_aos at the moved column.
        new_aos_up = state.aos_up.at[:, moved_index].set(aos_new_col)
        new_grad_aos_up = state.grad_aos_up.at[:, moved_index, :].set(grad_aos_new_col)
        new_lap_aos_up = state.lap_aos_up.at[:, moved_index].set(lap_aos_new_col)

        # Recompute per-electron grad_J3_*, lap_J3_* via einsum on updated
        # tables. Cost: O(n_ao * N_e * 3) — within target asymptotics.
        grad_J3_up = jnp.einsum("on,onj->nj", new_g_up, new_grad_aos_up)
        grad_J3_dn = jnp.einsum("on,onj->nj", new_g_dn, state.grad_aos_dn)
        lap_J3_up = jnp.einsum("on,on->n", new_g_up, new_lap_aos_up)
        lap_J3_dn = jnp.einsum("on,on->n", new_g_dn, state.lap_aos_dn)

        return state.replace(
            aos_up=new_aos_up,
            grad_aos_up=new_grad_aos_up,
            lap_aos_up=new_lap_aos_up,
            j3_mat_aos_up=new_j3_mat_aos_up,
            j3_mat_T_aos_up=new_j3_mat_T_aos_up,
            g_up=new_g_up,
            g_dn=new_g_dn,
            grad_J3_up=grad_J3_up,
            grad_J3_dn=grad_J3_dn,
            lap_J3_up=lap_J3_up,
            lap_J3_dn=lap_J3_dn,
        )

    def _branch_dn(_):
        # Single fused dispatch (see _branch_up).
        r_new = jnp.expand_dims(r_dn_carts_new[moved_index], axis=0)
        ao_v, gx, gy, gz, ao_lap = compute_orb_vgl(orb_data, r_new)
        aos_new_col = jnp.asarray(ao_v[:, 0], dtype=dtype_jnp)
        grad_aos_new_col = jnp.asarray(jnp.stack([gx[:, 0], gy[:, 0], gz[:, 0]], axis=-1), dtype=dtype_jnp)
        lap_aos_new_col = jnp.asarray(ao_lap[:, 0], dtype=dtype_jnp)

        delta_aos = aos_new_col - state.aos_dn[:, moved_index]
        d_J = j3_mat @ delta_aos
        d_JT = j3_mat.T @ delta_aos

        new_j3_mat_aos_dn = state.j3_mat_aos_dn.at[:, moved_index].add(d_J)
        new_j3_mat_T_aos_dn = state.j3_mat_T_aos_dn.at[:, moved_index].add(d_JT)

        col_idx_dn = jnp.arange(num_dn)
        mask_lt = (col_idx_dn < moved_index).astype(dtype_jnp)
        mask_gt = (col_idx_dn > moved_index).astype(dtype_jnp)
        new_g_dn = state.g_dn + d_J[:, None] * mask_lt[None, :] + d_JT[:, None] * mask_gt[None, :]

        # g_up term C is (j3_mat @ aos_dn) @ ones_dn, change = d_J for every up col.
        new_g_up = state.g_up + d_J[:, None]

        new_aos_dn = state.aos_dn.at[:, moved_index].set(aos_new_col)
        new_grad_aos_dn = state.grad_aos_dn.at[:, moved_index, :].set(grad_aos_new_col)
        new_lap_aos_dn = state.lap_aos_dn.at[:, moved_index].set(lap_aos_new_col)

        grad_J3_up = jnp.einsum("on,onj->nj", new_g_up, state.grad_aos_up)
        grad_J3_dn = jnp.einsum("on,onj->nj", new_g_dn, new_grad_aos_dn)
        lap_J3_up = jnp.einsum("on,on->n", new_g_up, state.lap_aos_up)
        lap_J3_dn = jnp.einsum("on,on->n", new_g_dn, new_lap_aos_dn)

        return state.replace(
            aos_dn=new_aos_dn,
            grad_aos_dn=new_grad_aos_dn,
            lap_aos_dn=new_lap_aos_dn,
            j3_mat_aos_dn=new_j3_mat_aos_dn,
            j3_mat_T_aos_dn=new_j3_mat_T_aos_dn,
            g_up=new_g_up,
            g_dn=new_g_dn,
            grad_J3_up=grad_J3_up,
            grad_J3_dn=grad_J3_dn,
            lap_J3_up=lap_J3_up,
            lap_J3_dn=lap_J3_dn,
        )

    # Edge case: zero-electron spin sector — no advance possible, just no-op.
    if num_up == 0:
        return _branch_dn(None)
    if num_dn == 0:
        return _branch_up(None)
    return jax.lax.cond(moved_spin_is_up, _branch_up, _branch_dn, operand=None)


def _compute_grads_and_laplacian_Jastrow_three_body_debug(
    jastrow_three_body_data: Jastrow_three_body_data,
    r_up_carts: np.ndarray,
    r_dn_carts: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """See _api method."""
    dtype_np = get_dtype_np("jastrow_grad_lap")
    diff_h = 1.0e-5

    # grad up
    grad_x_up = []
    grad_y_up = []
    grad_z_up = []
    for r_i, _ in enumerate(r_up_carts):
        diff_p_x_r_up_carts = r_up_carts.copy()
        diff_p_y_r_up_carts = r_up_carts.copy()
        diff_p_z_r_up_carts = r_up_carts.copy()
        diff_p_x_r_up_carts[r_i][0] += diff_h
        diff_p_y_r_up_carts[r_i][1] += diff_h
        diff_p_z_r_up_carts[r_i][2] += diff_h

        J3_p_x_up = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_p_x_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_p_y_up = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_p_y_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_p_z_up = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_p_z_r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        diff_m_x_r_up_carts = r_up_carts.copy()
        diff_m_y_r_up_carts = r_up_carts.copy()
        diff_m_z_r_up_carts = r_up_carts.copy()
        diff_m_x_r_up_carts[r_i][0] -= diff_h
        diff_m_y_r_up_carts[r_i][1] -= diff_h
        diff_m_z_r_up_carts[r_i][2] -= diff_h

        J3_m_x_up = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_m_x_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_m_y_up = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_m_y_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_m_z_up = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_m_z_r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        grad_x_up.append((J3_p_x_up - J3_m_x_up) / (2.0 * diff_h))
        grad_y_up.append((J3_p_y_up - J3_m_y_up) / (2.0 * diff_h))
        grad_z_up.append((J3_p_z_up - J3_m_z_up) / (2.0 * diff_h))

    # grad dn
    grad_x_dn = []
    grad_y_dn = []
    grad_z_dn = []
    for r_i, _ in enumerate(r_dn_carts):
        diff_p_x_r_dn_carts = r_dn_carts.copy()
        diff_p_y_r_dn_carts = r_dn_carts.copy()
        diff_p_z_r_dn_carts = r_dn_carts.copy()
        diff_p_x_r_dn_carts[r_i][0] += diff_h
        diff_p_y_r_dn_carts[r_i][1] += diff_h
        diff_p_z_r_dn_carts[r_i][2] += diff_h

        J3_p_x_dn = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_x_r_dn_carts,
        )
        J3_p_y_dn = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_y_r_dn_carts,
        )
        J3_p_z_dn = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_z_r_dn_carts,
        )

        diff_m_x_r_dn_carts = r_dn_carts.copy()
        diff_m_y_r_dn_carts = r_dn_carts.copy()
        diff_m_z_r_dn_carts = r_dn_carts.copy()
        diff_m_x_r_dn_carts[r_i][0] -= diff_h
        diff_m_y_r_dn_carts[r_i][1] -= diff_h
        diff_m_z_r_dn_carts[r_i][2] -= diff_h

        J3_m_x_dn = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_x_r_dn_carts,
        )
        J3_m_y_dn = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_y_r_dn_carts,
        )
        J3_m_z_dn = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_z_r_dn_carts,
        )

        grad_x_dn.append((J3_p_x_dn - J3_m_x_dn) / (2.0 * diff_h))
        grad_y_dn.append((J3_p_y_dn - J3_m_y_dn) / (2.0 * diff_h))
        grad_z_dn.append((J3_p_z_dn - J3_m_z_dn) / (2.0 * diff_h))

    grad_J3_up = np.array([grad_x_up, grad_y_up, grad_z_up], dtype=dtype_np).T
    grad_J3_dn = np.array([grad_x_dn, grad_y_dn, grad_z_dn], dtype=dtype_np).T

    # laplacian
    diff_h2 = 1.0e-3  # for laplacian

    J3_ref = compute_Jastrow_three_body(
        jastrow_three_body_data=jastrow_three_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    lap_J3_up = np.zeros(len(r_up_carts), dtype=dtype_np)

    # laplacians up
    for r_i, _ in enumerate(r_up_carts):
        diff_p_x_r_up2_carts = r_up_carts.copy()
        diff_p_y_r_up2_carts = r_up_carts.copy()
        diff_p_z_r_up2_carts = r_up_carts.copy()
        diff_p_x_r_up2_carts[r_i][0] += diff_h2
        diff_p_y_r_up2_carts[r_i][1] += diff_h2
        diff_p_z_r_up2_carts[r_i][2] += diff_h2

        J3_p_x_up2 = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_p_x_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_p_y_up2 = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_p_y_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )

        J3_p_z_up2 = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_p_z_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )

        diff_m_x_r_up2_carts = r_up_carts.copy()
        diff_m_y_r_up2_carts = r_up_carts.copy()
        diff_m_z_r_up2_carts = r_up_carts.copy()
        diff_m_x_r_up2_carts[r_i][0] -= diff_h2
        diff_m_y_r_up2_carts[r_i][1] -= diff_h2
        diff_m_z_r_up2_carts[r_i][2] -= diff_h2

        J3_m_x_up2 = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_m_x_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_m_y_up2 = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_m_y_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_m_z_up2 = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_m_z_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )

        gradgrad_x_up = (J3_p_x_up2 + J3_m_x_up2 - 2 * J3_ref) / (diff_h2**2)
        gradgrad_y_up = (J3_p_y_up2 + J3_m_y_up2 - 2 * J3_ref) / (diff_h2**2)
        gradgrad_z_up = (J3_p_z_up2 + J3_m_z_up2 - 2 * J3_ref) / (diff_h2**2)

        lap_J3_up[r_i] = gradgrad_x_up + gradgrad_y_up + gradgrad_z_up

    lap_J3_dn = np.zeros(len(r_dn_carts), dtype=dtype_np)

    # laplacians dn
    for r_i, _ in enumerate(r_dn_carts):
        diff_p_x_r_dn2_carts = r_dn_carts.copy()
        diff_p_y_r_dn2_carts = r_dn_carts.copy()
        diff_p_z_r_dn2_carts = r_dn_carts.copy()
        diff_p_x_r_dn2_carts[r_i][0] += diff_h2
        diff_p_y_r_dn2_carts[r_i][1] += diff_h2
        diff_p_z_r_dn2_carts[r_i][2] += diff_h2

        J3_p_x_dn2 = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_x_r_dn2_carts,
        )
        J3_p_y_dn2 = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_y_r_dn2_carts,
        )

        J3_p_z_dn2 = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_z_r_dn2_carts,
        )

        diff_m_x_r_dn2_carts = r_dn_carts.copy()
        diff_m_y_r_dn2_carts = r_dn_carts.copy()
        diff_m_z_r_dn2_carts = r_dn_carts.copy()
        diff_m_x_r_dn2_carts[r_i][0] -= diff_h2
        diff_m_y_r_dn2_carts[r_i][1] -= diff_h2
        diff_m_z_r_dn2_carts[r_i][2] -= diff_h2

        J3_m_x_dn2 = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_x_r_dn2_carts,
        )
        J3_m_y_dn2 = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_y_r_dn2_carts,
        )
        J3_m_z_dn2 = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_z_r_dn2_carts,
        )

        gradgrad_x_dn = (J3_p_x_dn2 + J3_m_x_dn2 - 2 * J3_ref) / (diff_h2**2)
        gradgrad_y_dn = (J3_p_y_dn2 + J3_m_y_dn2 - 2 * J3_ref) / (diff_h2**2)
        gradgrad_z_dn = (J3_p_z_dn2 + J3_m_z_dn2 - 2 * J3_ref) / (diff_h2**2)

        lap_J3_dn[r_i] = gradgrad_x_dn + gradgrad_y_dn + gradgrad_z_dn

    return grad_J3_up, grad_J3_dn, lap_J3_up, lap_J3_dn


"""
if __name__ == "__main__":
    import pickle
    import time

    log = getLogger("jqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)
"""
