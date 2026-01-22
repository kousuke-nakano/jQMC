"""Jastrow module."""

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

from .atomic_orbital import AOs_cart_data, AOs_sphe_data, compute_AOs, compute_AOs_grad, compute_AOs_laplacian
from .molecular_orbital import MOs_data, compute_MOs, compute_MOs_grad, compute_MOs_laplacian
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

            features = (d**2) * jnp.exp(-d - ((d - mu) ** 2) / (sigma**2 + 1e-12))
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
                return jnp.zeros((radial_features.shape[0], self.hidden_dim))
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
        if A.shape[0] == 0 or B.shape[0] == 0:
            return jnp.zeros((A.shape[0], B.shape[0]))
        diff = A[:, None, :] - B[None, :, :]
        return jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)

    def _nuclear_embeddings(self, Z_n: jnp.ndarray) -> jnp.ndarray:
        """Convert atomic numbers into learned embedding vectors.

        Args:
            Z_n: Integer array of atomic numbers with shape ``(n_nuc,)``.

        Returns:
            jnp.ndarray: ``(n_nuc, hidden_dim)`` embeddings looked up through
            ``species_lookup``. Returns an empty array when no nuclei are present.
        """
        n_nuc = Z_n.shape[0]
        if n_nuc == 0:
            return jnp.zeros((0, self.hidden_dim))

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
        r_up = jnp.asarray(r_up)
        r_dn = jnp.asarray(r_dn)
        R_n = jnp.asarray(R_n)
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
    """One-body Jastrow parameters and structure metadata.

    The one-body term models electron–nucleus correlations using the
    exponential form described in the original docstring. The numerical value
    is returned without the ``exp`` wrapper; callers attach ``exp(J)`` to the
    wavefunction.

    Args:
        jastrow_1b_param (float): Parameter controlling the one-body decay.
        structure_data (Structure_data): Nuclear positions and charges.
        core_electrons (tuple[float]): Removed core electrons per nucleus (for ECPs).
    """

    jastrow_1b_param: float = struct.field(pytree_node=True, default=1.0)  #: One-body Jastrow exponent parameter.
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
        info_lines.append("  1b Jastrow functional form is the exp type.")
        return info_lines

    def _logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self._get_info():
            logger.info(line)

    @classmethod
    def init_jastrow_one_body_data(cls, jastrow_1b_param, structure_data, core_electrons):
        """Initialization."""
        jastrow_one_body_data = cls(
            jastrow_1b_param=jastrow_1b_param, structure_data=structure_data, core_electrons=core_electrons
        )
        return jastrow_one_body_data


@jit
def compute_Jastrow_one_body(
    jastrow_one_body_data: Jastrow_one_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float:
    """Evaluate the one-body Jastrow $J_1$ (without ``exp``) for given coordinates.

    The original exponential form and usage remain unchanged: this routine
    returns the scalar ``J`` value; callers attach ``exp(J)`` to the wavefunction.

    Args:
        jastrow_one_body_data: One-body Jastrow parameters and structure data.
        r_up_carts: Spin-up electron coordinates with shape ``(N_up, 3)``.
        r_dn_carts: Spin-down electron coordinates with shape ``(N_dn, 3)``.

    Returns:
        float: One-body Jastrow value (before exponentiation).
    """
    # Retrieve structure data and convert to JAX arrays
    R_carts = jnp.array(jastrow_one_body_data.structure_data.positions)
    atomic_numbers = jnp.array(jastrow_one_body_data.structure_data.atomic_numbers)
    core_electrons = jnp.array(jastrow_one_body_data.core_electrons)
    effective_charges = atomic_numbers - core_electrons

    def one_body_jastrow_exp(
        param: float,
        coeff: float,
        r_cart: jnpt.ArrayLike,
        R_cart: jnpt.ArrayLike,
    ) -> float:
        """Exponential form of J1."""
        one_body_jastrow = 1.0 / (2.0 * param) * (1.0 - jnp.exp(-param * coeff * jnp.linalg.norm(r_cart - R_cart)))
        return one_body_jastrow

    # Function to compute the contribution from one atom
    def atom_contrib(r_cart, R_cart, Z_eff):
        j1b = jastrow_one_body_data.jastrow_1b_param
        coeff = (2.0 * Z_eff) ** (1.0 / 4.0)
        return -((2.0 * Z_eff) ** (3.0 / 4.0)) * one_body_jastrow_exp(j1b, coeff, r_cart, R_cart)

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
    positions = jastrow_one_body_data.structure_data.positions
    atomic_numbers = jastrow_one_body_data.structure_data.atomic_numbers
    core_electrons = jastrow_one_body_data.core_electrons
    effective_charges = np.array(atomic_numbers) - np.array(core_electrons)

    def one_body_jastrow_exp(
        param: float, coeff: float, r_cart: npt.NDArray[np.float64], R_cart: npt.NDArray[np.float64]
    ) -> float:
        """Exponential form of J1."""
        one_body_jastrow = 1.0 / (2.0 * param) * (1.0 - np.exp(-param * coeff * np.linalg.norm(r_cart - R_cart)))
        return one_body_jastrow

    J1_up = 0.0
    for r_up in r_up_carts:
        for R_cart, Z_eff in zip(positions, effective_charges, strict=True):
            coeff = (2.0 * Z_eff) ** (1.0 / 4.0)
            J1_up += -((2.0 * Z_eff) ** (3.0 / 4.0)) * one_body_jastrow_exp(
                jastrow_one_body_data.jastrow_1b_param, coeff, r_up, R_cart
            )

    J1_dn = 0.0
    for r_up in r_dn_carts:
        for R_cart, Z_eff in zip(positions, effective_charges, strict=True):
            coeff = (2.0 * Z_eff) ** (1.0 / 4.0)
            J1_dn += -((2.0 * Z_eff) ** (3.0 / 4.0)) * one_body_jastrow_exp(
                jastrow_one_body_data.jastrow_1b_param, coeff, r_up, R_cart
            )

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
    diff_h = 1.0e-5
    r_up_carts = np.array(r_up_carts, dtype=float)
    r_dn_carts = np.array(r_dn_carts, dtype=float)

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

    grad_J1_up = np.array([grad_x_up, grad_y_up, grad_z_up]).T
    grad_J1_dn = np.array([grad_x_dn, grad_y_dn, grad_z_dn]).T

    # laplacian
    diff_h2 = 1.0e-3
    J_ref = _compute_Jastrow_one_body_debug(jastrow_one_body_data, r_up_carts, r_dn_carts)

    lap_J1_up = np.zeros(len(r_up_carts), dtype=float)

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

    lap_J1_dn = np.zeros(len(r_dn_carts), dtype=float)

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
    r_up_carts = jnp.array(r_up_carts)
    r_dn_carts = jnp.array(r_dn_carts)

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
    positions = jnp.asarray(jastrow_one_body_data.structure_data.positions)
    atomic_numbers = jnp.asarray(jastrow_one_body_data.structure_data.atomic_numbers)
    core_electrons = jnp.asarray(jastrow_one_body_data.core_electrons)
    z_eff = atomic_numbers - core_electrons

    a = jastrow_one_body_data.jastrow_1b_param
    c = (2.0 * z_eff) ** (1.0 / 4.0)
    A = (2.0 * z_eff) ** (3.0 / 4.0)

    eps = 1.0e-12

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

    grad_up, lap_up = _grad_lap_one_spin(jnp.asarray(r_up_carts))
    grad_dn, lap_dn = _grad_lap_one_spin(jnp.asarray(r_dn_carts))

    return grad_up, grad_dn, lap_up, lap_dn


@struct.dataclass
class Jastrow_two_body_data:
    """Two-body Jastrow parameter container.

    The two-body term uses the Pade functional form described in the existing
    docstrings. Values are returned without exponentiation; callers use
    ``exp(J)`` when constructing the wavefunction.

    Args:
        jastrow_2b_param (float): Parameter for the two-body Jastrow part.
    """

    jastrow_2b_param: float = struct.field(pytree_node=True, default=1.0)  #: Pade ``a`` parameter for J2.

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        if self.jastrow_2b_param < 0.0:
            raise ValueError(f"jastrow_2b_param = {self.jastrow_2b_param} must be non-negative.")

    def _get_info(self) -> list[str]:
        """Return a list of strings representing the logged information."""
        info_lines = []
        info_lines.append("**" + self.__class__.__name__)
        info_lines.append(f"  Jastrow 2b param = {self.jastrow_2b_param}")
        info_lines.append("  2b Jastrow functional form is the pade type.")
        return info_lines

    def _logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self._get_info():
            logger.info(line)

    @classmethod
    def init_jastrow_two_body_data(cls, jastrow_2b_param=1.0):
        """Initialization."""
        jastrow_two_body_data = cls(jastrow_2b_param=jastrow_2b_param)
        return jastrow_two_body_data


@jit
def compute_Jastrow_two_body(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float:
    """Evaluate the two-body Jastrow $J_2$ (Pade form) without exponentiation.

    The functional form and usage remain identical to the original docstring;
    this returns ``J`` and callers attach ``exp(J)`` to the wavefunction.

    Args:
        jastrow_two_body_data: Two-body Jastrow parameter container.
        r_up_carts: Spin-up electron coordinates with shape ``(N_up, 3)``.
        r_dn_carts: Spin-down electron coordinates with shape ``(N_dn, 3)``.

    Returns:
        float: Two-body Jastrow value (before exponentiation).
    """

    def two_body_jastrow_anti_parallel_spins_exp(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> float:
        """Exponential form of J2 for anti-parallel spins."""
        two_body_jastrow = 1.0 / (2.0 * param) * (1.0 - jnp.exp(-param * jnp.linalg.norm(r_cart_i - r_cart_j)))
        return two_body_jastrow

    def two_body_jastrow_parallel_spins_exp(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> float:
        """Exponential form of J2 for parallel spins."""
        two_body_jastrow = 1.0 / (2.0 * param) * (1.0 - jnp.exp(-param * jnp.linalg.norm(r_cart_i - r_cart_j)))
        return two_body_jastrow

    def two_body_jastrow_anti_parallel_spins_pade(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> float:
        """Pade form of J2 for anti-parallel spins."""
        two_body_jastrow = (
            jnp.linalg.norm(r_cart_i - r_cart_j) / 2.0 * (1.0 + param * jnp.linalg.norm(r_cart_i - r_cart_j)) ** (-1.0)
        )
        return two_body_jastrow

    def two_body_jastrow_parallel_spins_pade(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> float:
        """Pade form of J2 for parallel spins."""
        two_body_jastrow = (
            jnp.linalg.norm(r_cart_i - r_cart_j) / 2.0 * (1.0 + param * jnp.linalg.norm(r_cart_i - r_cart_j)) ** (-1.0)
        )
        return two_body_jastrow

    vmap_two_body_jastrow_anti_parallel_spins = vmap(
        vmap(two_body_jastrow_anti_parallel_spins_pade, in_axes=(None, None, 0)), in_axes=(None, 0, None)
    )

    two_body_jastrow_anti_parallel = jnp.sum(
        vmap_two_body_jastrow_anti_parallel_spins(jastrow_two_body_data.jastrow_2b_param, r_up_carts, r_dn_carts)
    )

    def compute_parallel_sum(r_carts):
        num_particles = r_carts.shape[0]
        idx_i, idx_j = jnp.triu_indices(num_particles, k=1)
        r_i = r_carts[idx_i]
        r_j = r_carts[idx_j]
        vmap_two_body_jastrow_parallel_spins = vmap(two_body_jastrow_parallel_spins_pade, in_axes=(None, 0, 0))(
            jastrow_two_body_data.jastrow_2b_param, r_i, r_j
        )
        return jnp.sum(vmap_two_body_jastrow_parallel_spins)

    two_body_jastrow_parallel_up = compute_parallel_sum(r_up_carts)
    two_body_jastrow_parallel_dn = compute_parallel_sum(r_dn_carts)

    two_body_jastrow = two_body_jastrow_anti_parallel + two_body_jastrow_parallel_up + two_body_jastrow_parallel_dn

    return two_body_jastrow


def _compute_Jastrow_two_body_debug(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """See _api method."""

    def two_body_jastrow_anti_parallel_spins_exp(
        param: float, r_cart_i: npt.NDArray[np.float64], r_cart_j: npt.NDArray[np.float64]
    ) -> float:
        """Exponential form of J2 for anti-parallel spins."""
        two_body_jastrow = 1.0 / (2.0 * param) * (1.0 - np.exp(-param * np.linalg.norm(r_cart_i - r_cart_j)))
        return two_body_jastrow

    def two_body_jastrow_parallel_spins_exp(
        param: float, r_cart_i: npt.NDArray[np.float64], r_cart_j: npt.NDArray[np.float64]
    ) -> float:
        """Exponential form of J2 for parallel spins."""
        two_body_jastrow = 1.0 / (2.0 * param) * (1.0 - np.exp(-param * np.linalg.norm(r_cart_i - r_cart_j)))
        return two_body_jastrow

    def two_body_jastrow_anti_parallel_spins_pade(
        param: float, r_cart_i: npt.NDArray[np.float64], r_cart_j: npt.NDArray[np.float64]
    ) -> float:
        """Pade form of J2 for anti-parallel spins."""
        two_body_jastrow = (
            np.linalg.norm(r_cart_i - r_cart_j) / 2.0 * (1.0 + param * np.linalg.norm(r_cart_i - r_cart_j)) ** (-1.0)
        )
        return two_body_jastrow

    def two_body_jastrow_parallel_spins_pade(
        param: float, r_cart_i: npt.NDArray[np.float64], r_cart_j: npt.NDArray[np.float64]
    ) -> float:
        """Pade form of J2 for parallel spins."""
        two_body_jastrow = (
            np.linalg.norm(r_cart_i - r_cart_j) / 2.0 * (1.0 + param * np.linalg.norm(r_cart_i - r_cart_j)) ** (-1.0)
        )
        return two_body_jastrow

    two_body_jastrow = (
        np.sum(
            [
                two_body_jastrow_anti_parallel_spins_pade(
                    param=jastrow_two_body_data.jastrow_2b_param,
                    r_cart_i=r_up_cart,
                    r_cart_j=r_dn_cart,
                )
                for (r_up_cart, r_dn_cart) in itertools.product(r_up_carts, r_dn_carts)
            ]
        )
        + np.sum(
            [
                two_body_jastrow_parallel_spins_pade(
                    param=jastrow_two_body_data.jastrow_2b_param,
                    r_cart_i=r_up_cart_i,
                    r_cart_j=r_up_cart_j,
                )
                for (r_up_cart_i, r_up_cart_j) in itertools.combinations(r_up_carts, 2)
            ]
        )
        + np.sum(
            [
                two_body_jastrow_parallel_spins_pade(
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
        j_matrix (npt.NDArray | jax.Array): J matrix with shape ``(orb_num, orb_num + 1)``.
    """

    orb_data: AOs_sphe_data | AOs_cart_data | MOs_data = struct.field(
        pytree_node=True, default_factory=AOs_sphe_data
    )  #: Orbital basis (AOs or MOs) shared across spins.
    j_matrix: npt.NDArray | jax.Array = struct.field(
        pytree_node=True, default_factory=lambda: np.array([])
    )  #: J3/J1 matrix; square block plus final column.

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
        if random_init:
            rng = np.random.default_rng(seed)
            j_matrix = rng.uniform(0.0, random_scale, size=(orb_data._num_orb, orb_data._num_orb + 1))
        else:
            j_matrix = np.zeros((orb_data._num_orb, orb_data._num_orb + 1))

        jastrow_three_body_data = cls(
            orb_data=orb_data,
            j_matrix=j_matrix,
        )
        return jastrow_three_body_data


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
        r_up_init = jnp.zeros((1, 3))
        r_dn_init = jnp.zeros((1, 3))
        R_n = jnp.asarray(structure_data.positions)  # (n_nuc, 3)
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
    """Evaluate the three-body Jastrow $J_3$ (analytic) without exponentiation.

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
    num_electron_up = len(r_up_carts)
    num_electron_dn = len(r_dn_carts)

    aos_up = jnp.array(jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, r_up_carts))
    aos_dn = jnp.array(jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, r_dn_carts))

    K_up = jnp.tril(jnp.ones((num_electron_up, num_electron_up)), k=-1)
    K_dn = jnp.tril(jnp.ones((num_electron_dn, num_electron_dn)), k=-1)

    j1_matrix_up = jastrow_three_body_data.j_matrix[:, -1]
    j1_matrix_dn = jastrow_three_body_data.j_matrix[:, -1]
    j3_matrix_up_up = jastrow_three_body_data.j_matrix[:, :-1]
    j3_matrix_dn_dn = jastrow_three_body_data.j_matrix[:, :-1]
    j3_matrix_up_dn = jastrow_three_body_data.j_matrix[:, :-1]

    e_up = jnp.ones(num_electron_up).T
    e_dn = jnp.ones(num_electron_dn).T

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
        j1 = self.jastrow_one_body_data
        j2 = self.jastrow_two_body_data
        j3 = self.jastrow_three_body_data
        nn3 = self.jastrow_nn_data

        if block.name == "j1_param" and j1 is not None:
            new_param = float(np.array(block.values).reshape(()))
            j1 = Jastrow_one_body_data(
                jastrow_1b_param=new_param,
                structure_data=j1.structure_data,
                core_electrons=j1.core_electrons,
            )
        elif block.name == "j2_param" and j2 is not None:
            new_param = float(np.array(block.values).reshape(()))
            j2 = Jastrow_two_body_data(jastrow_2b_param=new_param)
        elif block.name == "j3_matrix" and j3 is not None:
            # Enforce J3 structural constraints here. The last column corresponds
            # to the J1-like rectangular part, while the remaining square block
            # is kept symmetric when the original matrix is symmetric.
            j3_old = np.array(j3.j_matrix)
            j3_new = np.array(block.values)

            # Split into square + last-column parts
            square_old = j3_old[:, :-1]
            square_new = j3_new[:, :-1]

            # If the original square block is symmetric, enforce symmetry on the update
            if np.allclose(square_old, square_old.T, atol=1e-8):
                square_new = 0.5 * (square_new + square_new.T)
                j3_new[:, :-1] = square_new

            j3 = Jastrow_three_body_data(orb_data=j3.orb_data, j_matrix=j3_new)
        elif block.name == "jastrow_nn_params" and nn3 is not None:
            # Update NN Jastrow parameters: block.values is the flattened parameter vector.
            flat = jnp.asarray(block.values).reshape(-1)
            params_new = nn3.unflatten_fn(flat)
            nn3 = nn3.replace(params=params_new)

        return Jastrow_data(
            jastrow_one_body_data=j1,
            jastrow_two_body_data=j2,
            jastrow_three_body_data=j3,
            jastrow_nn_data=nn3,
        )

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
    r_up_carts = jnp.asarray(r_up_carts)
    r_dn_carts = jnp.asarray(r_dn_carts)

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

        R_n = jnp.asarray(nn3.structure_data.positions)
        Z_n = jnp.asarray(nn3.structure_data.atomic_numbers)
        J3_nn = nn3.nn_def.apply({"params": nn3.params}, r_up_carts, r_dn_carts, R_n, Z_n)
        J3 = J3 + J3_nn

    J = J1 + J2 + J3

    return J


def _compute_Jastrow_part_debug(
    jastrow_data: Jastrow_data, r_up_carts: npt.NDArray[np.float64], r_dn_carts: npt.NDArray[np.float64]
) -> float:
    """See compute_Jastrow_part_jax for more details."""
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

        R_n = np.asarray(nn3.structure_data.positions, dtype=float)
        Z_n = np.asarray(nn3.structure_data.atomic_numbers, dtype=float)

        # Use JAX NN for debug as well; convert inputs to jnp and back to float
        J3_nn = nn3.nn_def.apply(
            {"params": nn3.params}, jnp.asarray(r_up_carts), jnp.asarray(r_dn_carts), jnp.asarray(R_n), jnp.asarray(Z_n)
        )
        J3 += float(J3_nn)

    J = J1 + J2 + J3

    return J


def compute_ratio_Jastrow_part(
    jastrow_data: Jastrow_data,
    old_r_up_carts: jax.Array,
    old_r_dn_carts: jax.Array,
    new_r_up_carts_arr: jax.Array,
    new_r_dn_carts_arr: jax.Array,
) -> jax.Array:
    r"""Compute $\exp(J(\mathbf r'))/\exp(J(\mathbf r))$ for batched moves.

    This follows the original ratio logic (including exp) while updating types
    to use ``jax.Array`` inputs. The return is one ratio per proposed grid
    configuration.

    Args:
        jastrow_data: Active Jastrow components.
        old_r_up_carts: Reference spin-up coordinates with shape ``(N_up, 3)``.
        old_r_dn_carts: Reference spin-down coordinates with shape ``(N_dn, 3)``.
        new_r_up_carts_arr: Proposed spin-up coordinates with shape ``(N_grid, N_up, 3)``.
        new_r_dn_carts_arr: Proposed spin-down coordinates with shape ``(N_grid, N_dn, 3)``.

    Returns:
        jax.Array: Jastrow ratios per grid with shape ``(N_grid,)`` (includes ``exp``).
    """
    old_r_up_carts = jnp.asarray(old_r_up_carts)
    old_r_dn_carts = jnp.asarray(old_r_dn_carts)
    new_r_up_carts_arr = jnp.asarray(new_r_up_carts_arr)
    new_r_dn_carts_arr = jnp.asarray(new_r_dn_carts_arr)

    num_up = old_r_up_carts.shape[0]
    num_dn = old_r_dn_carts.shape[0]
    if num_up == 0 or num_dn == 0:
        jastrow_x = compute_Jastrow_part(jastrow_data, old_r_up_carts, old_r_dn_carts)
        jastrow_xp = vmap(compute_Jastrow_part, in_axes=(None, 0, 0))(jastrow_data, new_r_up_carts_arr, new_r_dn_carts_arr)
        return jnp.exp(jastrow_xp - jastrow_x)

    J_ratio = 1.0

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
                j1_new = compute_Jastrow_one_body(j1_data, jnp.zeros((0, 3)), jnp.expand_dims(r_dn_new, axis=0))
                j1_old = compute_Jastrow_one_body(j1_data, jnp.zeros((0, 3)), jnp.expand_dims(r_dn_old, axis=0))
                return jnp.exp(j1_new - j1_old)

        elif num_dn == 0:

            def compute_one_grid_J1(j1_data, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts):
                delta_up = new_r_up_carts - old_r_up_carts
                nonzero_up = jnp.any(delta_up != 0, axis=1)
                idx_up = jnp.argmax(nonzero_up)
                r_up_new = new_r_up_carts[idx_up]
                r_up_old = old_r_up_carts[idx_up]
                j1_new = compute_Jastrow_one_body(j1_data, jnp.expand_dims(r_up_new, axis=0), jnp.zeros((0, 3)))
                j1_old = compute_Jastrow_one_body(j1_data, jnp.expand_dims(r_up_old, axis=0), jnp.zeros((0, 3)))
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
                    j1_new = compute_Jastrow_one_body(j1_data, jnp.expand_dims(r_up_new, axis=0), jnp.zeros((0, 3)))
                    j1_old = compute_Jastrow_one_body(j1_data, jnp.expand_dims(r_up_old, axis=0), jnp.zeros((0, 3)))
                    return jnp.exp(j1_new - j1_old)

                def dn_case(args):
                    j1_data, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts = args
                    r_dn_new = new_r_dn_carts[idx_dn]
                    r_dn_old = old_r_dn_carts[idx_dn]
                    j1_new = compute_Jastrow_one_body(j1_data, jnp.zeros((0, 3)), jnp.expand_dims(r_dn_new, axis=0))
                    j1_old = compute_Jastrow_one_body(j1_data, jnp.zeros((0, 3)), jnp.expand_dims(r_dn_old, axis=0))
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

    def two_body_jastrow_anti_parallel_spins_exp(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> float:
        """Exponential form of J2 for anti-parallel spins."""
        two_body_jastrow = 1.0 / (2.0 * param) * (1.0 - jnp.exp(-param * jnp.linalg.norm(r_cart_i - r_cart_j)))
        return two_body_jastrow

    def two_body_jastrow_parallel_spins_exp(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> float:
        """Exponential form of J2 for parallel spins."""
        two_body_jastrow = 1.0 / (2.0 * param) * (1.0 - jnp.exp(-param * jnp.linalg.norm(r_cart_i - r_cart_j)))
        return two_body_jastrow

    def two_body_jastrow_anti_parallel_spins_pade(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> float:
        """Pade form of J2 for anti-parallel spins."""
        two_body_jastrow = (
            jnp.linalg.norm(r_cart_i - r_cart_j) / 2.0 * (1.0 + param * jnp.linalg.norm(r_cart_i - r_cart_j)) ** (-1.0)
        )
        return two_body_jastrow

    def two_body_jastrow_parallel_spins_pade(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> float:
        """Pade form of J2 for parallel spins."""
        two_body_jastrow = (
            jnp.linalg.norm(r_cart_i - r_cart_j) / 2.0 * (1.0 + param * jnp.linalg.norm(r_cart_i - r_cart_j)) ** (-1.0)
        )
        return two_body_jastrow

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
                vmap(two_body_jastrow_parallel_spins_pade, in_axes=(None, None, 0))(
                    jastrow_2b_param, new_r_up_carts_extracted, new_r_up_carts
                )
            )
            J2_up_up_old = jnp.sum(
                vmap(two_body_jastrow_parallel_spins_pade, in_axes=(None, None, 0))(
                    jastrow_2b_param, old_r_up_carts_extracted, old_r_up_carts
                )
            )
            J2_up_dn_new = jnp.sum(
                vmap(two_body_jastrow_anti_parallel_spins_pade, in_axes=(None, None, 0))(
                    jastrow_2b_param, new_r_up_carts_extracted, old_r_dn_carts
                )
            )
            J2_up_dn_old = jnp.sum(
                vmap(two_body_jastrow_anti_parallel_spins_pade, in_axes=(None, None, 0))(
                    jastrow_2b_param, old_r_up_carts_extracted, old_r_dn_carts
                )
            )
            return jnp.exp(J2_up_dn_new - J2_up_dn_old + J2_up_up_new - J2_up_up_old)

        def dn_case(jastrow_2b_param, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts):
            new_r_dn_carts_extracted = jnp.expand_dims(new_r_dn_carts[idx], axis=0)  # shape=(1,3)
            old_r_dn_carts_extracted = jnp.expand_dims(old_r_dn_carts[idx], axis=0)  # shape=(1,3)
            J2_dn_dn_new = jnp.sum(
                vmap(two_body_jastrow_parallel_spins_pade, in_axes=(None, None, 0))(
                    jastrow_2b_param, new_r_dn_carts_extracted, new_r_dn_carts
                )
            )
            J2_dn_dn_old = jnp.sum(
                vmap(two_body_jastrow_parallel_spins_pade, in_axes=(None, None, 0))(
                    jastrow_2b_param, old_r_dn_carts_extracted, old_r_dn_carts
                )
            )
            J2_up_dn_new = jnp.sum(
                vmap(two_body_jastrow_anti_parallel_spins_pade, in_axes=(None, 0, None))(
                    jastrow_2b_param, old_r_up_carts, new_r_dn_carts_extracted
                )
            )
            J2_up_dn_old = jnp.sum(
                vmap(two_body_jastrow_anti_parallel_spins_pade, in_axes=(None, 0, None))(
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

    def compute_one_grid_J3(
        jastrow_three_body_data, aos_up, aos_dn, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts
    ):
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

        num_electron_up = len(old_r_up_carts)
        num_electron_dn = len(old_r_dn_carts)
        j1_matrix_up = jastrow_three_body_data.j_matrix[:, -1]
        j1_matrix_dn = jastrow_three_body_data.j_matrix[:, -1]
        j3_matrix_up_up = jastrow_three_body_data.j_matrix[:, :-1]
        j3_matrix_dn_dn = jastrow_three_body_data.j_matrix[:, :-1]
        j3_matrix_up_dn = jastrow_three_body_data.j_matrix[:, :-1]
        e_up = jnp.ones(num_electron_up).T
        e_dn = jnp.ones(num_electron_dn).T

        def up_case(new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts):
            new_r_up_carts_extracted = jnp.expand_dims(new_r_up_carts[idx], axis=0)  # shape=(1,3)
            old_r_up_carts_extracted = jnp.expand_dims(old_r_up_carts[idx], axis=0)  # shape=(1,3)

            aos_up_p = jnp.array(
                jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, new_r_up_carts_extracted)
            ) - jnp.array(jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, old_r_up_carts_extracted))

            indices = jnp.arange(num_electron_up)
            Q_up_c = (idx < indices).astype(jnp.float64).reshape(-1, 1)
            Q_up_r = (idx > indices).astype(jnp.float64).reshape(1, -1)
            J3_ratio = jnp.exp(
                j1_matrix_up @ aos_up_p
                + jnp.trace(aos_up_p.T @ j3_matrix_up_up @ aos_up @ Q_up_c)
                + jnp.trace(aos_up.T @ j3_matrix_up_up @ aos_up_p @ Q_up_r)
                + aos_up_p.T @ j3_matrix_up_dn @ aos_dn @ e_dn
            )

            return J3_ratio

        def dn_case(new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts):
            new_r_dn_carts_extracted = jnp.expand_dims(new_r_dn_carts[idx], axis=0)  # shape=(1,3)
            old_r_dn_carts_extracted = jnp.expand_dims(old_r_dn_carts[idx], axis=0)  # shape=(1,3)

            aos_dn_p = jnp.array(
                jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, new_r_dn_carts_extracted)
            ) - jnp.array(jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, old_r_dn_carts_extracted))

            indices = jnp.arange(num_electron_dn)
            Q_dn_c = (idx < indices).astype(jnp.float64).reshape(-1, 1)
            Q_dn_r = (idx > indices).astype(jnp.float64).reshape(1, -1)
            J3_ratio = jnp.exp(
                j1_matrix_dn @ aos_dn_p
                + jnp.trace(aos_dn_p.T @ j3_matrix_dn_dn @ aos_dn @ Q_dn_c)
                + jnp.trace(aos_dn.T @ j3_matrix_dn_dn @ aos_dn_p @ Q_dn_r)
                + e_up.T @ aos_up.T @ j3_matrix_up_dn @ aos_dn_p
            )

            return J3_ratio

        if num_up == 0:
            return dn_case(new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts)
        if num_dn == 0:
            return up_case(new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts)

        return jax.lax.cond(
            up_moved,
            up_case,
            dn_case,
            *(new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts),
        )

    # J2 part
    if jastrow_data.jastrow_two_body_data is not None:
        j2_param = jastrow_data.jastrow_two_body_data.jastrow_2b_param

        def compute_pairwise_sums(pos1, pos2):
            if pos1.shape[0] == 0 or pos2.shape[0] == 0:
                return jnp.zeros(pos1.shape[0])
            dists = jnp.linalg.norm(pos1[:, None, :] - pos2[None, :, :], axis=-1)
            vals = dists / 2.0 * (1.0 + j2_param * dists) ** (-1.0)
            return jnp.sum(vals, axis=1)

        J2_sum_up_up = compute_pairwise_sums(old_r_up_carts, old_r_up_carts)
        J2_sum_up_dn = compute_pairwise_sums(old_r_up_carts, old_r_dn_carts)
        J2_sum_dn_dn = compute_pairwise_sums(old_r_dn_carts, old_r_dn_carts)
        J2_sum_dn_up = compute_pairwise_sums(old_r_dn_carts, old_r_up_carts)

        def compute_one_grid_J2(
            jastrow_2b_param,
            J2_sum_up_up,
            J2_sum_up_dn,
            J2_sum_dn_dn,
            J2_sum_dn_up,
            new_r_up_carts,
            new_r_dn_carts,
            old_r_up_carts,
            old_r_dn_carts,
        ):
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
                J2_up_up_new = jnp.sum(
                    vmap(two_body_jastrow_parallel_spins_pade, in_axes=(None, None, 0))(
                        jastrow_2b_param, new_r_up_carts_extracted, new_r_up_carts
                    )
                )
                J2_up_up_old = J2_sum_up_up[idx]

                J2_up_dn_new = jnp.sum(
                    vmap(two_body_jastrow_anti_parallel_spins_pade, in_axes=(None, None, 0))(
                        jastrow_2b_param, new_r_up_carts_extracted, old_r_dn_carts
                    )
                )
                J2_up_dn_old = J2_sum_up_dn[idx]
                return jnp.exp(J2_up_dn_new - J2_up_dn_old + J2_up_up_new - J2_up_up_old)

            def dn_case(jastrow_2b_param, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts):
                new_r_dn_carts_extracted = jnp.expand_dims(new_r_dn_carts[idx], axis=0)  # shape=(1,3)
                J2_dn_dn_new = jnp.sum(
                    vmap(two_body_jastrow_parallel_spins_pade, in_axes=(None, None, 0))(
                        jastrow_2b_param, new_r_dn_carts_extracted, new_r_dn_carts
                    )
                )
                J2_dn_dn_old = J2_sum_dn_dn[idx]

                J2_up_dn_new = jnp.sum(
                    vmap(two_body_jastrow_anti_parallel_spins_pade, in_axes=(None, 0, None))(
                        jastrow_2b_param, old_r_up_carts, new_r_dn_carts_extracted
                    )
                )
                J2_up_dn_old = J2_sum_dn_up[idx]

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

        # vectorization along grid
        J2_ratio = vmap(compute_one_grid_J2, in_axes=(None, None, None, None, None, 0, 0, None, None))(
            jastrow_data.jastrow_two_body_data.jastrow_2b_param,
            J2_sum_up_up,
            J2_sum_up_dn,
            J2_sum_dn_dn,
            J2_sum_dn_up,
            new_r_up_carts_arr,
            new_r_dn_carts_arr,
            old_r_up_carts,
            old_r_dn_carts,
        )

        J_ratio *= jnp.ravel(J2_ratio)

    # J3 part
    if jastrow_data.jastrow_three_body_data is not None:
        jastrow_three_body_data = jastrow_data.jastrow_three_body_data
        aos_up_old = jnp.array(jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, old_r_up_carts))
        aos_dn_old = jnp.array(jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, old_r_dn_carts))
        # vectorization along grid
        J3_ratio = vmap(compute_one_grid_J3, in_axes=(None, None, None, 0, 0, None, None))(
            jastrow_data.jastrow_three_body_data,
            aos_up_old,
            aos_dn_old,
            new_r_up_carts_arr,
            new_r_dn_carts_arr,
            old_r_up_carts,
            old_r_dn_carts,
        )

        J_ratio *= jnp.ravel(J3_ratio)

    # JNN part
    if jastrow_data.jastrow_nn_data is not None:
        nn = jastrow_data.jastrow_nn_data
        if nn.structure_data is None:
            raise ValueError("NN_Jastrow_data.structure_data must be set to evaluate NN J3.")

        R_n = jnp.asarray(nn.structure_data.positions)
        Z_n = jnp.asarray(nn.structure_data.atomic_numbers)

        def compute_one_grid_JNN(new_r_up_carts, new_r_dn_carts):
            return nn.nn_def.apply({"params": nn.params}, new_r_up_carts, new_r_dn_carts, R_n, Z_n)

        JNN_old = compute_one_grid_JNN(old_r_up_carts, old_r_dn_carts)
        JNN_new = vmap(compute_one_grid_JNN, in_axes=(0, 0))(new_r_up_carts_arr, new_r_dn_carts_arr)
        JNN_ratio = jnp.exp(JNN_new - JNN_old)
        J_ratio *= jnp.ravel(JNN_ratio)

    return J_ratio


def _compute_ratio_Jastrow_part_debug(
    jastrow_data: Jastrow_data,
    old_r_up_carts: npt.NDArray[np.float64],
    old_r_dn_carts: npt.NDArray[np.float64],
    new_r_up_carts_arr: npt.NDArray[np.float64],
    new_r_dn_carts_arr: npt.NDArray[np.float64],
) -> npt.NDArray:
    """See _api method."""
    return np.array(
        [
            np.exp(compute_Jastrow_part(jastrow_data, new_r_up_carts, new_r_dn_carts))
            / np.exp(compute_Jastrow_part(jastrow_data, old_r_up_carts, old_r_dn_carts))
            for new_r_up_carts, new_r_dn_carts in zip(new_r_up_carts_arr, new_r_dn_carts_arr, strict=True)
        ]
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
    """Per-electron gradients and Laplacians of the full Jastrow $J$.

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
    r_up = jnp.asarray(r_up_carts)
    r_dn = jnp.asarray(r_dn_carts)

    grad_J_up = jnp.zeros_like(r_up)
    grad_J_dn = jnp.zeros_like(r_dn)
    lap_J_up = jnp.zeros((r_up.shape[0],))
    lap_J_dn = jnp.zeros((r_dn.shape[0],))

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

        r_up_carts_jnp = jnp.asarray(r_up_carts)
        r_dn_carts_jnp = jnp.asarray(r_dn_carts)
        R_n = jnp.asarray(nn3.structure_data.positions)
        Z_n = jnp.asarray(nn3.structure_data.atomic_numbers)

        def _compute_Jastrow_nn_only(r_up, r_dn):
            return nn3.nn_def.apply({"params": nn3.params}, r_up, r_dn, R_n, Z_n)

        grad_JNN_up = grad(_compute_Jastrow_nn_only, argnums=0)(r_up_carts_jnp, r_dn_carts_jnp)
        grad_JNN_dn = grad(_compute_Jastrow_nn_only, argnums=1)(r_up_carts_jnp, r_dn_carts_jnp)

        hessian_JNN_up = hessian(_compute_Jastrow_nn_only, argnums=0)(r_up_carts_jnp, r_dn_carts_jnp)
        lap_JNN_up = jnp.einsum("ijij->i", hessian_JNN_up)

        hessian_JNN_dn = hessian(_compute_Jastrow_nn_only, argnums=1)(r_up_carts_jnp, r_dn_carts_jnp)
        lap_JNN_dn = jnp.einsum("ijij->i", hessian_JNN_dn)

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
    r_up_carts_jnp = jnp.array(r_up_carts)
    r_dn_carts_jnp = jnp.array(r_dn_carts)

    grad_J_up = jnp.zeros_like(r_up_carts_jnp)
    grad_J_dn = jnp.zeros_like(r_dn_carts_jnp)
    lap_J_up = jnp.zeros((r_up_carts_jnp.shape[0],))
    lap_J_dn = jnp.zeros((r_dn_carts_jnp.shape[0],))

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

        R_n = jnp.asarray(nn3.structure_data.positions)
        Z_n = jnp.asarray(nn3.structure_data.atomic_numbers)

        def _compute_Jastrow_nn_only(r_up, r_dn):
            return nn3.nn_def.apply({"params": nn3.params}, r_up, r_dn, R_n, Z_n)

        grad_JNN_up = grad(_compute_Jastrow_nn_only, argnums=0)(r_up_carts_jnp, r_dn_carts_jnp)
        grad_JNN_dn = grad(_compute_Jastrow_nn_only, argnums=1)(r_up_carts_jnp, r_dn_carts_jnp)

        hessian_JNN_up = hessian(_compute_Jastrow_nn_only, argnums=0)(r_up_carts_jnp, r_dn_carts_jnp)
        lap_JNN_up = jnp.einsum("ijij->i", hessian_JNN_up)

        hessian_JNN_dn = hessian(_compute_Jastrow_nn_only, argnums=1)(r_up_carts_jnp, r_dn_carts_jnp)
        lap_JNN_dn = jnp.einsum("ijij->i", hessian_JNN_dn)

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
    diff_h = 1.0e-5

    r_up_carts = np.array(r_up_carts, dtype=float)
    r_dn_carts = np.array(r_dn_carts, dtype=float)

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

    grad_J_up = np.array([grad_x_up, grad_y_up, grad_z_up]).T
    grad_J_dn = np.array([grad_x_dn, grad_y_dn, grad_z_dn]).T

    # laplacian
    diff_h2 = 1.0e-3
    J_ref = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)

    lap_J_up = np.zeros(len(r_up_carts), dtype=float)

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

    lap_J_dn = np.zeros(len(r_dn_carts), dtype=float)

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
    r_up_carts = jnp.array(r_up_carts)
    r_dn_carts = jnp.array(r_dn_carts)

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
    """Analytic gradients and Laplacians for the Pade two-body Jastrow.

    Uses the unchanged functional form ``J2(r) = r / (2 * (1 + a r))`` with
    ``a = jastrow_2b_param``. Returns per-electron quantities (not summed).

    Args:
        jastrow_two_body_data: Two-body Jastrow parameter container.
        r_up_carts: Spin-up electron coordinates with shape ``(N_up, 3)``.
        r_dn_carts: Spin-down electron coordinates with shape ``(N_dn, 3)``.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            Gradients for up/down electrons with shapes ``(N_up, 3)`` and ``(N_dn, 3)``,
            Laplacians for up/down electrons with shapes ``(N_up,)`` and ``(N_dn,)``.
    """
    a = jastrow_two_body_data.jastrow_2b_param
    eps = 1.0e-12

    r_up = jnp.asarray(r_up_carts)
    r_dn = jnp.asarray(r_dn_carts)

    num_up = r_up.shape[0]
    num_dn = r_dn.shape[0]

    grad_up = jnp.zeros_like(r_up)
    grad_dn = jnp.zeros_like(r_dn)
    lap_up = jnp.zeros((num_up,))
    lap_dn = jnp.zeros((num_dn,))

    def pair_terms(diff):
        r = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
        r = jnp.maximum(r, eps)
        denom = 1.0 + a * r
        f_prime = 0.5 / (denom * denom)
        grad_coeff = f_prime / r  # scalar per pair
        lap = -a / (denom * denom * denom) + (2.0 * f_prime) / r
        return grad_coeff[..., None] * diff, lap

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

    grad_J2_up = np.array([grad_x_up, grad_y_up, grad_z_up]).T
    grad_J2_dn = np.array([grad_x_dn, grad_y_dn, grad_z_dn]).T

    # laplacian
    diff_h2 = 1.0e-3  # for laplacian

    J2_ref = compute_Jastrow_two_body(
        jastrow_two_body_data=jastrow_two_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    lap_J2_up = np.zeros(len(r_up_carts), dtype=float)

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

    lap_J2_dn = np.zeros(len(r_dn_carts), dtype=float)

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
    # compute grad
    grad_J3_up = grad(compute_Jastrow_three_body, argnums=1)(jastrow_three_body_data, r_up_carts, r_dn_carts)

    grad_J3_dn = grad(compute_Jastrow_three_body, argnums=2)(jastrow_three_body_data, r_up_carts, r_dn_carts)

    # compute laplacians
    hessian_J3_up = hessian(compute_Jastrow_three_body, argnums=1)(jastrow_three_body_data, r_up_carts, r_dn_carts)
    laplacian_J3_up = jnp.einsum("ijij->i", hessian_J3_up)

    hessian_J3_dn = hessian(compute_Jastrow_three_body, argnums=2)(jastrow_three_body_data, r_up_carts, r_dn_carts)
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
    orb_data = jastrow_three_body_data.orb_data

    if isinstance(orb_data, MOs_data):
        compute_orb = compute_MOs
        compute_orb_grad = compute_MOs_grad
        compute_orb_lapl = compute_MOs_laplacian
    elif isinstance(orb_data, (AOs_sphe_data, AOs_cart_data)):
        compute_orb = compute_AOs
        compute_orb_grad = compute_AOs_grad
        compute_orb_lapl = compute_AOs_laplacian
    else:
        raise NotImplementedError

    r_up = jnp.asarray(r_up_carts)
    r_dn = jnp.asarray(r_dn_carts)

    aos_up = jnp.asarray(compute_orb(orb_data, r_up))  # (n_orb, n_up)
    aos_dn = jnp.asarray(compute_orb(orb_data, r_dn))  # (n_orb, n_dn)

    grad_up_x, grad_up_y, grad_up_z = compute_orb_grad(orb_data, r_up)
    grad_dn_x, grad_dn_y, grad_dn_z = compute_orb_grad(orb_data, r_dn)

    grad_up = jnp.stack([grad_up_x, grad_up_y, grad_up_z], axis=-1)  # (n_orb, n_up, 3)
    grad_dn = jnp.stack([grad_dn_x, grad_dn_y, grad_dn_z], axis=-1)  # (n_orb, n_dn, 3)

    lap_up = jnp.asarray(compute_orb_lapl(orb_data, r_up))  # (n_orb, n_up)
    lap_dn = jnp.asarray(compute_orb_lapl(orb_data, r_dn))  # (n_orb, n_dn)

    j1_vec = jnp.asarray(jastrow_three_body_data.j_matrix[:, -1])  # (n_orb,)
    j3_mat = jnp.asarray(jastrow_three_body_data.j_matrix[:, :-1])  # (n_orb, n_orb)

    num_up = aos_up.shape[1]
    num_dn = aos_dn.shape[1]

    # Precompute pair-accumulation masks
    upper_up = jnp.triu(jnp.ones((num_up, num_up)), k=1)
    lower_up = jnp.tril(jnp.ones((num_up, num_up)), k=-1)
    upper_dn = jnp.triu(jnp.ones((num_dn, num_dn)), k=1)
    lower_dn = jnp.tril(jnp.ones((num_dn, num_dn)), k=-1)

    # dJ/dA for each electron (orbital-space coefficients)
    g_up = (
        j1_vec[:, None]
        + jnp.dot(j3_mat, aos_up) @ lower_up
        + jnp.dot(j3_mat.T, aos_up) @ upper_up
        + jnp.dot(j3_mat, aos_dn) @ jnp.ones((num_dn, 1))
    )  # (n_orb, n_up)

    g_dn = (
        j1_vec[:, None]
        + jnp.dot(j3_mat, aos_dn) @ lower_dn
        + jnp.dot(j3_mat.T, aos_dn) @ upper_dn
        + jnp.dot(j3_mat.T, aos_up) @ jnp.ones((num_up, 1))
    )  # (n_orb, n_dn)

    grad_J3_up = jnp.einsum("on,onj->nj", g_up, grad_up)
    grad_J3_dn = jnp.einsum("on,onj->nj", g_dn, grad_dn)

    lap_up_contrib = jnp.einsum("on,on->n", g_up, lap_up)
    lap_dn_contrib = jnp.einsum("on,on->n", g_dn, lap_dn)

    return grad_J3_up, grad_J3_dn, lap_up_contrib, lap_dn_contrib


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

    grad_J3_up = np.array([grad_x_up, grad_y_up, grad_z_up]).T
    grad_J3_dn = np.array([grad_x_dn, grad_y_dn, grad_z_dn]).T

    # laplacian
    diff_h2 = 1.0e-3  # for laplacian

    J3_ref = compute_Jastrow_three_body(
        jastrow_three_body_data=jastrow_three_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    lap_J3_up = np.zeros(len(r_up_carts), dtype=float)

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

    lap_J3_dn = np.zeros(len(r_dn_carts), dtype=float)

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
