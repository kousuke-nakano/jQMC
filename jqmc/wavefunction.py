"""Wavefunction module.

Precision Zones:
    - ``kinetic``: kinetic-energy evaluation (compute_kinetic_energy*).
    - Zone-boundary casts when combining results from ``jastrow`` and
      ``determinant`` zones.

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
# from dataclasses import dataclass
from logging import getLogger

# import jax
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import struct
from jax import grad, hessian, jit, tree_util, vmap
from jax import typing as jnpt

from ._diff_mask import DiffMask, apply_diff_mask
from ._precision import get_dtype_jnp
from .atomic_orbital import AOs_cart_data, AOs_sphe_data, ShellPrimMap
from .determinant import (
    Det_streaming_state,
    Geminal_data,
    _advance_grads_laplacian_ln_Det_streaming_state,
    _compute_ratio_determinant_part_split_spin,
    _init_grads_laplacian_ln_Det_streaming_state,
    compute_det_geminal_all_elements,
    compute_grads_and_laplacian_ln_Det,
    compute_grads_and_laplacian_ln_Det_fast,
    compute_ln_det_geminal_all_elements,
    compute_ln_det_geminal_all_elements_fast,
)
from .jastrow_factor import (
    Jastrow_data,
    Jastrow_one_body_streaming_state,
    Jastrow_three_body_streaming_state,
    Jastrow_two_body_streaming_state,
    _advance_grads_laplacian_Jastrow_one_body_streaming_state,
    _advance_grads_laplacian_Jastrow_three_body_streaming_state,
    _advance_grads_laplacian_Jastrow_two_body_streaming_state,
    _compute_ratio_Jastrow_part_rank1_update,
    _init_grads_laplacian_Jastrow_one_body_streaming_state,
    _init_grads_laplacian_Jastrow_three_body_streaming_state,
    _init_grads_laplacian_Jastrow_two_body_streaming_state,
    compute_grads_and_laplacian_Jastrow_one_body,
    compute_grads_and_laplacian_Jastrow_part,
    compute_grads_and_laplacian_Jastrow_two_body,
    compute_Jastrow_part,
)
from .molecular_orbital import MOs_data

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Shell-constraint helpers for AO basis optimization
# ---------------------------------------------------------------------------


def _get_aos_data(orb_data):
    """Extract the underlying AOs_*_data from orb_data (AO or MO)."""
    if isinstance(orb_data, (AOs_sphe_data, AOs_cart_data)):
        return orb_data
    if isinstance(orb_data, MOs_data):
        return orb_data.aos_data
    raise NotImplementedError(f"Unsupported orb_data type: {type(orb_data)}")


@struct.dataclass
class VariationalParameterBlock:
    """A block of variational parameters (e.g., J1, J2, J3, lambda).

    Design overview
    ----------------
    * A *block* is the smallest unit that the optimizer (MCMC + SR) sees.
      Each block corresponds to a contiguous slice in the global
      variational parameter vector and carries enough metadata to
      reconstruct its original shape (name, values, shape, size).
    * This class is intentionally **structure-agnostic**: it does not
      know anything about Jastrow vs Geminal, matrix symmetry, or how a
      block maps to concrete fields in :class:`Jastrow_data` or
      :class:`Geminal_data`.
    * All physics- and structure-specific semantics are owned by the
      corresponding data classes via their ``get_variational_blocks`` and
      ``apply_block_update`` methods.

    The goal is that adding or modifying a variational parameter only
    requires changes on the wavefunction side (Jastrow/Geminal data),
    while the MCMC/SR driver remains completely agnostic and operates
    purely on a list of blocks.
    """

    name: str  #: Identifier for this block (for example ``"j1_param"`` or ``"lambda_matrix"``).
    values: jnpt.ArrayLike = struct.field(pytree_node=True)  #: Parameter payload (keeps PyTree structure if present).
    shape: tuple[int, ...] = struct.field(pytree_node=False)  #: Original shape of ``values`` for unflattening updates.
    size: int = struct.field(pytree_node=False)  #: Flattened size of ``values`` used when slicing the global vector.
    symmetrize_metric: object = struct.field(
        pytree_node=False, default=None
    )  #: Optional callback ``flat_array -> flat_array`` (wraps a ``matrix -> matrix`` callback from the data class with flatten/unflatten).

    def apply_update(self, delta_flat: npt.NDArray, learning_rate: float) -> "VariationalParameterBlock":
        r"""Return a new block with values updated by a generic additive rule.

        This method is intentionally *structure-agnostic* and only performs a
        simple additive update::

            X_new = X_old + learning_rate * delta

        Any parameter-specific constraints (e.g., symmetry of J3 or
        ``lambda_matrix``) must be enforced by the owner of the parameter
        (``jastrow_data``, ``geminal_data``, etc.) inside their
        ``apply_block_update`` implementations.

        Args:
            delta_flat: Flattened update vector with length equal to ``size``.
            learning_rate: Scaling factor for the update.
        """
        dX = delta_flat.reshape(self.shape)
        new_values = np.array(self.values) + learning_rate * dX

        return VariationalParameterBlock(
            name=self.name,
            values=new_values,
            shape=new_values.shape,
            size=new_values.size,
        )


def _make_batch_symmetrize_j3(jastrow_data, shape):
    """Create a batch-aware symmetrize function for j3_matrix.

    Works on 1D (block_size,) or 2D (batch, block_size) input.
    The symmetry check (is the current j3 square sub-block symmetric?)
    is evaluated once at creation time.
    """
    from .jastrow_factor import atol_consistency

    j3 = jastrow_data.jastrow_three_body_data
    # Determine if symmetrization applies (same check as symmetrize_j3)
    _do_sym = False
    if j3 is not None:
        j3_arr = np.asarray(j3.j_matrix)
        if j3_arr.ndim == 2 and j3_arr.shape[1] >= 2:
            sq = j3_arr[:, :-1]
            if sq.shape[0] == sq.shape[1] and np.allclose(sq, sq.T, atol=atol_consistency):
                _do_sym = True
    _n_cols_sq = shape[1] - 1 if len(shape) == 2 else 0

    def _symmetrize(arr):
        if not _do_sym:
            return arr
        if arr.ndim == 1:
            mat = arr.reshape(shape)
            sq = mat[:, :_n_cols_sq]
            mat[:, :_n_cols_sq] = 0.5 * (sq + sq.T)
            return mat.ravel()
        # batch: arr shape (batch, block_size)
        batch = arr.reshape(arr.shape[0], *shape)
        sq = batch[:, :, :_n_cols_sq]
        batch[:, :, :_n_cols_sq] = 0.5 * (sq + np.swapaxes(sq, -2, -1))
        return batch.reshape(arr.shape)

    return _symmetrize


def _make_batch_symmetrize_lambda(geminal_data, shape):
    """Create a batch-aware symmetrize function for lambda_matrix.

    Works on 1D (block_size,) or 2D (batch, block_size) input.
    The symmetry check is evaluated once at creation time.
    """
    from .determinant import atol_consistency, rtol_consistency

    lam = np.asarray(geminal_data.lambda_matrix) if geminal_data.lambda_matrix is not None else None
    _do_sym = False
    _n_paired = 0
    if lam is not None and lam.ndim == 2:
        if lam.shape[0] == lam.shape[1]:
            if np.allclose(lam, lam.T, atol=atol_consistency, rtol=rtol_consistency):
                _do_sym = True
                _n_paired = lam.shape[0]
        else:
            _n_paired = lam.shape[0]
            paired = lam[:, :_n_paired]
            if np.allclose(paired, paired.T, atol=atol_consistency, rtol=rtol_consistency):
                _do_sym = True

    def _symmetrize(arr):
        if not _do_sym:
            return arr
        if arr.ndim == 1:
            mat = arr.reshape(shape)
            p = mat[:, :_n_paired]
            mat[:, :_n_paired] = 0.5 * (p + p.T)
            return mat.ravel()
        # batch: arr shape (batch, block_size)
        batch = arr.reshape(arr.shape[0], *shape)
        p = batch[:, :, :_n_paired]
        batch[:, :, :_n_paired] = 0.5 * (p + np.swapaxes(p, -2, -1))
        return batch.reshape(arr.shape)

    return _symmetrize


@struct.dataclass
class Wavefunction_data:
    """Container for Jastrow and Geminal parts used to evaluate a wavefunction.

    The class owns only the data needed to construct the wavefunction. All
    computations are delegated to the functions in this module and the
    underlying Jastrow/Geminal helpers.

    Args:
        jastrow_data: Optional Jastrow parameters. If ``None``, the Jastrow part is omitted.
        geminal_data: Optional Geminal parameters. If ``None``, the determinant part is omitted.
    """

    jastrow_data: Jastrow_data = struct.field(
        pytree_node=True, default_factory=Jastrow_data
    )  #: Variational Jastrow parameters.
    geminal_data: Geminal_data = struct.field(
        pytree_node=True, default_factory=Geminal_data
    )  #: Variational Geminal/determinant parameters.

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        self.jastrow_data.sanity_check()
        self.geminal_data.sanity_check()

    def _get_info(self) -> list[str]:
        """Return a list of strings representing the logged information."""
        info_lines = []
        # Replace geminal_data.logger_info() with geminal_data.get_info() output.
        info_lines.extend(self.geminal_data._get_info())
        # Replace jastrow_data.logger_info() with jastrow_data.get_info() output.
        info_lines.extend(self.jastrow_data._get_info())
        return info_lines

    def _logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self._get_info():
            logger.info(line)

    def apply_block_updates(
        self,
        blocks: list[VariationalParameterBlock],
        thetas: npt.NDArray,
        learning_rate: float,
    ) -> "Wavefunction_data":
        """Return a new :class:`Wavefunction_data` with variational blocks updated.

        Design notes
        ------------
        * ``blocks`` defines the ordering and shapes of all variational
          parameters; ``thetas`` is a single flattened update vector in
          the same order.
        * This method is responsible for slicing ``thetas`` into
          per-block pieces and performing a generic additive update via
          :meth:`VariationalParameterBlock.apply_update`.
        * The *interpretation* of each block ("this is J1", "this is the
          J3 matrix", "this is lambda") and any structural constraints
          (symmetry, rectangular layout, etc.) are delegated to
          :meth:`Jastrow_data.apply_block_update` and
          :meth:`Geminal_data.apply_block_update`.

        Because of this separation of concerns, the MCMC/SR driver only
        needs to work with the flattened ``thetas`` vector and the list of
        blocks; it never touches Jastrow/Geminal internals directly. To
        add a new parameter to the optimization, one only needs to
        (1) expose it in :meth:`get_variational_blocks`, and
        (2) handle it in the corresponding ``apply_block_update`` method.
        """
        jastrow_data = self.jastrow_data
        geminal_data = self.geminal_data

        pos = 0
        for block in blocks:
            start = pos
            end = pos + block.size
            pos = end
            delta_flat = thetas[start:end]
            if np.all(delta_flat == 0.0):
                continue

            updated_block = block.apply_update(delta_flat, learning_rate=learning_rate)

            # Delegate the mapping from block to internal parameters to
            # Jastrow_data and Geminal_data.
            if jastrow_data is not None:
                jastrow_data = jastrow_data.apply_block_update(updated_block)
            if geminal_data is not None:
                geminal_data = geminal_data.apply_block_update(updated_block)

        return Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data)

    def with_diff_mask(self, *, params: bool = True, coords: bool = True) -> "Wavefunction_data":
        """Return a copy with gradients masked according to the provided flags."""
        return apply_diff_mask(self, DiffMask(params=params, coords=coords))

    def with_param_grad_mask(
        self,
        *,
        opt_J1_param: bool = True,
        opt_J2_param: bool = True,
        opt_J3_param: bool = True,
        opt_JNN_param: bool = True,
        opt_lambda_param: bool = True,
        opt_J3_basis_exp: bool = False,
        opt_J3_basis_coeff: bool = False,
        opt_lambda_basis_exp: bool = False,
        opt_lambda_basis_coeff: bool = False,
    ) -> "Wavefunction_data":
        """Return a copy where disabled parameter blocks stop propagating gradients.

        Developer note
        --------------
        * The per-block flags (``opt_J1_param`` etc.) decide which high-level blocks are
            masked. Disabled blocks are wrapped with ``DiffMask(params=False, coords=True)``,
            meaning parameter gradients are stopped while coordinate gradients still flow.
        * Within each disabled block, ``apply_diff_mask`` uses field-name heuristics
            (see ``diff_mask._PARAM_FIELD_NAMES``) to tag parameter leaves such as
            ``lambda_matrix``, ``j_matrix``, ``jastrow_1b_param``, ``jastrow_2b_param``,
            ``jastrow_3b_param``, and ``params``. Those tagged leaves receive
            ``jax.lax.stop_gradient``, so their backpropagated gradients become zero.
        * AO basis flags (``opt_J3_basis_exp`` etc.) independently control whether
            ``exponents`` / ``coefficients`` gradients flow. These are handled as a
            separate masking step so they are fully independent of the high-level block
            flags.
        * Example: if ``opt_J1_param=False`` and others are True, only the J1 block is
            masked; its parameter leaves are stopped, while J2/J3/NN/lambda continue to
            propagate gradients normally.
        """
        mask_off = DiffMask(params=False, coords=True)

        def _maybe_mask(block, enabled):
            if enabled or block is None:
                return block, False
            return apply_diff_mask(block, mask_off), True

        jastrow_data = self.jastrow_data
        jastrow_updates = {}
        if jastrow_data is not None:
            j1_block, changed = _maybe_mask(jastrow_data.jastrow_one_body_data, opt_J1_param)
            if changed:
                jastrow_updates["jastrow_one_body_data"] = j1_block

            j2_block, changed = _maybe_mask(jastrow_data.jastrow_two_body_data, opt_J2_param)
            if changed:
                jastrow_updates["jastrow_two_body_data"] = j2_block

            j3_block, changed = _maybe_mask(jastrow_data.jastrow_three_body_data, opt_J3_param)
            if changed:
                jastrow_updates["jastrow_three_body_data"] = j3_block

            jnn_block, changed = _maybe_mask(jastrow_data.jastrow_nn_data, opt_JNN_param)
            if changed:
                jastrow_updates["jastrow_nn_data"] = jnn_block

            if jastrow_updates:
                jastrow_data = jastrow_data.replace(**jastrow_updates)

            # AO basis masking for J3: stop gradient on exponents/coefficients when not optimized.
            # This is independent of the J3 param (j_matrix) masking above.
            j3d = jastrow_data.jastrow_three_body_data
            if j3d is not None:
                j3_orb_updates = {}
                if not opt_J3_basis_exp:
                    j3_orb_updates["exponents"] = jax.lax.stop_gradient(j3d.ao_exponents)
                if not opt_J3_basis_coeff:
                    j3_orb_updates["coefficients"] = jax.lax.stop_gradient(j3d.ao_coefficients)
                if j3_orb_updates:
                    if isinstance(j3d.orb_data, MOs_data):
                        new_aos = j3d.orb_data.aos_data.replace(**j3_orb_updates)
                        new_orb = j3d.orb_data.replace(aos_data=new_aos)
                    else:
                        new_orb = j3d.orb_data.replace(**j3_orb_updates)
                    jastrow_data = jastrow_data.replace(jastrow_three_body_data=j3d.replace(orb_data=new_orb))

        geminal_data = self.geminal_data
        geminal_updates = {}
        if geminal_data is not None:
            geminal_masked, changed = _maybe_mask(geminal_data, opt_lambda_param)
            if changed:
                geminal_updates["lambda_matrix"] = geminal_masked.lambda_matrix

            if geminal_updates:
                geminal_data = geminal_data.replace(**geminal_updates)

            # AO basis masking for Geminal: stop gradient on exponents/coefficients when not optimized.
            if not opt_lambda_basis_exp or not opt_lambda_basis_coeff:
                orb_up_updates = {}
                orb_dn_updates = {}
                if not opt_lambda_basis_exp:
                    orb_up_updates["exponents"] = jax.lax.stop_gradient(geminal_data.ao_exponents_up)
                    orb_dn_updates["exponents"] = jax.lax.stop_gradient(geminal_data.ao_exponents_dn)
                if not opt_lambda_basis_coeff:
                    orb_up_updates["coefficients"] = jax.lax.stop_gradient(geminal_data.ao_coefficients_up)
                    orb_dn_updates["coefficients"] = jax.lax.stop_gradient(geminal_data.ao_coefficients_dn)

                up_orb = geminal_data.orb_data_up_spin
                dn_orb = geminal_data.orb_data_dn_spin
                if isinstance(up_orb, MOs_data):
                    new_aos_up = up_orb.aos_data.replace(**orb_up_updates)
                    up_orb = up_orb.replace(aos_data=new_aos_up)
                else:
                    up_orb = up_orb.replace(**orb_up_updates)
                if isinstance(dn_orb, MOs_data):
                    new_aos_dn = dn_orb.aos_data.replace(**orb_dn_updates)
                    dn_orb = dn_orb.replace(aos_data=new_aos_dn)
                else:
                    dn_orb = dn_orb.replace(**orb_dn_updates)
                geminal_data = geminal_data.replace(
                    orb_data_up_spin=up_orb,
                    orb_data_dn_spin=dn_orb,
                )

        if jastrow_data is not self.jastrow_data or geminal_data is not self.geminal_data:
            return Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data)

        return self

    def accumulate_position_grad(self, grad_wavefunction: "Wavefunction_data"):
        """Aggregate position gradients from geminal and Jastrow parts."""
        grad = 0.0
        if self.geminal_data is not None and grad_wavefunction.geminal_data is not None:
            grad += self.geminal_data.accumulate_position_grad(grad_wavefunction.geminal_data)
        if self.jastrow_data is not None and grad_wavefunction.jastrow_data is not None:
            grad += self.jastrow_data.accumulate_position_grad(grad_wavefunction.jastrow_data)
        return grad

    def collect_param_grads(self, grad_wavefunction: "Wavefunction_data") -> dict[str, object]:
        """Collect parameter gradients from Jastrow and Geminal into a flat dict."""
        grads: dict[str, object] = {}
        if self.jastrow_data is not None and grad_wavefunction.jastrow_data is not None:
            grads.update(self.jastrow_data.collect_param_grads(grad_wavefunction.jastrow_data))
        if self.geminal_data is not None and grad_wavefunction.geminal_data is not None:
            grads.update(self.geminal_data.collect_param_grads(grad_wavefunction.geminal_data))
        return grads

    def flatten_param_grads(self, param_grads: dict[str, object], num_walkers: int) -> dict[str, np.ndarray]:
        """Return parameter gradients as numpy arrays ready for storage.

        The caller does not need to know the internal block structure (e.g., NN trees);
        any necessary flattening is handled here.
        """
        flat: dict[str, np.ndarray] = {}
        jastrow_nn_data = self.jastrow_data.jastrow_nn_data if self.jastrow_data is not None else None

        for name, param_grad in param_grads.items():
            if name == "jastrow_nn_params" and jastrow_nn_data is not None:

                def _slice_walker(idx):
                    return tree_util.tree_map(lambda x: x[idx], param_grad)

                nn_grad_list = []
                for walker_idx in range(num_walkers):
                    walker_grad_tree = _slice_walker(walker_idx)
                    flat_vec = np.array(jastrow_nn_data.flatten_fn(walker_grad_tree))
                    nn_grad_list.append(flat_vec)

                flat[name] = np.stack(nn_grad_list, axis=0)
            else:
                flat[name] = np.array(param_grad)

        return flat

    def get_variational_blocks(
        self,
        opt_J1_param: bool = True,
        opt_J2_param: bool = True,
        opt_J3_param: bool = True,
        opt_JNN_param: bool = True,
        opt_lambda_param: bool = False,
        opt_J3_basis_exp: bool = False,
        opt_J3_basis_coeff: bool = False,
        opt_lambda_basis_exp: bool = False,
        opt_lambda_basis_coeff: bool = False,
    ) -> list[VariationalParameterBlock]:
        """Collect variational parameter blocks from Jastrow and Geminal parts.

        Each block corresponds to a contiguous group of variational parameters
        (e.g., J1, J2, J3 matrix, NN Jastrow, lambda matrix). This method only exposes the
        parameter arrays; the corresponding gradients are handled on the MCMC side.
        """
        blocks: list[VariationalParameterBlock] = []

        # Jastrow part
        if self.jastrow_data is not None:
            if opt_J1_param and self.jastrow_data.jastrow_one_body_data is not None:
                j1 = self.jastrow_data.jastrow_one_body_data.jastrow_1b_param
                j1_arr = np.asarray(j1)
                blocks.append(
                    VariationalParameterBlock(
                        name="j1_param",
                        values=j1_arr,
                        shape=j1_arr.shape if hasattr(j1_arr, "shape") else (),
                        size=int(j1_arr.size) if hasattr(j1_arr, "size") else 1,
                    )
                )

            if opt_J2_param and self.jastrow_data.jastrow_two_body_data is not None:
                j2 = self.jastrow_data.jastrow_two_body_data.jastrow_2b_param
                j2_arr = np.asarray(j2)
                blocks.append(
                    VariationalParameterBlock(
                        name="j2_param",
                        values=j2_arr,
                        shape=j2_arr.shape if hasattr(j2_arr, "shape") else (),
                        size=int(j2_arr.size) if hasattr(j2_arr, "size") else 1,
                    )
                )

            if opt_J3_param and self.jastrow_data.jastrow_three_body_data is not None:
                j3 = self.jastrow_data.jastrow_three_body_data.j_matrix
                j3_arr = np.asarray(j3)
                _jd = self.jastrow_data
                blocks.append(
                    VariationalParameterBlock(
                        name="j3_matrix",
                        values=j3_arr,
                        shape=j3_arr.shape,
                        size=int(j3_arr.size),
                        symmetrize_metric=_make_batch_symmetrize_j3(_jd, j3_arr.shape),
                    )
                )

            # J3 AO basis blocks
            if (opt_J3_basis_exp or opt_J3_basis_coeff) and self.jastrow_data.jastrow_three_body_data is not None:
                j3_data = self.jastrow_data.jastrow_three_body_data
                j3_spm = ShellPrimMap.from_aos_data(_get_aos_data(j3_data.orb_data))

                if opt_J3_basis_exp:
                    exp_arr = np.asarray(j3_data.ao_exponents)
                    blocks.append(
                        VariationalParameterBlock(
                            name="j3_basis_exp",
                            values=exp_arr,
                            shape=exp_arr.shape,
                            size=int(exp_arr.size),
                            symmetrize_metric=j3_spm.symmetrize,
                        )
                    )

                if opt_J3_basis_coeff:
                    coeff_arr = np.asarray(j3_data.ao_coefficients)
                    blocks.append(
                        VariationalParameterBlock(
                            name="j3_basis_coeff",
                            values=coeff_arr,
                            shape=coeff_arr.shape,
                            size=int(coeff_arr.size),
                            symmetrize_metric=j3_spm.symmetrize,
                        )
                    )

            if opt_JNN_param and self.jastrow_data.jastrow_nn_data is not None:
                nn3 = self.jastrow_data.jastrow_nn_data
                if nn3.params is not None and nn3.num_params > 0:
                    flat_params = np.array(nn3.flatten_fn(nn3.params))
                    blocks.append(
                        VariationalParameterBlock(
                            name="jastrow_nn_params",
                            values=flat_params,
                            shape=flat_params.shape,
                            size=int(flat_params.size),
                        )
                    )

        # Geminal part
        if opt_lambda_param and self.geminal_data is not None and self.geminal_data.lambda_matrix is not None:
            lam = self.geminal_data.lambda_matrix
            lam_arr = np.asarray(lam)
            _gd = self.geminal_data
            blocks.append(
                VariationalParameterBlock(
                    name="lambda_matrix",
                    values=lam_arr,
                    shape=lam_arr.shape,
                    size=int(lam_arr.size),
                    symmetrize_metric=_make_batch_symmetrize_lambda(_gd, lam_arr.shape),
                )
            )

        # Geminal AO basis blocks (up + dn concatenated into single blocks)
        if self.geminal_data is not None:
            if opt_lambda_basis_exp or opt_lambda_basis_coeff:
                lam_spm = ShellPrimMap.concat(
                    ShellPrimMap.from_aos_data(_get_aos_data(self.geminal_data.orb_data_up_spin)),
                    ShellPrimMap.from_aos_data(_get_aos_data(self.geminal_data.orb_data_dn_spin)),
                )

            if opt_lambda_basis_exp:
                lam_exp_arr = np.concatenate(
                    [
                        np.asarray(self.geminal_data.ao_exponents_up),
                        np.asarray(self.geminal_data.ao_exponents_dn),
                    ]
                )
                blocks.append(
                    VariationalParameterBlock(
                        name="lambda_basis_exp",
                        values=lam_exp_arr,
                        shape=lam_exp_arr.shape,
                        size=int(lam_exp_arr.size),
                        symmetrize_metric=lam_spm.symmetrize,
                    )
                )

            if opt_lambda_basis_coeff:
                lam_coeff_arr = np.concatenate(
                    [
                        np.asarray(self.geminal_data.ao_coefficients_up),
                        np.asarray(self.geminal_data.ao_coefficients_dn),
                    ]
                )
                blocks.append(
                    VariationalParameterBlock(
                        name="lambda_basis_coeff",
                        values=lam_coeff_arr,
                        shape=lam_coeff_arr.shape,
                        size=int(lam_coeff_arr.size),
                        symmetrize_metric=lam_spm.symmetrize,
                    )
                )

        return blocks


@jit
def evaluate_ln_wavefunction(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float:
    r"""Evaluate the logarithm of ``|wavefunction|`` (:math:`\ln |\Psi|`).

    This follows the original behavior: compute the Jastrow part, multiply the
    determinant part, and then take ``log(abs(det))`` while keeping the full
    Jastrow contribution. The inputs are converted to the determinant zone dtype
    ``jax.Array`` for downstream consistency.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons with shape ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons with shape ``(n_dn, 3)``.

    Returns:
        Scalar log-value of the wavefunction magnitude.
    """
    # NOTE: do not pre-cast r_*_carts. They are forwarded unchanged to
    # ``compute_Jastrow_part`` and ``compute_det_geminal_all_elements`` (which
    # ultimately reach the AO kernels that reconstruct ``r - R`` in float64);
    # a wrapper-level downcast would defeat that precision guard. The scalar
    # arithmetic below uses ``Jastrow_part`` and ``Determinant_part`` which
    # are explicitly cast to ``wf_eval``.
    dtype_wf_jnp = get_dtype_jnp("wf_eval")

    Jastrow_part = compute_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    Determinant_part = compute_det_geminal_all_elements(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # Consumer-zone explicit cast: both terms cast to wf_eval before addition,
    # never relying on implicit fp32 + fp64 -> fp64 promotion.
    return jnp.asarray(Jastrow_part, dtype=dtype_wf_jnp) + jnp.asarray(jnp.log(jnp.abs(Determinant_part)), dtype=dtype_wf_jnp)


def evaluate_ln_wavefunction_fast(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    geminal_inv: jax.Array,
) -> float:
    r"""Evaluate :math:`\ln|\Psi|` using pre-computed ``geminal_inv`` in gradients.

    Identical to :func:`evaluate_ln_wavefunction` in the forward direction.
    The backward pass (used when computing :math:`\partial\ln\Psi/\partial c`
    via JAX autodiff) replaces the fresh LU decomposition of the geminal matrix
    with ``geminal_inv`` — the Sherman-Morrison running inverse — so that
    near-singular configurations (``epsilon_AS > 0``) do not produce NaN
    gradients.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons ``(n_dn, 3)``.
        geminal_inv: Pre-computed inverse geminal matrix ``(N_up, N_up)`` from
            the Sherman-Morrison running update, valid at the supplied
            ``(r_up_carts, r_dn_carts)``.

    Returns:
        Scalar log-value of the wavefunction magnitude.

    Warning:
        ``geminal_inv`` **must** equal ``G(r_up_carts, r_dn_carts)^{-1}``
        exactly at the supplied electron positions.  Correctness is only
        guaranteed when the inverse is maintained via **single-electron
        (rank-1) Sherman-Morrison updates** starting from a freshly
        initialized LU inverse — the pattern used in the MCMC loop.
        Passing an inverse from a different configuration silently produces
        incorrect parameter gradients (``O_matrix`` / SR).
    """
    # r_*_carts forwarded unchanged (see ``evaluate_ln_wavefunction`` for rationale).
    dtype_wf_jnp = get_dtype_jnp("wf_eval")

    Jastrow_part = compute_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    ln_det = compute_ln_det_geminal_all_elements_fast(
        wavefunction_data.geminal_data,
        r_up_carts,
        r_dn_carts,
        geminal_inv,
    )

    # Consumer-zone explicit cast: both terms cast to wf_eval before addition.
    return jnp.asarray(Jastrow_part, dtype=dtype_wf_jnp) + jnp.asarray(ln_det, dtype=dtype_wf_jnp)


@jit
def evaluate_wavefunction(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float | complex:
    """Evaluate the wavefunction ``Psi`` at given electron coordinates.

    The method is for evaluate wavefunction (Psi) at ``(r_up_carts, r_dn_carts)`` and
    returns ``exp(Jastrow) * Determinant``. Inputs are coerced to the determinant
    zone dtype ``jax.Array`` to match other compute utilities.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons with shape ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons with shape ``(n_dn, 3)``.

    Returns:
        Complex or real wavefunction value.
    """
    # r_*_carts forwarded unchanged (see ``evaluate_ln_wavefunction`` for rationale).
    dtype_wf_jnp = get_dtype_jnp("wf_eval")

    Jastrow_part = compute_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    Determinant_part = compute_det_geminal_all_elements(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # Consumer-zone explicit cast: both factors cast to wf_eval before multiplication.
    return jnp.exp(jnp.asarray(Jastrow_part, dtype=dtype_wf_jnp)) * jnp.asarray(Determinant_part, dtype=dtype_wf_jnp)


def evaluate_jastrow(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float:
    r"""Evaluate the Jastrow factor :math:`\exp(J)` at the given coordinates.

    The method is for evaluate the Jastrow part of the wavefunction (Psi) at
    ``(r_up_carts, r_dn_carts)``. The returned value already includes the
    exponential, i.e., ``exp(J)``.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons with shape ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons with shape ``(n_dn, 3)``.

    Returns:
        Real Jastrow factor ``exp(J)``.
    """
    # r_*_carts forwarded unchanged (see ``evaluate_ln_wavefunction`` for rationale).
    Jastrow_part = compute_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    return jnp.exp(Jastrow_part)


def evaluate_determinant(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float:
    """Evaluate the determinant (Geminal) part of the wavefunction.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons with shape ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons with shape ``(n_dn, 3)``.

    Returns:
        Determinant value evaluated at the supplied coordinates.
    """
    # r_*_carts forwarded unchanged (see ``evaluate_ln_wavefunction`` for rationale).
    Determinant_part = compute_det_geminal_all_elements(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    return Determinant_part


@jit
def compute_kinetic_energy(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float | complex:
    """Compute kinetic energy using analytic gradients and Laplacians.

    The method is for computing kinetic energy of the given WF at
    ``(r_up_carts, r_dn_carts)`` and fully exploits the JAX library for the
    kinetic energy calculation. Inputs are converted to the kinetic zone dtype
    ``jax.Array`` for consistency with other compute utilities.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons with shape ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons with shape ``(n_dn, 3)``.

    Returns:
        Kinetic energy evaluated for the supplied configuration.
    """
    # r_*_carts forwarded unchanged (see ``evaluate_ln_wavefunction`` for rationale).
    dtype_jnp = get_dtype_jnp("wf_kinetic")

    # grad_J_up, grad_J_dn, sum_laplacian_J = 0.0, 0.0, 0.0
    # """
    grad_J_up, grad_J_dn, lap_J_up, lap_J_dn = compute_grads_and_laplacian_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )
    # """

    # grad_ln_D_up, grad_ln_D_dn, sum_laplacian_ln_D = 0.0, 0.0, 0.0
    # """
    grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn = compute_grads_and_laplacian_ln_Det(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )
    # """

    # Explicitly cast jastrow_grad_lap and det_grad_lap zone values to wf_kinetic
    # zone dtype before assembling T_L; do not rely on JAX implicit promotion.
    grad_J_up = jnp.asarray(grad_J_up, dtype=dtype_jnp)
    grad_J_dn = jnp.asarray(grad_J_dn, dtype=dtype_jnp)
    lap_J_up = jnp.asarray(lap_J_up, dtype=dtype_jnp)
    lap_J_dn = jnp.asarray(lap_J_dn, dtype=dtype_jnp)
    grad_ln_D_up = jnp.asarray(grad_ln_D_up, dtype=dtype_jnp)
    grad_ln_D_dn = jnp.asarray(grad_ln_D_dn, dtype=dtype_jnp)
    lap_ln_D_up = jnp.asarray(lap_ln_D_up, dtype=dtype_jnp)
    lap_ln_D_dn = jnp.asarray(lap_ln_D_dn, dtype=dtype_jnp)

    # compute kinetic energy
    L = (
        1.0
        / 2.0
        * (
            -(jnp.sum(lap_J_up) + jnp.sum(lap_J_dn) + jnp.sum(lap_ln_D_up) + jnp.sum(lap_ln_D_dn))
            - (
                jnp.sum((grad_J_up + grad_ln_D_up) * (grad_J_up + grad_ln_D_up))
                + jnp.sum((grad_J_dn + grad_ln_D_dn) * (grad_J_dn + grad_ln_D_dn))
            )
        )
    )

    return L


@jit
def _compute_kinetic_energy_auto(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float | complex:
    """The method is for computing kinetic energy of the given WF at (r_up_carts, r_dn_carts).

    Fully exploit the JAX library for the kinetic energy calculation.

    Args:
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (jax.Array): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (jax.Array): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        The kinetic energy with the given wavefunction (float | complex)
    """
    dtype_jnp = get_dtype_jnp("wf_kinetic")
    r_up = jnp.asarray(r_up_carts, dtype=dtype_jnp)
    r_dn = jnp.asarray(r_dn_carts, dtype=dtype_jnp)

    kinetic_energy_all_elements_up, kinetic_energy_all_elements_dn = _compute_kinetic_energy_all_elements_auto(
        wavefunction_data=wavefunction_data, r_up_carts=r_up, r_dn_carts=r_dn
    )

    K = jnp.sum(kinetic_energy_all_elements_up) + jnp.sum(kinetic_energy_all_elements_dn)

    return K


def _compute_kinetic_energy_debug(
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float | complex:
    """See compute_kinetic_energy_api."""
    kinetic_energy_all_elements_up, kinetic_energy_all_elements_dn = _compute_kinetic_energy_all_elements_debug(
        wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
    )

    return np.sum(kinetic_energy_all_elements_up) + np.sum(kinetic_energy_all_elements_dn)


def _compute_kinetic_energy_all_elements_debug(
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float | complex:
    """See compute_kinetic_energy_api.

    Uses 4th-order central finite differences for the Laplacian:
        f''(x) ≈ (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12h²)
    This allows a larger step size h while maintaining accuracy (O(h⁴) truncation error).
    """
    diff_h = 1.0e-3  # larger h is viable with 4th-order stencil

    Psi = evaluate_wavefunction(wavefunction_data, r_up_carts, r_dn_carts)

    def _eval_up(r_up):
        return evaluate_wavefunction(wavefunction_data, r_up, r_dn_carts)

    def _eval_dn(r_dn):
        return evaluate_wavefunction(wavefunction_data, r_up_carts, r_dn)

    def _fd4_second_deriv(eval_fn, r_carts, i, d, h):
        """4th-order central FD for d²f/dx²."""
        r_p1 = r_carts.copy()
        r_p2 = r_carts.copy()
        r_m1 = r_carts.copy()
        r_m2 = r_carts.copy()
        r_p1[i, d] += h
        r_p2[i, d] += 2 * h
        r_m1[i, d] -= h
        r_m2[i, d] -= 2 * h
        f_p1 = eval_fn(r_p1)
        f_p2 = eval_fn(r_p2)
        f_m1 = eval_fn(r_m1)
        f_m2 = eval_fn(r_m2)
        return (-f_p2 + 16 * f_p1 - 30 * Psi + 16 * f_m1 - f_m2) / (12 * h**2)

    n_up, d_up = r_up_carts.shape
    laplacian_Psi_up = np.zeros(n_up)
    for i in range(n_up):
        for d in range(d_up):
            laplacian_Psi_up[i] += _fd4_second_deriv(_eval_up, r_up_carts, i, d, diff_h)

    n_dn, d_dn = r_dn_carts.shape
    laplacian_Psi_dn = np.zeros(n_dn)
    for i in range(n_dn):
        for d in range(d_dn):
            laplacian_Psi_dn[i] += _fd4_second_deriv(_eval_dn, r_dn_carts, i, d, diff_h)

    kinetic_energy_all_elements_up = -1.0 / 2.0 * laplacian_Psi_up / Psi
    kinetic_energy_all_elements_dn = -1.0 / 2.0 * laplacian_Psi_dn / Psi

    return (kinetic_energy_all_elements_up, kinetic_energy_all_elements_dn)


@jit
def _compute_kinetic_energy_all_elements_auto(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> jax.Array:
    """See compute_kinetic_energy_api."""
    dtype_jnp = get_dtype_jnp("wf_kinetic")
    r_up = jnp.asarray(r_up_carts, dtype=dtype_jnp)
    r_dn = jnp.asarray(r_dn_carts, dtype=dtype_jnp)

    # compute gradients
    grad_J_up = grad(compute_Jastrow_part, argnums=1)(wavefunction_data.jastrow_data, r_up, r_dn)
    grad_J_dn = grad(compute_Jastrow_part, argnums=2)(wavefunction_data.jastrow_data, r_up, r_dn)
    grad_ln_Det_up = grad(compute_ln_det_geminal_all_elements, argnums=1)(wavefunction_data.geminal_data, r_up, r_dn)
    grad_ln_Det_dn = grad(compute_ln_det_geminal_all_elements, argnums=2)(wavefunction_data.geminal_data, r_up, r_dn)

    # Cast jastrow/det grad/lap zone values to wf_kinetic dtype before combining.
    grad_J_up = jnp.asarray(grad_J_up, dtype=dtype_jnp)
    grad_J_dn = jnp.asarray(grad_J_dn, dtype=dtype_jnp)
    grad_ln_Det_up = jnp.asarray(grad_ln_Det_up, dtype=dtype_jnp)
    grad_ln_Det_dn = jnp.asarray(grad_ln_Det_dn, dtype=dtype_jnp)

    grad_ln_Psi_up = grad_J_up + grad_ln_Det_up
    grad_ln_Psi_dn = grad_J_dn + grad_ln_Det_dn

    # compute laplacians
    hessian_J_up = hessian(compute_Jastrow_part, argnums=1)(wavefunction_data.jastrow_data, r_up, r_dn)
    laplacian_J_up = jnp.einsum("ijij->i", hessian_J_up)
    hessian_J_dn = hessian(compute_Jastrow_part, argnums=2)(wavefunction_data.jastrow_data, r_up, r_dn)
    laplacian_J_dn = jnp.einsum("ijij->i", hessian_J_dn)

    hessian_ln_Det_up = hessian(compute_ln_det_geminal_all_elements, argnums=1)(wavefunction_data.geminal_data, r_up, r_dn)
    laplacian_ln_Det_up = jnp.einsum("ijij->i", hessian_ln_Det_up)
    hessian_ln_Det_dn = hessian(compute_ln_det_geminal_all_elements, argnums=2)(wavefunction_data.geminal_data, r_up, r_dn)
    laplacian_ln_Det_dn = jnp.einsum("ijij->i", hessian_ln_Det_dn)

    laplacian_J_up = jnp.asarray(laplacian_J_up, dtype=dtype_jnp)
    laplacian_J_dn = jnp.asarray(laplacian_J_dn, dtype=dtype_jnp)
    laplacian_ln_Det_up = jnp.asarray(laplacian_ln_Det_up, dtype=dtype_jnp)
    laplacian_ln_Det_dn = jnp.asarray(laplacian_ln_Det_dn, dtype=dtype_jnp)

    laplacian_Psi_up = laplacian_J_up + laplacian_ln_Det_up
    laplacian_Psi_dn = laplacian_J_dn + laplacian_ln_Det_dn

    kinetic_energy_all_elements_up = -1.0 / 2.0 * (laplacian_Psi_up + jnp.sum(grad_ln_Psi_up**2, axis=1))
    kinetic_energy_all_elements_dn = -1.0 / 2.0 * (laplacian_Psi_dn + jnp.sum(grad_ln_Psi_dn**2, axis=1))

    return (kinetic_energy_all_elements_up, kinetic_energy_all_elements_dn)


@jit
def compute_kinetic_energy_all_elements(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> jax.Array:
    """Analytic-derivative kinetic energy per electron (matches auto output shape).

    Returns the per-electron kinetic energy using analytic gradients/Laplacians of
    both Jastrow and determinant parts. Shapes align with
    ``_compute_kinetic_energy_all_elements_auto``.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons with shape ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons with shape ``(n_dn, 3)``.

    Returns:
        Tuple of two ``jax.Array`` objects containing per-electron kinetic energies
        for spin-up and spin-down electrons, respectively.
    """
    # r_*_carts forwarded unchanged (see ``evaluate_ln_wavefunction`` for rationale).
    dtype_jnp = get_dtype_jnp("wf_kinetic")

    # --- Jastrow contributions (per-electron Laplacians) ---
    grad_J_up, grad_J_dn, lap_J_up, lap_J_dn = compute_grads_and_laplacian_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # --- Determinant contributions (per-electron Laplacians) ---
    grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn = compute_grads_and_laplacian_ln_Det(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # Cast jastrow_grad_lap / det_grad_lap zone values to wf_kinetic dtype.
    grad_J_up = jnp.asarray(grad_J_up, dtype=dtype_jnp)
    grad_J_dn = jnp.asarray(grad_J_dn, dtype=dtype_jnp)
    lap_J_up = jnp.asarray(lap_J_up, dtype=dtype_jnp)
    lap_J_dn = jnp.asarray(lap_J_dn, dtype=dtype_jnp)
    grad_ln_D_up = jnp.asarray(grad_ln_D_up, dtype=dtype_jnp)
    grad_ln_D_dn = jnp.asarray(grad_ln_D_dn, dtype=dtype_jnp)
    lap_ln_D_up = jnp.asarray(lap_ln_D_up, dtype=dtype_jnp)
    lap_ln_D_dn = jnp.asarray(lap_ln_D_dn, dtype=dtype_jnp)

    # --- Assemble kinetic energy per electron ---
    grad_ln_Psi_up = grad_J_up + grad_ln_D_up
    grad_ln_Psi_dn = grad_J_dn + grad_ln_D_dn

    lap_ln_Psi_up = lap_J_up + lap_ln_D_up
    lap_ln_Psi_dn = lap_J_dn + lap_ln_D_dn

    kinetic_energy_all_elements_up = -0.5 * (lap_ln_Psi_up + jnp.sum(grad_ln_Psi_up**2, axis=1))
    kinetic_energy_all_elements_dn = -0.5 * (lap_ln_Psi_dn + jnp.sum(grad_ln_Psi_dn**2, axis=1))

    return (kinetic_energy_all_elements_up, kinetic_energy_all_elements_dn)


@jit
def compute_kinetic_energy_all_elements_fast_update(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    geminal_inverse: jax.Array,
) -> jax.Array:
    """Kinetic energy per electron using a precomputed geminal inverse.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons ``(n_dn, 3)``.
        geminal_inverse: Pre-computed inverse geminal matrix ``(N_up, N_up)``
            valid at the supplied ``(r_up_carts, r_dn_carts)``.

    Returns:
        Tuple of per-electron kinetic energies (up, down).

    Warning:
        ``geminal_inverse`` **must** equal ``G(r_up_carts, r_dn_carts)^{-1}``
        exactly at the supplied electron positions.  Correctness is only
        guaranteed when the inverse is maintained via **single-electron
        (rank-1) Sherman-Morrison updates** starting from a freshly
        initialized LU inverse — the pattern used in the MCMC loop.
        Passing an inverse from a different configuration silently produces
        incorrect kinetic energy.
    """
    if geminal_inverse is None:
        raise ValueError("geminal_inverse must be provided for fast update")

    # r_*_carts forwarded unchanged (see ``evaluate_ln_wavefunction`` for rationale).
    dtype_jnp = get_dtype_jnp("wf_kinetic")

    grad_J_up, grad_J_dn, lap_J_up, lap_J_dn = compute_grads_and_laplacian_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn = compute_grads_and_laplacian_ln_Det_fast(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        geminal_inverse=geminal_inverse,
    )

    # Cast jastrow_grad_lap / det_grad_lap zone values to wf_kinetic dtype.
    grad_J_up = jnp.asarray(grad_J_up, dtype=dtype_jnp)
    grad_J_dn = jnp.asarray(grad_J_dn, dtype=dtype_jnp)
    lap_J_up = jnp.asarray(lap_J_up, dtype=dtype_jnp)
    lap_J_dn = jnp.asarray(lap_J_dn, dtype=dtype_jnp)
    grad_ln_D_up = jnp.asarray(grad_ln_D_up, dtype=dtype_jnp)
    grad_ln_D_dn = jnp.asarray(grad_ln_D_dn, dtype=dtype_jnp)
    lap_ln_D_up = jnp.asarray(lap_ln_D_up, dtype=dtype_jnp)
    lap_ln_D_dn = jnp.asarray(lap_ln_D_dn, dtype=dtype_jnp)

    grad_ln_Psi_up = grad_J_up + grad_ln_D_up
    grad_ln_Psi_dn = grad_J_dn + grad_ln_D_dn

    lap_ln_Psi_up = lap_J_up + lap_ln_D_up
    lap_ln_Psi_dn = lap_J_dn + lap_ln_D_dn

    kinetic_energy_all_elements_up = -0.5 * (lap_ln_Psi_up + jnp.sum(grad_ln_Psi_up**2, axis=1))
    kinetic_energy_all_elements_dn = -0.5 * (lap_ln_Psi_dn + jnp.sum(grad_ln_Psi_dn**2, axis=1))

    return (kinetic_energy_all_elements_up, kinetic_energy_all_elements_dn)


def _compute_kinetic_energy_all_elements_fast_update_debug(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> jax.Array:
    """Debug helper that builds geminal inverse then calls the fast update path."""
    return compute_kinetic_energy_all_elements(
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )


# ---------------------------------------------------------------------------
# Per-electron kinetic-energy streaming state (used by GFMC projection)
# ---------------------------------------------------------------------------
#
# Maintains enough auxiliary information to advance the per-electron kinetic
# energies after a single-electron move without recomputing them from
# scratch. PR1 (devel-speedup-lrdmc-incremental) enables this only for the
# J3 part — J1, J2 and the determinant gradients/Laplacians are still
# recomputed fresh inside ``_advance_*``. Subsequent PRs will replace those
# fresh recomputes with rank-1 updates while keeping the public per-electron
# fields (``grad_J_up`` etc.) shape-stable.
#
# The state is freshly built at every branching boundary by
# ``_init_kinetic_energy_all_elements_streaming_state`` (lifetime matches
# the Sherman-Morrison ``A_old_inv``).


@struct.dataclass
class Kinetic_streaming_state:
    """Streaming state for per-electron kinetic-energy evaluation.

    Fields evaluated at the current ``(r_up_carts, r_dn_carts)``:

    - ``j3_state``: J3 auxiliary tables (None if no J3 component is active).
    - ``det_state``: det auxiliary tables (always populated in PR2+; the
      determinant per-electron grad/lap fields below mirror its outputs).
    - ``grad_J_up`` / ``grad_J_dn``: total Jastrow per-electron gradient.
    - ``lap_J_up`` / ``lap_J_dn``: total Jastrow per-electron Laplacian.
    - ``grad_ln_D_up`` / ``grad_ln_D_dn``: per-electron ``∇ln|Det|`` from the
      geminal at the current ``A_old_inv``.
    - ``lap_ln_D_up`` / ``lap_ln_D_dn``: per-electron ``∇²ln|Det|``.
    """

    j1_state: Jastrow_one_body_streaming_state | None = struct.field(pytree_node=True, default=None)
    j2_state: Jastrow_two_body_streaming_state | None = struct.field(pytree_node=True, default=None)
    j3_state: Jastrow_three_body_streaming_state | None = struct.field(pytree_node=True, default=None)
    det_state: Det_streaming_state | None = struct.field(pytree_node=True, default=None)
    grad_J_up: jax.Array = struct.field(pytree_node=True, default=None)
    grad_J_dn: jax.Array = struct.field(pytree_node=True, default=None)
    lap_J_up: jax.Array = struct.field(pytree_node=True, default=None)
    lap_J_dn: jax.Array = struct.field(pytree_node=True, default=None)
    grad_ln_D_up: jax.Array = struct.field(pytree_node=True, default=None)
    grad_ln_D_dn: jax.Array = struct.field(pytree_node=True, default=None)
    lap_ln_D_up: jax.Array = struct.field(pytree_node=True, default=None)
    lap_ln_D_dn: jax.Array = struct.field(pytree_node=True, default=None)


def _kinetic_energy_from_grads_laps(
    grad_J_up,
    grad_J_dn,
    lap_J_up,
    lap_J_dn,
    grad_ln_D_up,
    grad_ln_D_dn,
    lap_ln_D_up,
    lap_ln_D_dn,
):
    """Common assembly: ``-(1/2) * (∇²ln Ψ + ||∇ln Ψ||²)`` per electron."""
    dtype_jnp = get_dtype_jnp("wf_kinetic")
    grad_J_up = jnp.asarray(grad_J_up, dtype=dtype_jnp)
    grad_J_dn = jnp.asarray(grad_J_dn, dtype=dtype_jnp)
    lap_J_up = jnp.asarray(lap_J_up, dtype=dtype_jnp)
    lap_J_dn = jnp.asarray(lap_J_dn, dtype=dtype_jnp)
    grad_ln_D_up = jnp.asarray(grad_ln_D_up, dtype=dtype_jnp)
    grad_ln_D_dn = jnp.asarray(grad_ln_D_dn, dtype=dtype_jnp)
    lap_ln_D_up = jnp.asarray(lap_ln_D_up, dtype=dtype_jnp)
    lap_ln_D_dn = jnp.asarray(lap_ln_D_dn, dtype=dtype_jnp)

    grad_ln_Psi_up = grad_J_up + grad_ln_D_up
    grad_ln_Psi_dn = grad_J_dn + grad_ln_D_dn
    lap_ln_Psi_up = lap_J_up + lap_ln_D_up
    lap_ln_Psi_dn = lap_J_dn + lap_ln_D_dn
    ke_up = -0.5 * (lap_ln_Psi_up + jnp.sum(grad_ln_Psi_up**2, axis=1))
    ke_dn = -0.5 * (lap_ln_Psi_dn + jnp.sum(grad_ln_Psi_dn**2, axis=1))
    return ke_up, ke_dn


def _init_kinetic_energy_all_elements_streaming_state(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    geminal_inverse: jax.Array,
) -> Kinetic_streaming_state:
    """Build a fresh streaming state at the supplied ``(r_up, r_dn)``.

    PR1+PR2+PR3 scope: J1, J2, J3, and det sub-states are all incrementally
    maintained. NN three-body falls back to ``compute_grads_and_laplacian_Jastrow_part``
    (the streaming dispatch in ``jqmc_gfmc.py`` already excludes the NN case).

    Note: ``geminal_inverse`` must be the inverse of ``G(r_up, r_dn)`` (the
    same invariant as :func:`compute_kinetic_energy_all_elements_fast_update`).
    """
    # Per-electron Jastrow grad/lap (sum of J1/J2/J3/NN parts) — used as the
    # initial total. Sub-states below are populated for the streaming path.
    grad_J_up, grad_J_dn, lap_J_up, lap_J_dn = compute_grads_and_laplacian_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # Cast totals to the jastrow_grad_lap zone so init and advance store
    # ``grad_J_*`` / ``lap_J_*`` in the same dtype (Principle 3b — required
    # for fori_loop carry-shape stability under mixed precision, where
    # ``advance`` reassembles the totals from streaming sub-states that
    # live in the jastrow_grad_lap zone).
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    grad_J_up = jnp.asarray(grad_J_up, dtype=dtype_jnp)
    grad_J_dn = jnp.asarray(grad_J_dn, dtype=dtype_jnp)
    lap_J_up = jnp.asarray(lap_J_up, dtype=dtype_jnp)
    lap_J_dn = jnp.asarray(lap_J_dn, dtype=dtype_jnp)

    # Determinant streaming state — drives grad_ln_D_*/lap_ln_D_* fields.
    det_state = _init_grads_laplacian_ln_Det_streaming_state(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        geminal_inverse=geminal_inverse,
    )

    jastrow_data = wavefunction_data.jastrow_data
    j1_data = jastrow_data.jastrow_one_body_data
    j2_data = jastrow_data.jastrow_two_body_data
    j3_data = jastrow_data.jastrow_three_body_data
    j1_state = (
        _init_grads_laplacian_Jastrow_one_body_streaming_state(j1_data, r_up_carts, r_dn_carts) if j1_data is not None else None
    )
    j2_state = (
        _init_grads_laplacian_Jastrow_two_body_streaming_state(j2_data, r_up_carts, r_dn_carts) if j2_data is not None else None
    )
    j3_state = (
        _init_grads_laplacian_Jastrow_three_body_streaming_state(j3_data, r_up_carts, r_dn_carts)
        if j3_data is not None
        else None
    )

    return Kinetic_streaming_state(
        j1_state=j1_state,
        j2_state=j2_state,
        j3_state=j3_state,
        det_state=det_state,
        grad_J_up=grad_J_up,
        grad_J_dn=grad_J_dn,
        lap_J_up=lap_J_up,
        lap_J_dn=lap_J_dn,
        grad_ln_D_up=det_state.grad_ln_D_up,
        grad_ln_D_dn=det_state.grad_ln_D_dn,
        lap_ln_D_up=det_state.lap_ln_D_up,
        lap_ln_D_dn=det_state.lap_ln_D_dn,
    )


def _kinetic_energy_from_streaming_state(state: Kinetic_streaming_state):
    """Per-electron kinetic energies extracted from a streaming state."""
    return _kinetic_energy_from_grads_laps(
        state.grad_J_up,
        state.grad_J_dn,
        state.lap_J_up,
        state.lap_J_dn,
        state.grad_ln_D_up,
        state.grad_ln_D_dn,
        state.lap_ln_D_up,
        state.lap_ln_D_dn,
    )


def _advance_kinetic_energy_all_elements_streaming_state(
    wavefunction_data: Wavefunction_data,
    state: Kinetic_streaming_state,
    moved_spin_is_up: jax.Array,
    moved_index: jax.Array,
    r_up_carts_new: jax.Array,
    r_dn_carts_new: jax.Array,
    A_new_inv: jax.Array,
) -> Kinetic_streaming_state:
    """Advance the streaming state after a single-electron move.

    PR1+PR2+PR3 scope: J1, J2, J3, and det sub-states are all updated
    incrementally. NN three-body falls back to a fresh
    ``compute_grads_and_laplacian_Jastrow_part`` call (defensive — the
    streaming dispatch in ``jqmc_gfmc.py`` excludes the NN case so this
    branch is unreachable in production).

    The returned state is consistent with ``(r_up_carts_new, r_dn_carts_new,
    A_new_inv)``; downstream consumers can read kinetic energies via
    :func:`_kinetic_energy_from_streaming_state`.
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    jastrow_data = wavefunction_data.jastrow_data

    # --- J1: incremental advance via streaming state ---------------------
    j1_data = jastrow_data.jastrow_one_body_data
    if j1_data is not None and state.j1_state is not None:
        new_j1_state = _advance_grads_laplacian_Jastrow_one_body_streaming_state(
            j1_data,
            state.j1_state,
            moved_spin_is_up,
            moved_index,
            r_up_carts_new,
            r_dn_carts_new,
        )
        grad_J1_up = new_j1_state.grad_J1_up
        grad_J1_dn = new_j1_state.grad_J1_dn
        lap_J1_up = new_j1_state.lap_J1_up
        lap_J1_dn = new_j1_state.lap_J1_dn
    else:
        new_j1_state = None
        grad_J1_up = jnp.zeros_like(state.grad_J_up)
        grad_J1_dn = jnp.zeros_like(state.grad_J_dn)
        lap_J1_up = jnp.zeros_like(state.lap_J_up)
        lap_J1_dn = jnp.zeros_like(state.lap_J_dn)

    # --- J2: incremental advance via streaming state ---------------------
    j2_data = jastrow_data.jastrow_two_body_data
    if j2_data is not None and state.j2_state is not None:
        new_j2_state = _advance_grads_laplacian_Jastrow_two_body_streaming_state(
            j2_data,
            state.j2_state,
            moved_spin_is_up,
            moved_index,
            r_up_carts_new,
            r_dn_carts_new,
        )
        grad_J2_up = new_j2_state.grad_J2_up
        grad_J2_dn = new_j2_state.grad_J2_dn
        lap_J2_up = new_j2_state.lap_J2_up
        lap_J2_dn = new_j2_state.lap_J2_dn
    else:
        new_j2_state = None
        grad_J2_up = jnp.zeros_like(state.grad_J_up)
        grad_J2_dn = jnp.zeros_like(state.grad_J_dn)
        lap_J2_up = jnp.zeros_like(state.lap_J_up)
        lap_J2_dn = jnp.zeros_like(state.lap_J_dn)

    # --- J3: incremental advance via streaming state ---------------------
    j3_data = jastrow_data.jastrow_three_body_data
    if j3_data is not None and state.j3_state is not None:
        new_j3_state = _advance_grads_laplacian_Jastrow_three_body_streaming_state(
            j3_data,
            state.j3_state,
            moved_spin_is_up,
            moved_index,
            r_up_carts_new,
            r_dn_carts_new,
        )
        grad_J3_up = new_j3_state.grad_J3_up
        grad_J3_dn = new_j3_state.grad_J3_dn
        lap_J3_up = new_j3_state.lap_J3_up
        lap_J3_dn = new_j3_state.lap_J3_dn
    else:
        new_j3_state = None
        grad_J3_up = jnp.zeros_like(state.grad_J_up)
        grad_J3_dn = jnp.zeros_like(state.grad_J_dn)
        lap_J3_up = jnp.zeros_like(state.lap_J_up)
        lap_J3_dn = jnp.zeros_like(state.lap_J_dn)

    # Reassemble Jastrow totals from the streamed sub-state contributions.
    grad_J_up = grad_J1_up + grad_J2_up + grad_J3_up
    grad_J_dn = grad_J1_dn + grad_J2_dn + grad_J3_dn
    lap_J_up = lap_J1_up + lap_J2_up + lap_J3_up
    lap_J_dn = lap_J1_dn + lap_J2_dn + lap_J3_dn

    # NN three-body (autodiff path) — defensive fallback. The streaming
    # dispatch in ``jqmc_gfmc.py`` already routes NN-on cases to the legacy
    # body, so this branch is unreachable in production.
    if jastrow_data.jastrow_nn_data is not None:
        grad_J_up_full, grad_J_dn_full, lap_J_up_full, lap_J_dn_full = compute_grads_and_laplacian_Jastrow_part(
            jastrow_data=jastrow_data,
            r_up_carts=r_up_carts_new,
            r_dn_carts=r_dn_carts_new,
        )
        grad_J_up = grad_J_up_full
        grad_J_dn = grad_J_dn_full
        lap_J_up = lap_J_up_full
        lap_J_dn = lap_J_dn_full

    # --- determinant: incremental advance via streaming state ------------
    new_det_state = _advance_grads_laplacian_ln_Det_streaming_state(
        geminal_data=wavefunction_data.geminal_data,
        state=state.det_state,
        moved_spin_is_up=moved_spin_is_up,
        moved_index=moved_index,
        r_up_carts_new=r_up_carts_new,
        r_dn_carts_new=r_dn_carts_new,
        A_new_inv=A_new_inv,
    )

    # Cast totals to jastrow_grad_lap dtype to match init's storage zone.
    grad_J_up = jnp.asarray(grad_J_up, dtype=dtype_jnp)
    grad_J_dn = jnp.asarray(grad_J_dn, dtype=dtype_jnp)
    lap_J_up = jnp.asarray(lap_J_up, dtype=dtype_jnp)
    lap_J_dn = jnp.asarray(lap_J_dn, dtype=dtype_jnp)

    return state.replace(
        j1_state=new_j1_state,
        j2_state=new_j2_state,
        j3_state=new_j3_state,
        det_state=new_det_state,
        grad_J_up=grad_J_up,
        grad_J_dn=grad_J_dn,
        lap_J_up=lap_J_up,
        lap_J_dn=lap_J_dn,
        grad_ln_D_up=new_det_state.grad_ln_D_up,
        grad_ln_D_dn=new_det_state.grad_ln_D_dn,
        lap_ln_D_up=new_det_state.lap_ln_D_up,
        lap_ln_D_dn=new_det_state.lap_ln_D_dn,
    )


def _compute_discretized_kinetic_energy_debug(
    alat: float, wavefunction_data: Wavefunction_data, r_up_carts: npt.NDArray, r_dn_carts: npt.NDArray
) -> list[tuple[npt.NDArray, npt.NDArray]]:
    r"""_summary.

    Args:
        alat (float): Hamiltonian discretization (bohr), which will be replaced with LRDMC_data.
        wavefunction_data (Wavefunction_data): an instance of Qavefunction_data, which will be replaced with LRDMC_data.
        r_carts_up (npt.NDArray): up electron position (N_e,3).
        r_carts_dn (npt.NDArray): down electron position (N_e,3).

    Returns:
        list[tuple[npt.NDArray, npt.NDArray]], list[npt.NDArray]:
            return mesh for the LRDMC kinetic part, a list containing tuples containing (r_carts_up, r_carts_dn),
            and a list containing values of the \Psi(x')/\Psi(x) corresponding to the grid.
    """
    mesh_kinetic_part = []

    # up electron
    for r_up_i in range(len(r_up_carts)):
        # x, plus
        r_up_carts_p = r_up_carts.copy()
        r_up_carts_p[r_up_i, 0] += alat
        mesh_kinetic_part.append((r_up_carts_p, r_dn_carts))
        # x, minus
        r_up_carts_p = r_up_carts.copy()
        r_up_carts_p[r_up_i, 0] -= alat
        mesh_kinetic_part.append((r_up_carts_p, r_dn_carts))
        # y, plus
        r_up_carts_p = r_up_carts.copy()
        r_up_carts_p[r_up_i, 1] += alat
        mesh_kinetic_part.append((r_up_carts_p, r_dn_carts))
        # y, minus
        r_up_carts_p = r_up_carts.copy()
        r_up_carts_p[r_up_i, 1] -= alat
        mesh_kinetic_part.append((r_up_carts_p, r_dn_carts))
        # z, plus
        r_up_carts_p = r_up_carts.copy()
        r_up_carts_p[r_up_i, 2] += alat
        mesh_kinetic_part.append((r_up_carts_p, r_dn_carts))
        # z, minus
        r_up_carts_p = r_up_carts.copy()
        r_up_carts_p[r_up_i, 2] -= alat
        mesh_kinetic_part.append((r_up_carts_p, r_dn_carts))

    # dn electron
    for r_dn_i in range(len(r_dn_carts)):
        # x, plus
        r_dn_carts_p = r_dn_carts.copy()
        r_dn_carts_p[r_dn_i, 0] += alat
        mesh_kinetic_part.append((r_up_carts, r_dn_carts_p))
        # x, minus
        r_dn_carts_p = r_dn_carts.copy()
        r_dn_carts_p[r_dn_i, 0] -= alat
        mesh_kinetic_part.append((r_up_carts, r_dn_carts_p))
        # y, plus
        r_dn_carts_p = r_dn_carts.copy()
        r_dn_carts_p[r_dn_i, 1] += alat
        mesh_kinetic_part.append((r_up_carts, r_dn_carts_p))
        # y, minus
        r_dn_carts_p = r_dn_carts.copy()
        r_dn_carts_p[r_dn_i, 1] -= alat
        mesh_kinetic_part.append((r_up_carts, r_dn_carts_p))
        # z, plus
        r_dn_carts_p = r_dn_carts.copy()
        r_dn_carts_p[r_dn_i, 2] += alat
        mesh_kinetic_part.append((r_up_carts, r_dn_carts_p))
        # z, minus
        r_dn_carts_p = r_dn_carts.copy()
        r_dn_carts_p[r_dn_i, 2] -= alat
        mesh_kinetic_part.append((r_up_carts, r_dn_carts_p))

    elements_kinetic_part = [
        float(
            -1.0
            / (2.0 * alat**2)
            * evaluate_wavefunction(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_, r_dn_carts=r_dn_carts_)
            / evaluate_wavefunction(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
        )
        for r_up_carts_, r_dn_carts_ in mesh_kinetic_part
    ]

    r_up_carts_combined = np.array([up for up, _ in mesh_kinetic_part])
    r_dn_carts_combined = np.array([dn for _, dn in mesh_kinetic_part])

    return r_up_carts_combined, r_dn_carts_combined, elements_kinetic_part


@jit
def compute_discretized_kinetic_energy(
    alat: float, wavefunction_data, r_up_carts: jax.Array, r_dn_carts: jax.Array, RT: jax.Array
) -> tuple[list[tuple[npt.NDArray, npt.NDArray]], list[npt.NDArray], jax.Array]:
    r"""Compute discretized kinetic mesh points and energies for a given lattice spacing ``alat``.

    Function for computing discretized kinetic grid points and their energies with a
    given lattice space (alat). This keeps the original semantics used by the LRDMC
    path: ratios are computed as ``exp(J_xp - J_x) * det_xp / det_x``. Inputs are
    coerced to the kinetic zone dtype ``jax.Array`` before evaluation.

    Args:
        alat: Hamiltonian discretization (bohr), which will be replaced with ``LRDMC_data``.
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Up-electron positions with shape ``(n_up, 3)``.
        r_dn_carts: Down-electron positions with shape ``(n_dn, 3)``.
        RT: Rotation matrix (:math:`R^T`) with shape ``(3, 3)``.

    Returns:
        A tuple ``(r_up_carts_combined, r_dn_carts_combined, elements_kinetic_part)`` where the
        combined coordinate arrays have shapes ``(n_grid, n_up, 3)`` and ``(n_grid, n_dn, 3)``
        and ``elements_kinetic_part`` contains the kinetic prefactor-scaled ratios.
    """
    dtype_jnp = get_dtype_jnp("wf_kinetic")
    r_up = jnp.asarray(r_up_carts, dtype=dtype_jnp)
    r_dn = jnp.asarray(r_dn_carts, dtype=dtype_jnp)
    rt = jnp.asarray(RT, dtype=dtype_jnp)
    # Define the shifts to apply (+/- alat in each coordinate direction)
    shifts = alat * jnp.array(
        [
            [1, 0, 0],  # x+
            [-1, 0, 0],  # x-
            [0, 1, 0],  # y+
            [0, -1, 0],  # y-
            [0, 0, 1],  # z+
            [0, 0, -1],  # z-
        ]
    )  # Shape: (6, 3)

    shifts = shifts @ rt  # Shape: (6, 3)

    # num shift
    num_shifts = shifts.shape[0]

    # Process up-spin electrons
    num_up_electrons = r_up.shape[0]
    num_up_configs = num_up_electrons * num_shifts

    # Create base positions repeated for each configuration
    base_positions_up = jnp.repeat(r_up[None, :, :], num_up_configs, axis=0)  # Shape: (num_up_configs, N_up, 3)

    # Initialize shifts_to_apply_up
    shifts_to_apply_up = jnp.zeros_like(base_positions_up)

    # Create indices for configurations
    config_indices_up = jnp.arange(num_up_configs)
    electron_indices_up = jnp.repeat(jnp.arange(num_up_electrons), num_shifts)
    shift_indices_up = jnp.tile(jnp.arange(num_shifts), num_up_electrons)

    # Apply shifts to the appropriate electron in each configuration
    shifts_to_apply_up = shifts_to_apply_up.at[config_indices_up, electron_indices_up, :].set(shifts[shift_indices_up])

    # Apply shifts to base positions
    r_up_carts_shifted = base_positions_up + shifts_to_apply_up  # Shape: (num_up_configs, N_up, 3)

    # Repeat down-spin electrons for up-spin configurations
    r_dn_carts_repeated_up = jnp.repeat(r_dn[None, :, :], num_up_configs, axis=0)  # Shape: (num_up_configs, N_dn, 3)

    # Process down-spin electrons
    num_dn_electrons = r_dn.shape[0]
    num_dn_configs = num_dn_electrons * num_shifts

    base_positions_dn = jnp.repeat(r_dn[None, :, :], num_dn_configs, axis=0)  # Shape: (num_dn_configs, N_dn, 3)
    shifts_to_apply_dn = jnp.zeros_like(base_positions_dn)

    config_indices_dn = jnp.arange(num_dn_configs)
    electron_indices_dn = jnp.repeat(jnp.arange(num_dn_electrons), num_shifts)
    shift_indices_dn = jnp.tile(jnp.arange(num_shifts), num_dn_electrons)

    # Apply shifts to the appropriate electron in each configuration
    shifts_to_apply_dn = shifts_to_apply_dn.at[config_indices_dn, electron_indices_dn, :].set(shifts[shift_indices_dn])

    r_dn_carts_shifted = base_positions_dn + shifts_to_apply_dn  # Shape: (num_dn_configs, N_dn, 3)

    # Repeat up-spin electrons for down-spin configurations
    r_up_carts_repeated_dn = jnp.repeat(r_up[None, :, :], num_dn_configs, axis=0)  # Shape: (num_dn_configs, N_up, 3)

    # Combine configurations
    r_up_carts_combined = jnp.concatenate([r_up_carts_shifted, r_up_carts_repeated_dn], axis=0)  # Shape: (N_configs, N_up, 3)
    r_dn_carts_combined = jnp.concatenate([r_dn_carts_repeated_up, r_dn_carts_shifted], axis=0)  # Shape: (N_configs, N_dn, 3)

    # Evaluate the wavefunction at the original positions
    jastrow_x = compute_Jastrow_part(wavefunction_data.jastrow_data, r_up, r_dn)
    # Evaluate the wavefunction at the shifted positions using vectorization
    jastrow_xp = vmap(compute_Jastrow_part, in_axes=(None, 0, 0))(
        wavefunction_data.jastrow_data, r_up_carts_combined, r_dn_carts_combined
    )
    # Evaluate the wavefunction at the original positions
    det_x = compute_det_geminal_all_elements(wavefunction_data.geminal_data, r_up, r_dn)
    # Evaluate the wavefunction at the shifted positions using vectorization
    det_xp = vmap(compute_det_geminal_all_elements, in_axes=(None, 0, 0))(
        wavefunction_data.geminal_data, r_up_carts_combined, r_dn_carts_combined
    )
    # Explicitly cast both jastrow (jastrow_eval zone, possibly fp32) and det
    # values to the wf_ratio zone dtype so that exp() and the wf_ratio arithmetic
    # do not rely on JAX implicit fp32 x fp64 -> fp64 promotion.
    dtype_wf_ratio_jnp = get_dtype_jnp("wf_ratio")
    jastrow_x = jnp.asarray(jastrow_x, dtype=dtype_wf_ratio_jnp)
    jastrow_xp = jnp.asarray(jastrow_xp, dtype=dtype_wf_ratio_jnp)
    det_x = jnp.asarray(det_x, dtype=dtype_wf_ratio_jnp)
    det_xp = jnp.asarray(det_xp, dtype=dtype_wf_ratio_jnp)
    wf_ratio = jnp.exp(jastrow_xp - jastrow_x) * det_xp / det_x

    # Compute the kinetic part elements
    elements_kinetic_part = -1.0 / (2.0 * alat**2) * wf_ratio

    # Return the combined configurations and the kinetic elements
    return r_up_carts_combined, r_dn_carts_combined, elements_kinetic_part


@jit
def compute_discretized_kinetic_energy_fast_update(
    alat: float,
    wavefunction_data: Wavefunction_data,
    A_old_inv: jnp.ndarray,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    RT: jax.Array,
    j3_state: "Jastrow_three_body_streaming_state | None" = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Fast-update version of discretized kinetic mesh and ratios.

    Function for computing discretized kinetic grid points and their energies with
    a given lattice space (alat). Uses precomputed ``A_old_inv`` to evaluate
    determinant ratios efficiently. Inputs are converted to the kinetic zone dtype
    ``jax.Array`` before use.

    Args:
        alat: Hamiltonian discretization (bohr), which will be replaced with ``LRDMC_data``.
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        A_old_inv: Inverse of the geminal matrix evaluated at ``(r_up_carts, r_dn_carts)``.
        r_up_carts: Up-electron positions with shape ``(n_up, 3)``.
        r_dn_carts: Down-electron positions with shape ``(n_dn, 3)``.
        RT: Rotation matrix (:math:`R^T`) with shape ``(3, 3)``.
        j3_state: Optional cached J3 streaming auxiliaries consistent with
            ``(r_up_carts, r_dn_carts)``. Forwarded to the Jastrow ratio kernel
            so it can skip the per-call ``aos_*_old``/``W``/``U``/cross_vec
            recomputation. Use the value carried in the projection's
            ``Kinetic_streaming_state.j3_state``; pass ``None`` (default) for
            the original 1-shot path used by observation/MCMC code.

    Returns:
        Tuple ``(r_up_carts_combined, r_dn_carts_combined, elements_kinetic_part)`` with combined
        coordinate arrays of shapes ``(n_grid, n_up, 3)`` and ``(n_grid, n_dn, 3)``, and kinetic
        prefactor-scaled ratios ``elements_kinetic_part``.
    """
    dtype_jnp = get_dtype_jnp("wf_kinetic")
    r_up = jnp.asarray(r_up_carts, dtype=dtype_jnp)
    r_dn = jnp.asarray(r_dn_carts, dtype=dtype_jnp)
    rt = jnp.asarray(RT, dtype=dtype_jnp)
    # Define the shifts to apply (+/- alat in each coordinate direction)
    shifts = alat * jnp.array(
        [
            [1, 0, 0],  # x+
            [-1, 0, 0],  # x-
            [0, 1, 0],  # y+
            [0, -1, 0],  # y-
            [0, 0, 1],  # z+
            [0, 0, -1],  # z-
        ]
    )  # Shape: (6, 3)

    shifts = shifts @ rt  # Shape: (6, 3)

    # num shift
    num_shifts = shifts.shape[0]

    # Process up-spin electrons
    num_up_electrons = r_up.shape[0]
    num_up_configs = num_up_electrons * num_shifts

    # Build shifted configurations via outer-product broadcast (no scatter, GEMM-compatible).
    # delta_up[i, s, j, :] = shifts[s] if j==i else 0
    eye_up = jnp.eye(num_up_electrons)  # (N_up, N_up)
    delta_up = eye_up[:, None, :, None] * shifts[None, :, None, :]  # (N_up, 6, N_up, 3)
    r_up_carts_shifted = (r_up[None, None, :, :] + delta_up).reshape(
        num_up_configs, num_up_electrons, 3
    )  # (num_up_configs, N_up, 3)

    # Broadcast unchanged spin block (avoids repeat allocation)
    r_dn_carts_repeated_up = jnp.broadcast_to(r_dn[None, :, :], (num_up_configs, r_dn.shape[0], 3))  # (num_up_configs, N_dn, 3)

    # Process down-spin electrons
    num_dn_electrons = r_dn.shape[0]
    num_dn_configs = num_dn_electrons * num_shifts

    eye_dn = jnp.eye(num_dn_electrons)  # (N_dn, N_dn)
    delta_dn = eye_dn[:, None, :, None] * shifts[None, :, None, :]  # (N_dn, 6, N_dn, 3)
    r_dn_carts_shifted = (r_dn[None, None, :, :] + delta_dn).reshape(
        num_dn_configs, num_dn_electrons, 3
    )  # (num_dn_configs, N_dn, 3)

    # Broadcast unchanged spin block (avoids repeat allocation)
    r_up_carts_repeated_dn = jnp.broadcast_to(r_up[None, :, :], (num_dn_configs, r_up.shape[0], 3))  # (num_dn_configs, N_up, 3)

    # Combine configurations
    r_up_carts_combined = jnp.concatenate([r_up_carts_shifted, r_up_carts_repeated_dn], axis=0)  # Shape: (N_configs, N_up, 3)
    r_dn_carts_combined = jnp.concatenate([r_dn_carts_repeated_up, r_dn_carts_shifted], axis=0)  # Shape: (N_configs, N_dn, 3)

    # Evaluate the ratios of wavefunctions between the shifted positions and the original position.
    # det_ratio (det_ratio zone) and jastrow_ratio (jastrow_ratio zone) are explicitly cast to
    # the wf_ratio zone dtype to avoid relying on JAX implicit promotion.
    dtype_wf_ratio_jnp = get_dtype_jnp("wf_ratio")
    det_ratio = jnp.asarray(
        _compute_ratio_determinant_part_split_spin(
            geminal_data=wavefunction_data.geminal_data,
            A_old_inv=A_old_inv,
            old_r_up_carts=r_up,
            old_r_dn_carts=r_dn,
            new_r_up_shifted=r_up_carts_shifted,
            new_r_dn_shifted=r_dn_carts_shifted,
        ),
        dtype=dtype_wf_ratio_jnp,
    )
    jastrow_ratio = jnp.asarray(
        _compute_ratio_Jastrow_part_rank1_update(
            jastrow_data=wavefunction_data.jastrow_data,
            old_r_up_carts=r_up,
            old_r_dn_carts=r_dn,
            new_r_up_carts_arr=r_up_carts_combined,
            new_r_dn_carts_arr=r_dn_carts_combined,
            j3_state=j3_state,
        ),
        dtype=dtype_wf_ratio_jnp,
    )
    wf_ratio = det_ratio * jastrow_ratio

    # Compute the kinetic part elements
    elements_kinetic_part = -1.0 / (2.0 * alat**2) * wf_ratio

    # Return the combined configurations and the kinetic elements
    return r_up_carts_combined, r_dn_carts_combined, elements_kinetic_part


# ---------------------------------------------------------------------------
# Nodal distance and f_epsilon regularization
# ---------------------------------------------------------------------------
# References:
#   Pathak & Wagner, AIP Advances 10, 085213 (2020).
#   DOI: 10.1063/5.0004008
# ---------------------------------------------------------------------------


def f_epsilon_PW(nodal_distance: jnpt.ArrayLike, epsilon_PW: float) -> jnpt.ArrayLike:
    r"""Evaluate the Pathak--Wagner regularization function :math:`f_\varepsilon`.

    .. math::

        f_\varepsilon(t) =
        \begin{cases}
            7|t|^6 - 15|t|^4 + 9|t|^2 & (|t| < 1) \\
            1                           & (|t| \geq 1)
        \end{cases}

    where :math:`t = |x| / \varepsilon` and :math:`|x|` is the nodal distance.

    The coefficients :math:`(7, -15, 9)` satisfy :math:`f(0) = 0`, :math:`f(1) = 1`,
    and continuity of the first two derivatives at :math:`|t| = 1`.

    Args:
        nodal_distance: Nodal distance(s), shape arbitrary (scalar or array).
        epsilon_PW: Regularization cutoff length :math:`\varepsilon`.

    Returns:
        Regularization factor with the same shape as *nodal_distance*.
    """
    t = jnp.abs(nodal_distance) / epsilon_PW
    t2 = t * t
    t4 = t2 * t2
    t6 = t4 * t2
    f_inner = 7.0 * t6 - 15.0 * t4 + 9.0 * t2
    return jnp.where(t < 1.0, f_inner, 1.0)


def compute_nodal_distance(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> jax.Array:
    r"""Compute the nodal distance using analytic derivatives.

    The nodal distance is defined as

    .. math::

        |x| = \frac{1}{|\nabla \ln |\Psi||},

    where :math:`\nabla` is the many-body gradient over all electron coordinates,
    computed analytically from the Jastrow and determinant parts.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons with shape ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons with shape ``(n_dn, 3)``.

    Returns:
        Scalar nodal distance value.
    """
    # r_*_carts forwarded unchanged (see ``evaluate_ln_wavefunction`` for rationale).
    dtype_jnp = get_dtype_jnp("wf_eval")

    grad_J_up, grad_J_dn, _, _ = compute_grads_and_laplacian_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    grad_ln_D_up, grad_ln_D_dn, _, _ = compute_grads_and_laplacian_ln_Det(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # Cast jastrow_grad_lap / det_grad_lap zone values to wf_eval dtype.
    grad_J_up = jnp.asarray(grad_J_up, dtype=dtype_jnp)
    grad_J_dn = jnp.asarray(grad_J_dn, dtype=dtype_jnp)
    grad_ln_D_up = jnp.asarray(grad_ln_D_up, dtype=dtype_jnp)
    grad_ln_D_dn = jnp.asarray(grad_ln_D_dn, dtype=dtype_jnp)

    grad_ln_Psi_up = grad_J_up + grad_ln_D_up  # (n_up, 3)
    grad_ln_Psi_dn = grad_J_dn + grad_ln_D_dn  # (n_dn, 3)

    grad_norm_sq = jnp.sum(grad_ln_Psi_up**2) + jnp.sum(grad_ln_Psi_dn**2)
    return 1.0 / jnp.sqrt(grad_norm_sq)


def _compute_nodal_distance_debug(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> jax.Array:
    r"""Compute the nodal distance using the paper's original formula (debug).

    Uses the definition from Eq. (2) of Pathak & Wagner (2020):

    .. math::

        \vec{x} = \frac{\Psi \, \nabla \Psi}{|\nabla \Psi|^2},

    and returns :math:`|x|`.  This is mathematically identical to
    :func:`compute_nodal_distance` (:math:`1/|\nabla \ln|\Psi||`), but uses
    :func:`evaluate_wavefunction` and automatic differentiation of :math:`\Psi`
    instead of analytic :math:`\nabla \ln|\Psi|` derivatives.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons with shape ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons with shape ``(n_dn, 3)``.

    Returns:
        Scalar nodal distance value.
    """
    dtype_jnp = get_dtype_jnp("wf_eval")
    r_up = jnp.asarray(r_up_carts, dtype=dtype_jnp)
    r_dn = jnp.asarray(r_dn_carts, dtype=dtype_jnp)

    Psi = evaluate_wavefunction(wavefunction_data, r_up, r_dn)

    grad_Psi_r_up = grad(evaluate_wavefunction, argnums=1)(wavefunction_data, r_up, r_dn)  # (n_up, 3)
    grad_Psi_r_dn = grad(evaluate_wavefunction, argnums=2)(wavefunction_data, r_up, r_dn)  # (n_dn, 3)

    grad_Psi_norm_sq = jnp.sum(grad_Psi_r_up**2) + jnp.sum(grad_Psi_r_dn**2)

    # x_vec = Psi * grad_Psi / |grad_Psi|^2, so |x| = |Psi| / |grad_Psi|
    return jnp.abs(Psi) / jnp.sqrt(grad_Psi_norm_sq)


"""
if __name__ == "__main__":
    log = getLogger("jqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)
"""
