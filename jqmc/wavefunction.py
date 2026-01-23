"""Wavefunction module."""

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

from .determinant import (
    Geminal_data,
    compute_det_geminal_all_elements,
    compute_grads_and_laplacian_ln_Det,
    compute_grads_and_laplacian_ln_Det_fast,
    compute_ln_det_geminal_all_elements,
    compute_ratio_determinant_part,
)
from .diff_mask import DiffMask, apply_diff_mask
from .jastrow_factor import (
    Jastrow_data,
    compute_grads_and_laplacian_Jastrow_part,
    compute_Jastrow_part,
    compute_ratio_Jastrow_part,
)

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


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

        geminal_data = self.geminal_data
        geminal_updates = {}
        if geminal_data is not None:
            geminal_masked, changed = _maybe_mask(geminal_data, opt_lambda_param)
            if changed:
                geminal_updates["lambda_matrix"] = geminal_masked.lambda_matrix

            if geminal_updates:
                geminal_data = geminal_data.replace(**geminal_updates)

        if jastrow_updates or geminal_updates:
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
                blocks.append(
                    VariationalParameterBlock(
                        name="j3_matrix",
                        values=j3_arr,
                        shape=j3_arr.shape,
                        size=int(j3_arr.size),
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
            blocks.append(
                VariationalParameterBlock(
                    name="lambda_matrix",
                    values=lam_arr,
                    shape=lam_arr.shape,
                    size=int(lam_arr.size),
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
    Jastrow contribution. The inputs are converted to float64 ``jax.Array`` for
    downstream consistency.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons with shape ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons with shape ``(n_dn, 3)``.

    Returns:
        Scalar log-value of the wavefunction magnitude.
    """
    r_up = jnp.asarray(r_up_carts, dtype=jnp.float64)
    r_dn = jnp.asarray(r_dn_carts, dtype=jnp.float64)

    Jastrow_part = compute_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up,
        r_dn_carts=r_dn,
    )

    Determinant_part = compute_det_geminal_all_elements(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up,
        r_dn_carts=r_dn,
    )

    return Jastrow_part + jnp.log(jnp.abs(Determinant_part))


@jit
def evaluate_wavefunction(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float | complex:
    """Evaluate the wavefunction ``Psi`` at given electron coordinates.

    The method is for evaluate wavefunction (Psi) at ``(r_up_carts, r_dn_carts)`` and
    returns ``exp(Jastrow) * Determinant``. Inputs are coerced to float64
    ``jax.Array`` to match other compute utilities.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons with shape ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons with shape ``(n_dn, 3)``.

    Returns:
        Complex or real wavefunction value.
    """
    r_up = jnp.asarray(r_up_carts, dtype=jnp.float64)
    r_dn = jnp.asarray(r_dn_carts, dtype=jnp.float64)

    Jastrow_part = compute_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up,
        r_dn_carts=r_dn,
    )

    Determinant_part = compute_det_geminal_all_elements(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up,
        r_dn_carts=r_dn,
    )

    return jnp.exp(Jastrow_part) * Determinant_part


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
    r_up = jnp.asarray(r_up_carts, dtype=jnp.float64)
    r_dn = jnp.asarray(r_dn_carts, dtype=jnp.float64)

    Jastrow_part = compute_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up,
        r_dn_carts=r_dn,
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
    r_up = jnp.asarray(r_up_carts, dtype=jnp.float64)
    r_dn = jnp.asarray(r_dn_carts, dtype=jnp.float64)

    Determinant_part = compute_det_geminal_all_elements(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up,
        r_dn_carts=r_dn,
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
    kinetic energy calculation. Inputs are converted to float64 ``jax.Array``
    for consistency with other compute utilities.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons with shape ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons with shape ``(n_dn, 3)``.

    Returns:
        Kinetic energy evaluated for the supplied configuration.
    """
    r_up = jnp.asarray(r_up_carts, dtype=jnp.float64)
    r_dn = jnp.asarray(r_dn_carts, dtype=jnp.float64)

    # grad_J_up, grad_J_dn, sum_laplacian_J = 0.0, 0.0, 0.0
    # """
    grad_J_up, grad_J_dn, lap_J_up, lap_J_dn = compute_grads_and_laplacian_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up,
        r_dn_carts=r_dn,
    )
    # """

    # grad_ln_D_up, grad_ln_D_dn, sum_laplacian_ln_D = 0.0, 0.0, 0.0
    # """
    grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn = compute_grads_and_laplacian_ln_Det(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up,
        r_dn_carts=r_dn,
    )
    # """

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
    r_up = jnp.asarray(r_up_carts, dtype=jnp.float64)
    r_dn = jnp.asarray(r_dn_carts, dtype=jnp.float64)

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
    """See compute_kinetic_energy_api."""
    # compute laplacians
    diff_h = 2.0e-4

    Psi = evaluate_wavefunction(wavefunction_data, r_up_carts, r_dn_carts)

    n_up, d_up = r_up_carts.shape
    laplacian_Psi_up = np.zeros(n_up)
    for i in range(n_up):
        for d in range(d_up):
            r_up_plus = r_up_carts.copy()
            r_up_minus = r_up_carts.copy()
            r_up_plus[i, d] += diff_h
            r_up_minus[i, d] -= diff_h

            Psi_plus = evaluate_wavefunction(wavefunction_data, r_up_plus, r_dn_carts)
            Psi_minus = evaluate_wavefunction(wavefunction_data, r_up_minus, r_dn_carts)

            laplacian_Psi_up[i] += (Psi_plus + Psi_minus - 2 * Psi) / (diff_h**2)

    n_dn, d_dn = r_dn_carts.shape
    laplacian_Psi_dn = np.zeros(n_dn)
    for i in range(n_dn):
        for d in range(d_dn):
            r_dn_plus = r_dn_carts.copy()
            r_dn_minus = r_dn_carts.copy()
            r_dn_plus[i, d] += diff_h
            r_dn_minus[i, d] -= diff_h

            Psi_plus = evaluate_wavefunction(wavefunction_data, r_up_carts, r_dn_plus)
            Psi_minus = evaluate_wavefunction(wavefunction_data, r_up_carts, r_dn_minus)

            laplacian_Psi_dn[i] += (Psi_plus + Psi_minus - 2 * Psi) / (diff_h**2)

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
    r_up = jnp.asarray(r_up_carts, dtype=jnp.float64)
    r_dn = jnp.asarray(r_dn_carts, dtype=jnp.float64)

    # compute gradients
    grad_J_up = grad(compute_Jastrow_part, argnums=1)(wavefunction_data.jastrow_data, r_up, r_dn)
    grad_J_dn = grad(compute_Jastrow_part, argnums=2)(wavefunction_data.jastrow_data, r_up, r_dn)
    grad_ln_Det_up = grad(compute_ln_det_geminal_all_elements, argnums=1)(wavefunction_data.geminal_data, r_up, r_dn)
    grad_ln_Det_dn = grad(compute_ln_det_geminal_all_elements, argnums=2)(wavefunction_data.geminal_data, r_up, r_dn)

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
    r_up = jnp.asarray(r_up_carts, dtype=jnp.float64)
    r_dn = jnp.asarray(r_dn_carts, dtype=jnp.float64)

    # --- Jastrow contributions (per-electron Laplacians) ---
    grad_J_up, grad_J_dn, lap_J_up, lap_J_dn = compute_grads_and_laplacian_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up,
        r_dn_carts=r_dn,
    )

    # --- Determinant contributions (per-electron Laplacians) ---
    grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn = compute_grads_and_laplacian_ln_Det(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up,
        r_dn_carts=r_dn,
    )

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
    """Kinetic energy per electron using a precomputed geminal inverse."""
    if geminal_inverse is None:
        raise ValueError("geminal_inverse must be provided for fast update")

    r_up = jnp.asarray(r_up_carts, dtype=jnp.float64)
    r_dn = jnp.asarray(r_dn_carts, dtype=jnp.float64)

    grad_J_up, grad_J_dn, lap_J_up, lap_J_dn = compute_grads_and_laplacian_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up,
        r_dn_carts=r_dn,
    )

    grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn = compute_grads_and_laplacian_ln_Det_fast(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up,
        r_dn_carts=r_dn,
        geminal_inverse=geminal_inverse,
    )

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
    coerced to float64 ``jax.Array`` before evaluation.

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
    r_up = jnp.asarray(r_up_carts, dtype=jnp.float64)
    r_dn = jnp.asarray(r_dn_carts, dtype=jnp.float64)
    rt = jnp.asarray(RT, dtype=jnp.float64)
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
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Fast-update version of discretized kinetic mesh and ratios.

    Function for computing discretized kinetic grid points and their energies with
    a given lattice space (alat). Uses precomputed ``A_old_inv`` to evaluate
    determinant ratios efficiently. Inputs are converted to float64 ``jax.Array``
    before use.

    Args:
        alat: Hamiltonian discretization (bohr), which will be replaced with ``LRDMC_data``.
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        A_old_inv: Inverse of the geminal matrix evaluated at ``(r_up_carts, r_dn_carts)``.
        r_up_carts: Up-electron positions with shape ``(n_up, 3)``.
        r_dn_carts: Down-electron positions with shape ``(n_dn, 3)``.
        RT: Rotation matrix (:math:`R^T`) with shape ``(3, 3)``.

    Returns:
        Tuple ``(r_up_carts_combined, r_dn_carts_combined, elements_kinetic_part)`` with combined
        coordinate arrays of shapes ``(n_grid, n_up, 3)`` and ``(n_grid, n_dn, 3)``, and kinetic
        prefactor-scaled ratios ``elements_kinetic_part``.
    """
    r_up = jnp.asarray(r_up_carts, dtype=jnp.float64)
    r_dn = jnp.asarray(r_dn_carts, dtype=jnp.float64)
    rt = jnp.asarray(RT, dtype=jnp.float64)
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

    # Evaluate the ratios of wavefunctions between the shifted positions and the original position
    wf_ratio = compute_ratio_determinant_part(
        geminal_data=wavefunction_data.geminal_data,
        A_old_inv=A_old_inv,
        old_r_up_carts=r_up,
        old_r_dn_carts=r_dn,
        new_r_up_carts_arr=r_up_carts_combined,
        new_r_dn_carts_arr=r_dn_carts_combined,
    ) * compute_ratio_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        old_r_up_carts=r_up,
        old_r_dn_carts=r_dn,
        new_r_up_carts_arr=r_up_carts_combined,
        new_r_dn_carts_arr=r_dn_carts_combined,
    )

    # Compute the kinetic part elements
    elements_kinetic_part = -1.0 / (2.0 * alat**2) * wf_ratio

    # Return the combined configurations and the kinetic elements
    return r_up_carts_combined, r_dn_carts_combined, elements_kinetic_part


# no longer used in the main code
def compute_quantum_force(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute quantum forces ``2 * grad ln |Psi|`` at the given coordinates.

    The method is for computing quantum forces at ``(r_up_carts, r_dn_carts)``.
    Gradients from the Jastrow part are currently set to zero (as in the original
    implementation); determinant gradients are included via
    ``compute_grads_and_laplacian_ln_Det``. Inputs are coerced to float64
    ``jax.Array`` for consistency.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons with shape ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons with shape ``(n_dn, 3)``.

    Returns:
        Tuple ``(force_up, force_dn)`` with shapes matching the input coordinate arrays.
    """
    r_up = jnp.asarray(r_up_carts, dtype=jnp.float64)
    r_dn = jnp.asarray(r_dn_carts, dtype=jnp.float64)

    grad_J_up, grad_J_dn, _ = 0, 0, 0  # tentative

    grad_ln_D_up, grad_ln_D_dn, _ = compute_grads_and_laplacian_ln_Det(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up,
        r_dn_carts=r_dn,
    )

    grad_ln_WF_up = grad_J_up + grad_ln_D_up
    grad_ln_WF_dn = grad_J_dn + grad_ln_D_dn

    return 2.0 * grad_ln_WF_up, 2.0 * grad_ln_WF_dn


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
