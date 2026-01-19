"""SWCT module."""

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
# import os
from logging import getLogger

# JAX
import jax
import numpy as np
import numpy.typing as npt
from flax import struct
from jax import jacrev, jit, vmap
from jax import numpy as jnp
from jax import typing as jnpt

# jQMC modules
from .structure import Structure_data

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


@struct.dataclass
class SWCT_data:
    """Space-warp coordinate transformation metadata.

    Args:
        structure (Structure_data): Nuclear geometry used to build SWCT weights.
    """

    structure: Structure_data = struct.field(pytree_node=True)

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        self.structure.sanity_check()


@jit
def evaluate_swct_omega(
    swct_data: SWCT_data,
    r_carts: jax.Array,
) -> jax.Array:
    r"""Compute SWCT weights :math:`\omega_{\alpha i}` for each atom/electron pair.

    Args:
        swct_data (SWCT_data): Structure and cached geometry information.
        r_carts (jax.Array): Electron Cartesian coordinates with shape ``(N_e, 3)`` and ``float64`` dtype.

    Returns:
        jax.Array: Normalized weights with shape ``(N_a, N_e)``, summing to 1 over atoms for each electron.
    """
    R_carts = swct_data.structure._positions_cart_jnp

    def compute_omega(R_cart, r_cart):
        kappa = 1.0 / jnp.linalg.norm(r_cart - R_cart) ** 4
        kappa_sum = jnp.sum(1.0 / jnp.linalg.norm(r_cart - R_carts, axis=1) ** 4)
        return kappa / kappa_sum

    vmap_compute_omega = vmap(
        vmap(
            compute_omega,
            in_axes=(None, 0),
        ),
        in_axes=(0, None),
    )

    omega = vmap_compute_omega(R_carts, r_carts)

    return omega


def _evaluate_swct_omega_debug(
    swct_data: SWCT_data,
    r_carts: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """NumPy fallback for ``evaluate_swct_omega`` used in debug paths."""
    R_carts = swct_data.structure._positions_cart_np
    omega = np.zeros((len(R_carts), len(r_carts)))

    for alpha in range(len(R_carts)):
        for i in range(len(r_carts)):
            kappa = 1.0 / np.linalg.norm(r_carts[i] - R_carts[alpha]) ** 4
            kappa_sum = np.sum([1.0 / np.linalg.norm(r_carts[i] - R_carts[beta]) ** 4 for beta in range(len(R_carts))])
            omega[alpha, i] = kappa / kappa_sum

    return omega


@jit
def evaluate_swct_domega(
    swct_data: SWCT_data,
    r_carts: jax.Array,
) -> npt.NDArray[np.float64]:
    r"""Evaluate :math:`\sum_i \nabla_{r_i} \omega_{\alpha i}` for each atom.

    Args:
        swct_data (SWCT_data): Structure and cached geometry information.
        r_carts (jax.Array): Electron Cartesian coordinates with shape ``(N_e, 3)`` and ``float64`` dtype.

    Returns:
        jax.Array: Sum of gradients per atom with shape ``(N_a, 3)``.
    """
    domega = jnp.sum(jacrev(evaluate_swct_omega, argnums=1)(swct_data, r_carts), axis=(1, 2))

    return domega


def _evaluate_swct_domega_debug(
    swct_data: SWCT_data,
    r_carts: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """NumPy fallback for ``evaluate_swct_domega`` used in debug paths."""
    R_carts = swct_data.structure._positions_cart_np
    domega = np.zeros((len(R_carts), 3))

    def compute_omega(R_cart, r_cart, R_carts):
        kappa = 1.0 / np.linalg.norm(r_cart - R_cart) ** 4
        kappa_sum = np.sum([1.0 / np.linalg.norm(r_cart - R_carts[beta]) ** 4 for beta in range(len(R_carts))])
        omega = kappa / kappa_sum
        return omega

    diff_h = 1.0e-5

    for i, R_cart in enumerate(R_carts):
        for r_cart in r_carts:
            r_cart_p_dx = r_cart.copy()
            r_cart_m_dx = r_cart.copy()
            r_cart_p_dx[0] += diff_h
            r_cart_m_dx[0] -= diff_h
            omega_dx = (compute_omega(R_cart, r_cart_p_dx, R_carts) - compute_omega(R_cart, r_cart_m_dx, R_carts)) / (
                2 * diff_h
            )

            r_cart_p_dy = r_cart.copy()
            r_cart_m_dy = r_cart.copy()
            r_cart_p_dy[1] += diff_h
            r_cart_m_dy[1] -= diff_h
            omega_dy = (compute_omega(R_cart, r_cart_p_dy, R_carts) - compute_omega(R_cart, r_cart_m_dy, R_carts)) / (
                2 * diff_h
            )

            r_cart_p_dz = r_cart.copy()
            r_cart_m_dz = r_cart.copy()
            r_cart_p_dz[2] += diff_h
            r_cart_m_dz[2] -= diff_h
            omega_dz = (compute_omega(R_cart, r_cart_p_dz, R_carts) - compute_omega(R_cart, r_cart_m_dz, R_carts)) / (
                2 * diff_h
            )

            domega[i, 0] += omega_dx
            domega[i, 1] += omega_dy
            domega[i, 2] += omega_dz

    return domega


"""
if __name__ == "__main__":
    from logging import Formatter, StreamHandler

    log = getLogger("jqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    from .trexio_wrapper import read_trexio_file

    # water  cc-pVTZ with Mitas ccECP.
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "water_trexio.hdf5"))

    swct_data = SWCT_data(structure=structure_data)

    num_ele_up = geminal_mo_data.num_electron_up
    num_ele_dn = geminal_mo_data.num_electron_dn
    r_cart_min, r_cart_max = -5.0, +5.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_up, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_dn, 3) + r_cart_min

    omega_up = evaluate_swct_omega_jax(swct_data=swct_data, r_carts=r_up_carts, debug=True)
    omega_dn = evaluate_swct_omega_jax(swct_data=swct_data, r_carts=r_dn_carts, debug=True)
    print(omega_up)
    print(omega_dn)
    omega_up = evaluate_swct_omega_jax(swct_data=swct_data, r_carts=r_up_carts, debug=False)
    omega_dn = evaluate_swct_omega_jax(swct_data=swct_data, r_carts=r_dn_carts, debug=False)
    print(omega_up)
    print(omega_dn)

    d_omega_up = np.sum(jacrev(evaluate_swct_omega_jax, argnums=1)(swct_data, r_up_carts), axis=(1, 2))
    print(f"shape(d_omega_up) = {d_omega_up.shape}")
    d_omega_dn = np.sum(jacrev(evaluate_swct_omega_jax, argnums=1)(swct_data, r_dn_carts), axis=(1, 2))
    print(f"shape(d_omega_dn) = {d_omega_dn.shape}")
    print(d_omega_up)
    print(d_omega_dn)

    d_omega_up = evaluate_swct_domega_debug(swct_data, r_up_carts)
    print(d_omega_up)
    print(f"shape(d_omega_up) = {d_omega_up.shape}")
    d_omega_dn = evaluate_swct_domega_debug(swct_data, r_dn_carts)
    print(d_omega_dn)
    print(f"shape(d_omega_dn) = {d_omega_dn.shape}")
"""
