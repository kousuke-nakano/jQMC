"""Jastrow module"""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
#
# This file is part of phonopy.
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
# * Neither the name of the phonopy project nor the names of its
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

# set logger
from logging import Formatter, StreamHandler, getLogger

# jax modules
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import struct
from jax import grad, hessian, jacrev, jit, vmap

# jqmc module
from .atomic_orbital import AOs_data, AOs_data_debug, compute_AOs_api

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


# @dataclass
@struct.dataclass
class Jastrow_two_body_data:
    """
    The class contains data for evaluating the two-body Jastrow function.

    Args:
        jastrow_2b_param (float): the parameter for 2b Jastrow part
    """

    jastrow_2b_param: float = struct.field(pytree_node=True)

    def __post_init__(self) -> None:
        pass


# @dataclass
@struct.dataclass
class Jastrow_three_body_data:
    """
    The class contains data for evaluating the three-body Jastrow function.

    Args:
        orb_data_up_spin (AOs_data): AOs data for up-spin.
        orb_data_dn_spin (AOs_data): AOs data for dn-spin.
        j_matrix_up_up (npt.NDArray[np.float64]): J matrix dim. (orb_data_up_spin.num_ao, orb_data_up_spin.num_ao+1))
        j_matrix_dn_dn (npt.NDArray[np.float64]): J matrix dim. (orb_data_dn_spin.num_ao, orb_data_dn_spin.num_ao+1))
        j_matrix_up_dn (npt.NDArray[np.float64]): J matrix dim. (orb_data_up_spin.num_ao, orb_data_dn_spin.num_ao))
    """

    orb_data_up_spin: AOs_data = struct.field(pytree_node=True)
    orb_data_dn_spin: AOs_data = struct.field(pytree_node=True)
    j_matrix_up_up: npt.NDArray[np.float64] = struct.field(pytree_node=True)
    j_matrix_dn_dn: npt.NDArray[np.float64] = struct.field(pytree_node=True)
    j_matrix_up_dn: npt.NDArray[np.float64] = struct.field(pytree_node=True)

    def __post_init__(self) -> None:
        if self.j_matrix_up_up.shape != (
            self.orb_num_up,
            self.orb_num_up + 1,
        ):
            logger.error(
                f"dim. of j_matrix_up_up = {self.j_matrix_up_up.shape} is imcompatible with the expected one "
                + f"= ({self.orb_num_up}, {self.orb_num_up + 1}).",
            )
            raise ValueError

        if self.j_matrix_dn_dn.shape != (
            self.orb_num_dn,
            self.orb_num_dn + 1,
        ):
            logger.error(
                f"dim. of j_matrix_dn_dn = {self.j_matrix_dn_dn.shape} is imcompatible with the expected one "
                + f"= ({self.orb_num_dn}, {self.orb_num_dn + 1}).",
            )
            raise ValueError

        if self.j_matrix_up_dn.shape != (
            self.orb_num_up,
            self.orb_num_dn,
        ):
            logger.error(
                f"dim. of j_matrix_up_dn = {self.j_matrix_up_dn.shape} is imcompatible with the expected one "
                + f"= ({self.orb_num_up}, {self.orb_num_dn}).",
            )
            raise ValueError

    @property
    def orb_num_up(self) -> int:
        return self.orb_data_up_spin.num_ao

    @property
    def orb_num_dn(self) -> int:
        return self.orb_data_dn_spin.num_ao


# @dataclass
@struct.dataclass
class Jastrow_data:
    """
    The class contains data for evaluating a Jastrow function.

    Args:
        jastrow_two_body_data (Jastrow_two_body_data): parameter for parallel spins
        param_antiparallel_spin (float): parameter for anti-parallel spins
    """

    jastrow_two_body_type: str = struct.field(pytree_node=False)  # off, on
    jastrow_three_body_type: str = struct.field(pytree_node=False)  # off, on
    jastrow_two_body_data: Jastrow_two_body_data = struct.field(pytree_node=True)
    jastrow_three_body_data: Jastrow_three_body_data = struct.field(pytree_node=True)

    def __post_init__(self) -> None:
        pass


def compute_Jastrow_part_api(
    jastrow_data: Jastrow_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
):
    # two-body
    if jastrow_data.jastrow_two_body_type == "off":
        J2 = 0.0
    elif jastrow_data.jastrow_two_body_type == "on":
        J2 = compute_Jastrow_two_body_api(
            jastrow_data.jastrow_two_body_data,
            r_up_carts,
            r_dn_carts,
        )
    else:
        raise NotImplementedError

    # three-body
    if jastrow_data.jastrow_three_body_type == "off":
        J3 = 0.0
    elif jastrow_data.jastrow_three_body_type == "on":
        J3 = compute_Jastrow_three_body_api(
            jastrow_data.jastrow_three_body_data,
            r_up_carts,
            r_dn_carts,
        )
    else:
        raise NotImplementedError

    J = J2 + J3

    return J


def compute_Jastrow_three_body_api(
    jastrow_three_body_data: Jastrow_three_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    # return compute_Jastrow_three_body_debug(jastrow_three_body_data, r_up_carts, r_dn_carts)
    return compute_Jastrow_three_body_jax(jastrow_three_body_data, r_up_carts, r_dn_carts)


def compute_Jastrow_three_body_debug(
    jastrow_three_body_data: Jastrow_three_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    aos_up = compute_AOs_api(aos_data=jastrow_three_body_data.orb_data_up_spin, r_carts=r_up_carts)
    aos_dn = compute_AOs_api(aos_data=jastrow_three_body_data.orb_data_dn_spin, r_carts=r_dn_carts)

    # compute one body
    J_1_up = 0.0
    j1_vector_up = jastrow_three_body_data.j_matrix_up_up[:, -1]
    for i in range(len(r_up_carts)):
        ao_up = aos_up[:, i]
        for al in range(len(ao_up)):
            J_1_up += j1_vector_up[al] * ao_up[al]

    J_1_dn = 0.0
    j1_vector_dn = jastrow_three_body_data.j_matrix_dn_dn[:, -1]
    for i in range(len(r_dn_carts)):
        ao_dn = aos_dn[:, i]
        for al in range(len(ao_dn)):
            J_1_dn += j1_vector_dn[al] * ao_dn[al]

    # compute three-body
    J_3_up_up = 0.0
    j3_matrix_up_up = jastrow_three_body_data.j_matrix_up_up[:, :-1]
    for i in range(len(r_up_carts)):
        for j in range(i + 1, len(r_up_carts)):
            ao_up_i = aos_up[:, i]
            ao_up_j = aos_up[:, j]
            for al in range(len(ao_up_i)):
                for bm in range(len(ao_up_j)):
                    J_3_up_up += j3_matrix_up_up[al, bm] * ao_up_i[al] * ao_up_j[bm]

    J_3_dn_dn = 0.0
    j3_matrix_dn_dn = jastrow_three_body_data.j_matrix_dn_dn[:, :-1]
    for i in range(len(r_dn_carts)):
        for j in range(i + 1, len(r_dn_carts)):
            ao_dn_i = aos_dn[:, i]
            ao_dn_j = aos_dn[:, j]
            for al in range(len(ao_dn_i)):
                for bm in range(len(ao_dn_j)):
                    J_3_dn_dn += j3_matrix_dn_dn[al, bm] * ao_dn_i[al] * ao_dn_j[bm]

    J_3_up_dn = 0.0
    j3_matrix_up_dn = jastrow_three_body_data.j_matrix_up_dn[:, :]
    for i in range(len(r_up_carts)):
        for j in range(len(r_dn_carts)):
            ao_up_i = aos_up[:, i]
            ao_dn_j = aos_dn[:, j]
            for al in range(len(ao_up_i)):
                for bm in range(len(ao_dn_j)):
                    J_3_up_dn += j3_matrix_up_dn[al, bm] * ao_up_i[al] * ao_dn_j[bm]

    J3 = J_1_up + J_1_dn + J_3_up_up + J_3_dn_dn + J_3_up_dn

    return J3


@jit
def compute_Jastrow_three_body_jax(
    jastrow_three_body_data: Jastrow_three_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    num_electron_up = len(r_up_carts)
    num_electron_dn = len(r_dn_carts)

    aos_up = jnp.array(
        compute_AOs_api(aos_data=jastrow_three_body_data.orb_data_up_spin, r_carts=r_up_carts)
    )
    aos_dn = jnp.array(
        compute_AOs_api(aos_data=jastrow_three_body_data.orb_data_dn_spin, r_carts=r_dn_carts)
    )

    K_up = jnp.tril(jnp.ones((num_electron_up, num_electron_up)), k=-1)
    K_dn = jnp.tril(jnp.ones((num_electron_dn, num_electron_dn)), k=-1)

    j1_matrix_up = jastrow_three_body_data.j_matrix_up_up[:, -1]
    j1_matrix_dn = jastrow_three_body_data.j_matrix_dn_dn[:, -1]
    j3_matrix_up_up = jastrow_three_body_data.j_matrix_up_up[:, :-1]
    j3_matrix_dn_dn = jastrow_three_body_data.j_matrix_dn_dn[:, :-1]
    j3_matrix_up_dn = jastrow_three_body_data.j_matrix_up_dn

    e_up = jnp.ones(num_electron_up).T
    e_dn = jnp.ones(num_electron_dn).T

    J3 = (
        j1_matrix_up @ aos_up @ e_up
        + j1_matrix_dn @ aos_dn @ e_dn
        + jnp.trace(aos_up.T @ j3_matrix_up_up @ aos_up @ K_up)
        + jnp.trace(aos_dn.T @ j3_matrix_dn_dn @ aos_dn @ K_dn)
        + e_up.T @ aos_up.T @ j3_matrix_up_dn @ aos_dn @ e_dn
    )

    return J3


def compute_grads_and_laplacian_Jastrow_part_api(
    jastrow_data: Jastrow_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64 | np.complex128],
    npt.NDArray[np.float64 | np.complex128],
    float | complex,
]:
    # two-body
    if jastrow_data.jastrow_two_body_type == "off":
        grad_J2_up, grad_J2_dn, sum_laplacian_J2 = 0.0, 0.0, 0.0
    elif jastrow_data.jastrow_two_body_type == "on":
        grad_J2_up, grad_J2_dn, sum_laplacian_J2 = compute_grads_and_laplacian_Jastrow_two_body_api(
            jastrow_data.jastrow_two_body_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
        )
    else:
        raise NotImplementedError

    # three-body
    if jastrow_data.jastrow_three_body_type == "off":
        grad_J3_up, grad_J3_dn, sum_laplacian_J3 = 0.0, 0.0, 0.0
    elif jastrow_data.jastrow_three_body_type == "on":
        grad_J3_up, grad_J3_dn, sum_laplacian_J3 = (
            compute_grads_and_laplacian_Jastrow_three_body_api(
                jastrow_data.jastrow_three_body_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )
        )
    else:
        raise NotImplementedError

    grad_J_up = grad_J2_up + grad_J3_up
    grad_J_dn = grad_J2_dn + grad_J3_dn
    sum_laplacian_J = sum_laplacian_J2 + sum_laplacian_J3

    return grad_J_up, grad_J_dn, sum_laplacian_J


def compute_Jastrow_two_body_api(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    # return compute_Jastrow_two_body_debug(jastrow_two_body_data, r_up_carts, r_dn_carts)
    return compute_Jastrow_two_body_jax(jastrow_two_body_data, r_up_carts, r_dn_carts)


def compute_Jastrow_two_body_debug(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    def two_body_jastrow_anti_parallel_spins(
        param: float, rel_r_cart: npt.NDArray[np.float64]
    ) -> float:
        # """ exp
        two_body_jastrow = 1.0 / (2.0 * param) * (1.0 - np.exp(-param * np.linalg.norm(rel_r_cart)))
        return two_body_jastrow
        # """

        """pade
        two_body_jastrow = (
            np.linalg.norm(rel_r_cart)
            / 2.0
            * (1.0 + param * np.linalg.norm(rel_r_cart)) ** (-1.0)
        )
        return two_body_jastrow
        """

    def two_body_jastrow_parallel_spins(param: float, rel_r_cart: npt.NDArray[np.float64]) -> float:
        # """ exp
        two_body_jastrow = 1.0 / (2.0 * param) * (1.0 - np.exp(-param * np.linalg.norm(rel_r_cart)))
        return two_body_jastrow
        # """

        """pade
        two_body_jastrow = (
            np.linalg.norm(rel_r_cart)
            / 4.0
            * (1.0 + param * np.linalg.norm(rel_r_cart)) ** (-1.0)
        )
        return two_body_jastrow
        """

    two_body_jastrow = (
        np.sum(
            [
                two_body_jastrow_anti_parallel_spins(
                    param=jastrow_two_body_data.jastrow_2b_param,
                    rel_r_cart=r_up_cart - r_dn_cart,
                )
                for (r_up_cart, r_dn_cart) in itertools.product(r_up_carts, r_dn_carts)
            ]
        )
        + np.sum(
            [
                two_body_jastrow_parallel_spins(
                    param=jastrow_two_body_data.jastrow_2b_param,
                    rel_r_cart=r_up_cart_i - r_up_cart_j,
                )
                for (r_up_cart_i, r_up_cart_j) in itertools.combinations(r_up_carts, 2)
            ]
        )
        + np.sum(
            [
                two_body_jastrow_parallel_spins(
                    param=jastrow_two_body_data.jastrow_2b_param,
                    rel_r_cart=r_dn_cart_i - r_dn_cart_j,
                )
                for (r_dn_cart_i, r_dn_cart_j) in itertools.combinations(r_dn_carts, 2)
            ]
        )
    )

    return two_body_jastrow


@jit
def compute_Jastrow_two_body_jax(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    def J2_anti_parallel_spins(r_cart_i, r_cart_j):
        # """exp
        two_body_jastrow = (
            1.0
            / (2.0 * jastrow_two_body_data.jastrow_2b_param)
            * (
                1.0
                - jnp.exp(
                    -jastrow_two_body_data.jastrow_2b_param * jnp.linalg.norm(r_cart_i - r_cart_j)
                )
            )
        )
        return two_body_jastrow
        # """

        """pade
        two_body_jastrow = (
            jnp.linalg.norm(r_cart_i - r_cart_j)
            / 2.0
            * (
                1.0
                + jastrow_two_body_data.param_anti_parallel_spin
                * jnp.linalg.norm(r_cart_i - r_cart_j)
            )
            ** (-1.0)
        )
        return two_body_jastrow
        """

    def J2_parallel_spins(r_cart_i, r_cart_j):
        # """exp
        two_body_jastrow = (
            1.0
            / (2.0 * jastrow_two_body_data.jastrow_2b_param)
            * (
                1.0
                - jnp.exp(
                    -jastrow_two_body_data.jastrow_2b_param * jnp.linalg.norm(r_cart_i - r_cart_j)
                )
            )
        )
        return two_body_jastrow
        # """

        """pade
        two_body_jastrow = (
            jnp.linalg.norm(r_cart_i - r_cart_j)
            / 4.0
            * (
                1.0
                + jastrow_two_body_data.param_parallel_spin * jnp.linalg.norm(r_cart_i - r_cart_j)
            )
            ** (-1.0)
        )
        return two_body_jastrow
        """

    vmap_two_body_jastrow_anti_parallel_spins = vmap(
        vmap(J2_anti_parallel_spins, in_axes=(None, 0)), in_axes=(0, None)
    )

    two_body_jastrow_anti_parallel = jnp.sum(
        vmap_two_body_jastrow_anti_parallel_spins(r_up_carts, r_dn_carts)
    )

    def compute_parallel_sum(r_carts):
        num_particles = r_carts.shape[0]
        idx_i, idx_j = jnp.triu_indices(num_particles, k=1)
        r_i = r_carts[idx_i]
        r_j = r_carts[idx_j]
        vmap_two_body_jastrow_parallel_spins = vmap(J2_parallel_spins)(r_i, r_j)
        return jnp.sum(vmap_two_body_jastrow_parallel_spins)

    two_body_jastrow_parallel_up = compute_parallel_sum(r_up_carts)
    two_body_jastrow_parallel_dn = compute_parallel_sum(r_dn_carts)

    two_body_jastrow = (
        two_body_jastrow_anti_parallel + two_body_jastrow_parallel_up + two_body_jastrow_parallel_dn
    )

    return two_body_jastrow


def compute_grads_and_laplacian_Jastrow_two_body_api(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64 | np.complex128],
    npt.NDArray[np.float64 | np.complex128],
    float | complex,
]:
    """
    The method is for computing the gradients and the sum of laplacians of J at (r_up_carts, r_dn_carts).

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
    grad_J2_up, grad_J2_dn, sum_laplacian_J2 = compute_grads_and_laplacian_Jastrow_two_body_jax(
        jastrow_two_body_data, r_up_carts, r_dn_carts
    )

    if grad_J2_up.shape != r_up_carts.shape:
        logger.error(
            f"grad_J2_up.shape = {grad_J2_up.shape} is inconsistent with the expected one = {r_up_carts.shape}"
        )
        raise ValueError

    if grad_J2_dn.shape != r_dn_carts.shape:
        logger.error(
            f"grad_J2_dn.shape = {grad_J2_dn.shape} is inconsistent with the expected one = {r_dn_carts.shape}"
        )
        raise ValueError

    return grad_J2_up, grad_J2_dn, sum_laplacian_J2


def compute_grads_and_laplacian_Jastrow_two_body_debug(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64 | np.complex128],
    npt.NDArray[np.float64 | np.complex128],
    float | complex,
]:
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

        J2_p_x_up = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_x_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_p_y_up = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_y_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_p_z_up = compute_Jastrow_two_body_api(
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

        J2_m_x_up = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_x_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_m_y_up = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_y_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_m_z_up = compute_Jastrow_two_body_api(
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

        J2_p_x_dn = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_x_r_dn_carts,
        )
        J2_p_y_dn = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_y_r_dn_carts,
        )
        J2_p_z_dn = compute_Jastrow_two_body_api(
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

        J2_m_x_dn = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_x_r_dn_carts,
        )
        J2_m_y_dn = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_y_r_dn_carts,
        )
        J2_m_z_dn = compute_Jastrow_two_body_api(
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

    J2_ref = compute_Jastrow_two_body_api(
        jastrow_two_body_data=jastrow_two_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    sum_laplacian_J2 = 0.0

    # laplacians up
    for r_i, _ in enumerate(r_up_carts):
        diff_p_x_r_up2_carts = r_up_carts.copy()
        diff_p_y_r_up2_carts = r_up_carts.copy()
        diff_p_z_r_up2_carts = r_up_carts.copy()
        diff_p_x_r_up2_carts[r_i][0] += diff_h2
        diff_p_y_r_up2_carts[r_i][1] += diff_h2
        diff_p_z_r_up2_carts[r_i][2] += diff_h2

        J2_p_x_up2 = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_x_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_p_y_up2 = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_y_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )

        J2_p_z_up2 = compute_Jastrow_two_body_api(
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

        J2_m_x_up2 = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_x_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_m_y_up2 = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_y_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_m_z_up2 = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_z_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )

        gradgrad_x_up = (J2_p_x_up2 + J2_m_x_up2 - 2 * J2_ref) / (diff_h2**2)
        gradgrad_y_up = (J2_p_y_up2 + J2_m_y_up2 - 2 * J2_ref) / (diff_h2**2)
        gradgrad_z_up = (J2_p_z_up2 + J2_m_z_up2 - 2 * J2_ref) / (diff_h2**2)

        sum_laplacian_J2 += gradgrad_x_up + gradgrad_y_up + gradgrad_z_up

    # laplacians dn
    for r_i, _ in enumerate(r_dn_carts):
        diff_p_x_r_dn2_carts = r_dn_carts.copy()
        diff_p_y_r_dn2_carts = r_dn_carts.copy()
        diff_p_z_r_dn2_carts = r_dn_carts.copy()
        diff_p_x_r_dn2_carts[r_i][0] += diff_h2
        diff_p_y_r_dn2_carts[r_i][1] += diff_h2
        diff_p_z_r_dn2_carts[r_i][2] += diff_h2

        J2_p_x_dn2 = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_x_r_dn2_carts,
        )
        J2_p_y_dn2 = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_y_r_dn2_carts,
        )

        J2_p_z_dn2 = compute_Jastrow_two_body_api(
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

        J2_m_x_dn2 = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_x_r_dn2_carts,
        )
        J2_m_y_dn2 = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_y_r_dn2_carts,
        )
        J2_m_z_dn2 = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_z_r_dn2_carts,
        )

        gradgrad_x_dn = (J2_p_x_dn2 + J2_m_x_dn2 - 2 * J2_ref) / (diff_h2**2)
        gradgrad_y_dn = (J2_p_y_dn2 + J2_m_y_dn2 - 2 * J2_ref) / (diff_h2**2)
        gradgrad_z_dn = (J2_p_z_dn2 + J2_m_z_dn2 - 2 * J2_ref) / (diff_h2**2)

        sum_laplacian_J2 += gradgrad_x_dn + gradgrad_y_dn + gradgrad_z_dn

    return grad_J2_up, grad_J2_dn, sum_laplacian_J2


@jit
def compute_grads_and_laplacian_Jastrow_two_body_jax(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: npt.NDArray[jnp.float64],
    r_dn_carts: npt.NDArray[jnp.float64],
) -> tuple[
    npt.NDArray[jnp.float64 | jnp.complex128],
    npt.NDArray[jnp.float64 | jnp.complex128],
    float | complex,
]:
    # compute grad
    grad_J2_up = grad(compute_Jastrow_two_body_jax, argnums=1)(
        jastrow_two_body_data, r_up_carts, r_dn_carts
    )

    grad_J2_dn = grad(compute_Jastrow_two_body_jax, argnums=2)(
        jastrow_two_body_data, r_up_carts, r_dn_carts
    )

    # compute laplacians
    hessian_J2_up = hessian(compute_Jastrow_two_body_jax, argnums=1)(
        jastrow_two_body_data, r_up_carts, r_dn_carts
    )
    sum_laplacian_J2_up = jnp.einsum("ijij->", hessian_J2_up)

    hessian_J2_dn = hessian(compute_Jastrow_two_body_jax, argnums=2)(
        jastrow_two_body_data, r_up_carts, r_dn_carts
    )
    sum_laplacian_J2_dn = jnp.einsum("ijij->", hessian_J2_dn)

    sum_laplacian_J2 = sum_laplacian_J2_up + sum_laplacian_J2_dn

    return grad_J2_up, grad_J2_dn, sum_laplacian_J2


def compute_grads_and_laplacian_Jastrow_three_body_api(
    jastrow_three_body_data: Jastrow_three_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64 | np.complex128],
    npt.NDArray[np.float64 | np.complex128],
    float | complex,
]:
    """
    The method is for computing the gradients and the sum of laplacians of J3 at (r_up_carts, r_dn_carts).

    Args:
        jastrow_three_body_data (Jastrow_three_body_data): an instance of Jastrow_three_body_data class
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
    Returns:
        the gradients(x,y,z) of J(threebody) and the sum of laplacians of J(threebody) at (r_up_carts, r_dn_carts).
    """
    grad_J3_up, grad_J3_dn, sum_laplacian_J3 = compute_grads_and_laplacian_Jastrow_three_body_jax(
        jastrow_three_body_data, r_up_carts, r_dn_carts
    )

    if grad_J3_up.shape != r_up_carts.shape:
        logger.error(
            f"grad_J3_up.shape = {grad_J3_up.shape} is inconsistent with the expected one = {r_up_carts.shape}"
        )
        raise ValueError

    if grad_J3_dn.shape != r_dn_carts.shape:
        logger.error(
            f"grad_J3_dn.shape = {grad_J3_dn.shape} is inconsistent with the expected one = {r_dn_carts.shape}"
        )
        raise ValueError

    return grad_J3_up, grad_J3_dn, sum_laplacian_J3


def compute_grads_and_laplacian_Jastrow_three_body_debug(
    jastrow_three_body_data: Jastrow_three_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64 | np.complex128],
    npt.NDArray[np.float64 | np.complex128],
    float | complex,
]:
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

        J3_p_x_up = compute_Jastrow_three_body_api(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_p_x_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_p_y_up = compute_Jastrow_three_body_api(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_p_y_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_p_z_up = compute_Jastrow_three_body_api(
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

        J3_m_x_up = compute_Jastrow_three_body_api(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_m_x_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_m_y_up = compute_Jastrow_three_body_api(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_m_y_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_m_z_up = compute_Jastrow_three_body_api(
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

        J3_p_x_dn = compute_Jastrow_three_body_api(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_x_r_dn_carts,
        )
        J3_p_y_dn = compute_Jastrow_three_body_api(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_y_r_dn_carts,
        )
        J3_p_z_dn = compute_Jastrow_three_body_api(
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

        J3_m_x_dn = compute_Jastrow_three_body_api(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_x_r_dn_carts,
        )
        J3_m_y_dn = compute_Jastrow_three_body_api(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_y_r_dn_carts,
        )
        J3_m_z_dn = compute_Jastrow_three_body_api(
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

    J3_ref = compute_Jastrow_three_body_api(
        jastrow_three_body_data=jastrow_three_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    sum_laplacian_J3 = 0.0

    # laplacians up
    for r_i, _ in enumerate(r_up_carts):
        diff_p_x_r_up2_carts = r_up_carts.copy()
        diff_p_y_r_up2_carts = r_up_carts.copy()
        diff_p_z_r_up2_carts = r_up_carts.copy()
        diff_p_x_r_up2_carts[r_i][0] += diff_h2
        diff_p_y_r_up2_carts[r_i][1] += diff_h2
        diff_p_z_r_up2_carts[r_i][2] += diff_h2

        J3_p_x_up2 = compute_Jastrow_three_body_api(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_p_x_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_p_y_up2 = compute_Jastrow_three_body_api(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_p_y_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )

        J3_p_z_up2 = compute_Jastrow_three_body_api(
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

        J3_m_x_up2 = compute_Jastrow_three_body_api(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_m_x_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_m_y_up2 = compute_Jastrow_three_body_api(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_m_y_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_m_z_up2 = compute_Jastrow_three_body_api(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_m_z_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )

        gradgrad_x_up = (J3_p_x_up2 + J3_m_x_up2 - 2 * J3_ref) / (diff_h2**2)
        gradgrad_y_up = (J3_p_y_up2 + J3_m_y_up2 - 2 * J3_ref) / (diff_h2**2)
        gradgrad_z_up = (J3_p_z_up2 + J3_m_z_up2 - 2 * J3_ref) / (diff_h2**2)

        sum_laplacian_J3 += gradgrad_x_up + gradgrad_y_up + gradgrad_z_up

    # laplacians dn
    for r_i, _ in enumerate(r_dn_carts):
        diff_p_x_r_dn2_carts = r_dn_carts.copy()
        diff_p_y_r_dn2_carts = r_dn_carts.copy()
        diff_p_z_r_dn2_carts = r_dn_carts.copy()
        diff_p_x_r_dn2_carts[r_i][0] += diff_h2
        diff_p_y_r_dn2_carts[r_i][1] += diff_h2
        diff_p_z_r_dn2_carts[r_i][2] += diff_h2

        J3_p_x_dn2 = compute_Jastrow_three_body_api(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_x_r_dn2_carts,
        )
        J3_p_y_dn2 = compute_Jastrow_three_body_api(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_y_r_dn2_carts,
        )

        J3_p_z_dn2 = compute_Jastrow_three_body_api(
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

        J3_m_x_dn2 = compute_Jastrow_three_body_api(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_x_r_dn2_carts,
        )
        J3_m_y_dn2 = compute_Jastrow_three_body_api(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_y_r_dn2_carts,
        )
        J3_m_z_dn2 = compute_Jastrow_three_body_api(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_z_r_dn2_carts,
        )

        gradgrad_x_dn = (J3_p_x_dn2 + J3_m_x_dn2 - 2 * J3_ref) / (diff_h2**2)
        gradgrad_y_dn = (J3_p_y_dn2 + J3_m_y_dn2 - 2 * J3_ref) / (diff_h2**2)
        gradgrad_z_dn = (J3_p_z_dn2 + J3_m_z_dn2 - 2 * J3_ref) / (diff_h2**2)

        sum_laplacian_J3 += gradgrad_x_dn + gradgrad_y_dn + gradgrad_z_dn

    return grad_J3_up, grad_J3_dn, sum_laplacian_J3


@jit
def compute_grads_and_laplacian_Jastrow_three_body_jax(
    jastrow_three_body_data: Jastrow_two_body_data,
    r_up_carts: npt.NDArray[jnp.float64],
    r_dn_carts: npt.NDArray[jnp.float64],
) -> tuple[
    npt.NDArray[jnp.float64 | jnp.complex128],
    npt.NDArray[jnp.float64 | jnp.complex128],
    float | complex,
]:
    # compute grad
    grad_J3_up = grad(compute_Jastrow_three_body_jax, argnums=1)(
        jastrow_three_body_data, r_up_carts, r_dn_carts
    )

    grad_J3_dn = grad(compute_Jastrow_three_body_jax, argnums=2)(
        jastrow_three_body_data, r_up_carts, r_dn_carts
    )

    # compute laplacians
    hessian_J3_up = hessian(compute_Jastrow_three_body_jax, argnums=1)(
        jastrow_three_body_data, r_up_carts, r_dn_carts
    )
    sum_laplacian_J3_up = jnp.einsum("ijij->", hessian_J3_up)

    hessian_J3_dn = hessian(compute_Jastrow_three_body_jax, argnums=2)(
        jastrow_three_body_data, r_up_carts, r_dn_carts
    )
    sum_laplacian_J3_dn = jnp.einsum("ijij->", hessian_J3_dn)

    sum_laplacian_J3 = sum_laplacian_J3_up + sum_laplacian_J3_dn

    return grad_J3_up, grad_J3_dn, sum_laplacian_J3


if __name__ == "__main__":
    log = getLogger("jqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    num_r_up_cart_samples = 5
    num_r_dn_cart_samples = 2

    r_cart_min, r_cart_max = -3.0, 3.0

    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_up_cart_samples, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_dn_cart_samples, 3) + r_cart_min

    jastrow_two_body_data = Jastrow_two_body_data(
        param_anti_parallel_spin=1.0, jastrow_2b_param=1.0
    )
    jastrow_two_body_debug = compute_Jastrow_two_body_debug(
        jastrow_two_body_data=jastrow_two_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # logger.debug(f"jastrow_two_body_debug = {jastrow_two_body_debug}")

    jastrow_two_body_jax = compute_Jastrow_two_body_jax(
        jastrow_two_body_data=jastrow_two_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # logger.debug(f"jastrow_two_body_jax = {jastrow_two_body_jax}")

    np.testing.assert_almost_equal(jastrow_two_body_debug, jastrow_two_body_jax, decimal=10)

    (
        grad_jastrow_J2_up_debug,
        grad_jastrow_J2_dn_debug,
        sum_laplacian_J2_debug,
    ) = compute_grads_and_laplacian_Jastrow_two_body_debug(
        jastrow_two_body_data,
        r_up_carts,
        r_dn_carts,
        True,
    )

    # logger.debug(f"grad_jastrow_J2_up_debug = {grad_jastrow_J2_up_debug}")
    # logger.debug(f"grad_jastrow_J2_dn_debug = {grad_jastrow_J2_dn_debug}")
    # logger.debug(f"sum_laplacian_J2_debug = {sum_laplacian_J2_debug}")

    grad_jastrow_J2_up_jax, grad_jastrow_J2_dn_jax, sum_laplacian_J2_jax = (
        compute_grads_and_laplacian_Jastrow_two_body_jax(
            jastrow_two_body_data,
            r_up_carts,
            r_dn_carts,
            False,
        )
    )

    # logger.debug(f"grad_jastrow_J2_up_jax = {grad_jastrow_J2_up_jax}")
    # logger.debug(f"grad_jastrow_J2_dn_jax = {grad_jastrow_J2_dn_jax}")
    # logger.debug(f"sum_laplacian_J2_jax = {sum_laplacian_J2_jax}")

    np.testing.assert_almost_equal(grad_jastrow_J2_up_debug, grad_jastrow_J2_up_jax, decimal=5)
    np.testing.assert_almost_equal(grad_jastrow_J2_dn_debug, grad_jastrow_J2_dn_jax, decimal=5)
    np.testing.assert_almost_equal(sum_laplacian_J2_debug, sum_laplacian_J2_jax, decimal=5)

    # test MOs
    num_r_up_cart_samples = 4
    num_r_dn_cart_samples = 2
    num_R_cart_samples = 6
    num_ao = 6
    num_ao_prim = 6
    orbital_indices = [0, 1, 2, 3, 4, 5]
    exponents = [1.2, 0.5, 0.1, 0.05, 0.05, 0.05]
    coefficients = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    angular_momentums = [0, 0, 0, 1, 1, 1]
    magnetic_quantum_numbers = [0, 0, 0, 0, +1, -1]

    # generate matrices for the test
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_up_cart_samples, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_dn_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    aos_up_data = AOs_data_debug(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    aos_dn_data = AOs_data_debug(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    j_matrix_up_up = np.random.rand(aos_up_data.num_ao, aos_up_data.num_ao + 1)
    j_matrix_dn_dn = np.random.rand(aos_dn_data.num_ao, aos_dn_data.num_ao + 1)
    j_matrix_up_dn = np.random.rand(aos_up_data.num_ao, aos_dn_data.num_ao)

    jastrow_three_body_data = Jastrow_three_body_data(
        orb_data_up_spin=aos_up_data,
        orb_data_dn_spin=aos_dn_data,
        j_matrix_up_up=j_matrix_up_up,
        j_matrix_dn_dn=j_matrix_dn_dn,
        j_matrix_up_dn=j_matrix_up_dn,
    )

    J3_debug = compute_Jastrow_three_body_debug(
        jastrow_three_body_data=jastrow_three_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # logger.debug(f"J3_debug = {J3_debug}")

    J3_jax = compute_Jastrow_three_body_jax(
        jastrow_three_body_data=jastrow_three_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # logger.debug(f"J3_jax = {J3_jax}")

    np.testing.assert_almost_equal(J3_debug, J3_jax, decimal=8)

    (
        grad_jastrow_J3_up_debug,
        grad_jastrow_J3_dn_debug,
        sum_laplacian_J3_debug,
    ) = compute_grads_and_laplacian_Jastrow_three_body_debug(
        jastrow_three_body_data,
        r_up_carts,
        r_dn_carts,
    )

    # logger.debug(f"grad_jastrow_J3_up_debug = {grad_jastrow_J3_up_debug}")
    # logger.debug(f"grad_jastrow_J3_dn_debug = {grad_jastrow_J3_dn_debug}")
    # logger.debug(f"sum_laplacian_J3_debug = {sum_laplacian_J3_debug}")

    grad_jastrow_J3_up_jax, grad_jastrow_J3_dn_jax, sum_laplacian_J3_jax = (
        compute_grads_and_laplacian_Jastrow_three_body_jax(
            jastrow_three_body_data,
            r_up_carts,
            r_dn_carts,
        )
    )

    # logger.debug(f"grad_jastrow_J3_up_jax = {grad_jastrow_J3_up_jax}")
    # logger.debug(f"grad_jastrow_J3_dn_jax = {grad_jastrow_J3_dn_jax}")
    # logger.debug(f"sum_laplacian_J3_jax = {sum_laplacian_J3_jax}")

    np.testing.assert_almost_equal(grad_jastrow_J3_up_debug, grad_jastrow_J3_up_jax, decimal=5)
    np.testing.assert_almost_equal(grad_jastrow_J3_dn_debug, grad_jastrow_J3_dn_jax, decimal=5)
    np.testing.assert_almost_equal(sum_laplacian_J3_debug, sum_laplacian_J3_jax, decimal=5)
