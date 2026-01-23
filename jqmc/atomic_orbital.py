"""Atomic Orbitals module.

Module containing classes and methods related to Atomic Orbitals

"""

from __future__ import annotations

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
from logging import getLogger

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
import numpy.typing as npt
import scipy
from flax import struct
from jax import hessian, jacrev, jit, vmap
from jax import typing as jnpt
from numpy import linalg as LA

from .setting import EPS_stabilizing_jax_AO_cart_deriv
from .structure import Structure_data

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)

# Tolerances for comparing float values
rtol = 1e-6
atol = 1e-8


@struct.dataclass
class AOs_cart_data:
    """Atomic orbital definitions in Cartesian form.

    Stores contracted Gaussian basis data used to evaluate atomic orbitals (AOs) on a grid.
    The angular part is represented by Cartesian polynomials (:math:`x^{n_x} y^{n_y} z^{n_z}`) with
    :math:`n_x + n_y + n_z = l` for each AO.

    Attributes:
        structure_data (Structure_data): Molecular structure and atomic positions used to place AOs.
        nucleus_index (list[int] | tuple[int]): AO -> atom index mapping (``len == num_ao``).
        num_ao (int): Number of contracted AOs.
        num_ao_prim (int): Number of primitive Gaussians.
        orbital_indices (list[int] | tuple[int]): For each primitive, the parent AO index (``len == num_ao_prim``).
        exponents (list[float] | tuple[float]): Gaussian exponents for primitives (``len == num_ao_prim``).
        coefficients (list[float] | tuple[float]): Contraction coefficients per primitive (``len == num_ao_prim``).
        angular_momentums (list[int] | tuple[int]): Angular momentum quantum numbers ``l`` per AO (``len == num_ao``).
        polynominal_order_x (list[int] | tuple[int]): Cartesian power ``n_x`` for each AO (``len == num_ao``).
        polynominal_order_y (list[int] | tuple[int]): Cartesian power ``n_y`` for each AO (``len == num_ao``).
        polynominal_order_z (list[int] | tuple[int]): Cartesian power ``n_z`` for each AO (``len == num_ao``).

    Examples:
        Minimal hydrogen dimer (bohr) with all-electron cc-pVTZ (Gaussian format) in Cartesian form::

            from jqmc.structure import Structure_data
            from jqmc.atomic_orbital import AOs_cart_data

            structure = Structure_data(
                positions=[[0.0, 0.0, -0.70], [0.0, 0.0, 0.70]],
                pbc_flag=False,
                atomic_numbers=[1, 1],
                element_symbols=["H", "H"],
                atomic_labels=["H1", "H2"],
            )

            # cc-pVTZ primitives duplicated per Cartesian component; counts:
            # per atom -> 15 AOs, 19 primitives; for two atoms
            num_ao=30; num_ao_prim=38

            exponents = [
                0.3258,
                33.87, 5.095, 1.159, 0.3258, 0.1027,
                0.1027,
                1.407, 1.407, 1.407,
                0.388, 0.388, 0.388,
                1.057, 1.057, 1.057, 1.057, 1.057, 1.057,
                0.3258,
                33.87, 5.095, 1.159, 0.3258, 0.1027,
                0.1027,
                1.407, 1.407, 1.407,
                0.388, 0.388, 0.388,
                1.057, 1.057, 1.057, 1.057, 1.057, 1.057,
            ]

            coefficients = [
                1.0,
                0.006068, 0.045308, 0.202822, 0.503903, 0.383421,
                1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0,
                0.006068, 0.045308, 0.202822, 0.503903, 0.383421,
                1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ]

            orbital_indices = [
                0,                      # S shell (uncontracted)
                1, 1, 1, 1, 1,          # S shell (contracted)
                2,                      # S shell (uncontracted)
                3, 4, 5,                # P shell (uncontracted)
                6, 7, 8,                # P shell (uncontracted)
                9, 10, 11, 12, 13, 14,  # D shell (uncontracted)
                15,                     # S shell (uncontracted)
                16, 16, 16, 16, 16,     # S shell (contracted)
                17,                     # S shell (uncontracted)
                18, 19, 20,             # P shell (uncontracted)
                21, 22, 23,             # P shell (uncontracted)
                24, 25, 26, 27, 28, 29, # D shell (uncontracted)
            ]

            nucleus_index = [
                *([0] * 15),
                *([1] * 15),
            ]

            angular_momentums = [
                0, 0, 0,                # S shells (1 orbital each)
                1, 1, 1, 1, 1, 1,       # two P shells (3 orbitals each)
                2, 2, 2, 2, 2, 2,       # one D shell (6 orbitals each)
                0, 0, 0,                # S shells (1 orbital each)
                1, 1, 1, 1, 1, 1,       # two P shells (3 orbitals each)
                2, 2, 2, 2, 2, 2,       # one D shell (6 orbitals each)
            ]

            polynominal_order_x = [
                0, 0, 0,
                1, 0, 0,
                1, 0, 0,
                2, 1, 1, 0, 0, 0,
                0, 0, 0,
                1, 0, 0,
                1, 0, 0,
                2, 1, 1, 0, 0, 0,
            ]

            polynominal_order_y = [
                0, 0, 0,
                0, 1, 0,
                0, 1, 0,
                0, 1, 0, 2, 1, 0,
                0, 0, 0,
                0, 1, 0,
                0, 1, 0,
                0, 1, 0, 2, 1, 0,
            ]

            polynominal_order_z = [
                0, 0, 0,
                0, 0, 1,
                0, 0, 1,
                0, 0, 1, 0, 1, 2,
                0, 0, 0,
                0, 0, 1,
                0, 0, 1,
                0, 0, 1, 0, 1, 2,
            ]

            aos = AOs_cart_data(
                structure_data=structure,
                nucleus_index=nucleus_index,
                num_ao=num_ao,
                num_ao_prim=num_ao_prim,
                orbital_indices=orbital_indices,
                exponents=exponents,
                coefficients=coefficients,
                angular_momentums=angular_momentums,
                polynominal_order_x=polynominal_order_x,
                polynominal_order_y=polynominal_order_y,
                polynominal_order_z=polynominal_order_z,
            )

            aos.sanity_check()
    """

    #: Molecular structure and atomic positions used to place AOs.
    structure_data: Structure_data = struct.field(pytree_node=True, default_factory=lambda: Structure_data())
    #: AO -> atom index mapping (``len == num_ao``).
    nucleus_index: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)
    #: Number of contracted AOs.
    num_ao: int = struct.field(pytree_node=False, default=0)
    #: Number of primitive Gaussians.
    num_ao_prim: int = struct.field(pytree_node=False, default=0)
    #: For each primitive, the parent AO index (``len == num_ao_prim``).
    orbital_indices: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)
    #: Gaussian exponents for primitives (``len == num_ao_prim``).
    exponents: list[float] | tuple[float] = struct.field(pytree_node=False, default_factory=tuple)
    #: Contraction coefficients per primitive (``len == num_ao_prim``).
    coefficients: list[float] | tuple[float] = struct.field(pytree_node=False, default_factory=tuple)
    #: Angular momentum quantum numbers ``l`` per AO (``len == num_ao``).
    angular_momentums: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)
    #: Cartesian power ``n_x`` for each AO (``len == num_ao``).
    polynominal_order_x: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)
    #: Cartesian power ``n_y`` for each AO (``len == num_ao``).
    polynominal_order_y: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)
    #: Cartesian power ``n_z`` for each AO (``len == num_ao``).
    polynominal_order_z: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)

    def sanity_check(self) -> None:
        """Validate AO array shapes and basic types.

        Ensures that array lengths match declared counts (``num_ao``, ``num_ao_prim``) and
        that inputs are provided as lists/tuples (ints for counts). Call this after constructing
        ``AOs_cart_data`` and before AO evaluation routines.

        Raises:
            ValueError: When any of the following holds:
                - ``len(nucleus_index) != num_ao``
                - ``len(unique(orbital_indices)) != num_ao``
                - ``len(exponents) != num_ao_prim`` or ``len(coefficients) != num_ao_prim``
                - ``len(angular_momentums) != num_ao``
                - ``len(polynominal_order_{x,y,z}) != num_ao``
                - any attribute has an unexpected Python type (e.g., non-list/tuple arrays, non-int counts)
        """
        if len(self.nucleus_index) != self.num_ao:
            raise ValueError("dim. of self.nucleus_index is wrong")
        if len(np.unique(self.orbital_indices)) != self.num_ao:
            raise ValueError(f"num_ao={self.num_ao} and/or num_ao_prim={self.num_ao_prim} is wrong")
        if len(self.exponents) != self.num_ao_prim:
            raise ValueError("dim. of self.exponents is wrong")
        if len(self.coefficients) != self.num_ao_prim:
            raise ValueError("dim. of self.coefficients is wrong")
        if len(self.angular_momentums) != self.num_ao:
            raise ValueError("dim. of self.angular_momentums is wrong")
        if len(self.polynominal_order_x) != self.num_ao:
            raise ValueError("dim. of self.polynominal_order_x is wrong")
        if len(self.polynominal_order_y) != self.num_ao:
            raise ValueError("dim. of self.polynominal_order_y is wrong")
        if len(self.polynominal_order_z) != self.num_ao:
            raise ValueError("dim. of self.polynominal_order_z is wrong")

        # Post-initialization method to check the types of the attributes.
        # tuple is very slow in practice!! So, we must use list for production runs!!

        if not isinstance(self.nucleus_index, (tuple, list)):
            raise ValueError(f"nucleus_index = {type(self.nucleus_index)} must be a list or tuple.")
        if not isinstance(self.num_ao, (int, np.integer)):
            raise ValueError(f"num_ao = {type(self.num_ao)} must be an int.")
        if not isinstance(self.num_ao_prim, (int, np.integer)):
            raise ValueError(f"num_ao_prim = {type(self.num_ao_prim)} must be an int.")
        if not isinstance(self.orbital_indices, (tuple, list)):
            raise ValueError(f"orbital_indices = {type(self.orbital_indices)} must be a list or tuple.")
        if not isinstance(self.exponents, (tuple, list)):
            raise ValueError(f"exponents = {type(self.exponents)} must be a tuple.")
        if not isinstance(self.coefficients, (tuple, list)):
            raise ValueError(f"coefficients = {type(self.coefficients)} must be a list or tuple.")
        if not isinstance(self.angular_momentums, (tuple, list)):
            raise ValueError(f"angular_momentums = {type(self.angular_momentums)} must be a list or tuple.")
        if not isinstance(self.polynominal_order_x, (tuple, list)):
            raise ValueError(f"polynominal_order_x = {type(self.polynominal_order_x)} must be a list or tuple.")
        if not isinstance(self.polynominal_order_y, (tuple, list)):
            raise ValueError(f"polynominal_order_y = {type(self.polynominal_order_y)} must be a list or tuple.")
        if not isinstance(self.polynominal_order_z, (tuple, list)):
            raise ValueError(f"polynominal_order_z = {type(self.polynominal_order_z)} must be a list or tuple.")

        """ It works for practical cases, but it is not good for the test cases!!
        # Assert that, for each nucleus_index:
        # 1) primitives are clustered by (exp, coef, l) within tol,
        # 2) each cluster contains (l+2)(l+1)/2 primitives,
        # 3) all combinations of nx+ny+nz = l are present.
        primitive_info = []
        for prim_idx, ao_idx in enumerate(self.orbital_indices):
            # validate ao_idx range
            if not (0 <= ao_idx < self.num_ao):
                logger.error(f"Primitive {prim_idx}: AO index {ao_idx} out of range [0, {self.num_ao})")
                raise ValueError
            Z = self.exponents[prim_idx]
            coeff = self.coefficients[prim_idx]
            l = self.angular_momentums[ao_idx]
            nx = self.polynominal_order_x[ao_idx]
            ny = self.polynominal_order_y[ao_idx]
            nz = self.polynominal_order_z[ao_idx]

            # Consider the normalization factor
            fact_term = (scipy.special.factorial(nx) * scipy.special.factorial(ny) * scipy.special.factorial(nz)) / (
                scipy.special.factorial(2 * nx) * scipy.special.factorial(2 * ny) * scipy.special.factorial(2 * nz)
            )
            z_term = (2.0 * Z / np.pi) ** (3.0 / 2.0) * (8.0 * Z) ** l
            Norm = np.sqrt(fact_term * z_term)  # both are ok, but it is better to use the one which is used below (get_info()).
            Norm = np.sqrt(fact_term)  # both are ok, but it is better to use the one which is used below (get_info()).
            coeff = coeff * Norm

            info = {
                "prim_index": prim_idx,
                "ao_index": ao_idx,
                "exponent": Z,
                "coefficient": coeff,
                "l": l,
                "nx": nx,
                "ny": ny,
                "nz": nz,
            }
            primitive_info.append(info)

        # 1) Attach nucleus index to each info entry
        for info in primitive_info:
            ao_idx = info["ao_index"]
            info["nucleus"] = self.nucleus_index[ao_idx]

        # 2) Process primitives for each nucleus
        for nucleus in set(self.nucleus_index):
            infos_nuc = [info for info in primitive_info if info["nucleus"] == nucleus]
            if not infos_nuc:
                continue  # skip if no primitives for this nucleus

            # --- Clustering based on (exp, coef, l) within tolerance ---
            # each entry in clusters is [cluster_exp, cluster_coef, l, [infos list]]
            clusters = []
            for info in infos_nuc:
                exp, coef, l = info["exponent"], info["coefficient"], info["l"]
                # search for a matching existing cluster
                for c_exp, c_coef, c_l, c_infos in clusters:
                    if (
                        c_l == l
                        and np.isclose(exp, c_exp, atol=atol, rtol=rtol)
                        and np.isclose(coef, c_coef, atol=atol, rtol=rtol)
                    ):
                        c_infos.append(info)
                        break
                else:
                    # create a new cluster
                    clusters.append([exp, coef, l, [info]])

            # --- Check each cluster ---
            for c_exp, c_coef, l, c_infos in clusters:
                expected_count = (l + 2) * (l + 1) // 2
                actual_coords = {(i["nx"], i["ny"], i["nz"]) for i in c_infos}
                expected_coords = {(nx, ny, l - nx - ny) for nx in range(l + 1) for ny in range(l + 1 - nx)}

                # 3.1 Count check
                if len(c_infos) != expected_count:
                    logger.error(
                        f"[nucleus={nucleus}] "
                        f"(exp={c_exp:.5g}, coef={c_coef:.5g}, l={l}): "
                        f"found {len(c_infos)}, expected {expected_count}"
                    )
                    raise ValueError

                # 3.2 Coverage check
                missing = expected_coords - actual_coords
                extra = actual_coords - expected_coords
                if len(missing) != 0 or len(extra) != 0:
                    logger.error(
                        f"[nucleus={nucleus}] "
                        f"(exp={c_exp:.5g}, coef={c_coef:.5g}, l={l}):\n"
                        f"  missing combos:   {missing}\n"
                        f"  unexpected combos:{extra}"
                    )
                    raise ValueError
        """
        self.structure_data.sanity_check()

    def _get_info(self) -> list[str]:
        """Return a list of strings containing information about the class attributes."""
        info_lines = []
        info_lines.append(f"**{self.__class__.__name__}**")
        info_lines.append(f"  Number of AOs = {self.num_ao}")
        info_lines.append(f"  Number of primitive AOs = {self.num_ao_prim}")
        info_lines.append("  Angular part is the polynomial (Cartesian) function.")

        # Map angular momentum quantum number to NWChem shell label
        l_map = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h", 6: "i"}

        # Build mapping from AO index to its list of primitive indices
        prim_per_ao = {}
        for prim_idx, ao_idx in enumerate(self.orbital_indices):
            prim_per_ao.setdefault(ao_idx, []).append(prim_idx)

        # Build mapping from atom index to its list of AO indices
        ao_per_atom = {}
        for ao_idx, atom_idx in enumerate(self.nucleus_index):
            ao_per_atom.setdefault(atom_idx, []).append(ao_idx)

        # Loop over atoms in sorted order
        for atom_idx in sorted(ao_per_atom):
            symbol = self.structure_data.atomic_labels[atom_idx]
            info_lines.append("  " + "-" * 36)
            info_lines.append(f"  **basis set for atom index {atom_idx + 1}: {symbol}**")
            info_lines.append("  " + "-" * 36)

            # Collect unique shells with approximate comparison
            shell_groups = []

            for ao_idx in ao_per_atom[atom_idx]:
                prim_idxs = prim_per_ao.get(ao_idx, [])
                nx = self.polynominal_order_x[ao_idx]
                ny = self.polynominal_order_y[ao_idx]
                nz = self.polynominal_order_z[ao_idx]
                l = self.angular_momentums[ao_idx]

                # Recover original coefficients and build (exp, coef) pairs
                ec_pairs = []
                for prim_idx in prim_idxs:
                    Z = self.exponents[prim_idx]
                    stored_coef = self.coefficients[prim_idx]
                    # Consider the normalization factor
                    fact_term = (scipy.special.factorial(nx) * scipy.special.factorial(ny) * scipy.special.factorial(nz)) / (
                        scipy.special.factorial(2 * nx) * scipy.special.factorial(2 * ny) * scipy.special.factorial(2 * nz)
                    )
                    z_term = (2.0 * Z / np.pi) ** (3.0 / 2.0) * (8.0 * Z) ** l
                    Norm = np.sqrt(fact_term * z_term)  # which is better for its output?? Todo.
                    Norm = np.sqrt(fact_term)  # which is better for its output?? Todo.
                    orig_coef = stored_coef * Norm
                    ec_pairs.append((Z, orig_coef))

                # Attempt to match existing group within tolerance
                matched = False
                for existing_ec, _ in shell_groups:
                    if len(existing_ec) == len(ec_pairs):
                        exps1, coefs1 = zip(*existing_ec, strict=True)
                        exps2, coefs2 = zip(*ec_pairs, strict=True)
                        if np.allclose(exps1, exps2, rtol=rtol, atol=atol) and np.allclose(
                            coefs1, coefs2, rtol=rtol, atol=atol
                        ):
                            matched = True
                            break
                if not matched:
                    shell_groups.append((ec_pairs, ao_idx))

            # Output one entry per unique shell
            for ec_pairs, rep_ao_idx in shell_groups:
                l = self.angular_momentums[rep_ao_idx]
                shell_label = l_map.get(l, "l > i")
                info_lines.append(f"  {symbol} {shell_label}")
                for Z, coef in ec_pairs:
                    info_lines.append(f"    {Z:.6f} {coef:.7f}")

        return info_lines

    def _logger_info(self) -> None:
        """Output the information from get_info using logger.info."""
        for line in self._get_info():
            logger.info(line)

    @property
    def _nucleus_index_np(self) -> npt.NDArray[np.int32]:
        """nucleus_index."""
        return np.array(self.nucleus_index, dtype=np.int32)

    @property
    def _nucleus_index_jnp(self) -> jax.Array:
        """nucleus_index."""
        return jnp.array(self.nucleus_index, dtype=jnp.int32)

    @property
    def _nucleus_index_prim_np(self) -> npt.NDArray[np.int32]:
        """nucleus_index."""
        return np.array(self.nucleus_index)[self._orbital_indices_np]

    @property
    def _nucleus_index_prim_jnp(self) -> jax.Array:
        """nucleus_index."""
        return jnp.array(self._nucleus_index_prim_np, dtype=jnp.int32)

    @property
    def _orbital_indices_np(self) -> npt.NDArray[np.int32]:
        """orbital_index."""
        return np.array(self.orbital_indices, dtype=np.int32)

    @property
    def _orbital_indices_jnp(self) -> jax.Array:
        """orbital_index."""
        return jnp.array(self.orbital_indices, dtype=jnp.int32)

    @property
    def _atomic_center_carts_np(self) -> npt.NDArray[np.float64]:
        """Atomic positions in cartesian.

        Returns atomic positions in cartesian

        Returns:
            npt.NDArray[np.float64]: atomic positions in cartesian
        """
        return self.structure_data._positions_cart_np[self._nucleus_index_np]

    @property
    def _atomic_center_carts_jnp(self) -> jax.Array:
        """Atomic positions in cartesian.

        Returns atomic positions in cartesian

        Returns:
            jax.Array: atomic positions in cartesian
        """
        # this is super slow!!! Do not use list comprehension.
        # return jnp.array([self.structure_data.positions_cart[i] for i in self.nucleus_index])
        return self.structure_data._positions_cart_jnp[self._nucleus_index_jnp]

    @property
    def _atomic_center_carts_unique_jnp(self) -> jax.Array:
        """Unique atomic positions in cartesian.

        Returns unique atomic positions in cartesian

        Returns:
            jax.Array: atomic positions in cartesian
        """
        return self.structure_data._positions_cart_jnp
        """ the same as above.
        _, first_indices = np.unique(self.nucleus_index_np, return_index=True)
        sorted_order = jnp.argsort(first_indices)
        return self.structure_data.positions_cart_jnp[sorted_order]
        """

    @property
    def _atomic_center_carts_prim_np(self) -> npt.NDArray[np.float64]:
        """Atomic positions in cartesian for primitve orbitals.

        Returns atomic positions in cartesian for primitive orbitals

        Returns:
            npt.NDArray[np.float]: atomic positions in cartesian for primitive orbitals
        """
        return self._atomic_center_carts_np[self.orbital_indices]

    @property
    def _atomic_center_carts_prim_jnp(self) -> jax.Array:
        """Atomic positions in cartesian for primitve orbitals.

        Returns atomic positions in cartesian for primitive orbitals

        Returns:
            jax.Array: atomic positions in cartesian for primitive orbitals
        """
        # this is super slow!!! Do not use list comprehension.
        # return jnp.array([self.atomic_center_carts_jnp[i] for i in self.orbital_indices])
        return self._atomic_center_carts_jnp[self._orbital_indices_jnp]

    @property
    def _angular_momentums_prim_np(self) -> npt.NDArray[np.int32]:
        """Angular momentums for primitive orbitals.

        Returns angular momentums for primitive orbitals

        Returns:
            npt.NDArray[np.float64]: angular momentums for primitive orbitals
        """
        return np.array(self.angular_momentums, dtype=np.int32)[self._orbital_indices_np]

    @property
    def _angular_momentums_prim_jnp(self) -> jax.Array:
        """Angular momentums for primitive orbitals.

        Returns angular momentums for primitive orbitals

        Returns:
            jax.Array: angular momentums for primitive orbitals
        """
        return jnp.array(self._angular_momentums_prim_np, dtype=jnp.int32)

    @property
    def _polynominal_order_x_prim_np(self) -> npt.NDArray[np.int32]:
        """Polynominal order of x for primitive orbitals.

        Returns Polynominal order of x for primitive orbitals

        Returns:
            jax.Array: Polynominal order of x for primitive orbitals
        """
        return np.array(self.polynominal_order_x, dtype=np.int32)[self._orbital_indices_np]

    @property
    def _polynominal_order_x_prim_jnp(self) -> jax.Array:
        """Polynominal order of x for primitive orbitals.

        Returns Polynominal order of x for primitive orbitals

        Returns:
            jax.Array: Polynominal order of x for primitive orbitals
        """
        return jnp.array(self._polynominal_order_x_prim_np, dtype=np.int32)

    @property
    def _polynominal_order_y_prim_np(self) -> npt.NDArray[np.int32]:
        """Polynominal order of y for primitive orbitals.

        Returns Polynominal order of y for primitive orbitals

        Returns:
            jax.Array: Polynominal order of y for primitive orbitals
        """
        return np.array(self.polynominal_order_y, dtype=np.int32)[self._orbital_indices_np]

    @property
    def _polynominal_order_y_prim_jnp(self) -> jax.Array:
        """Polynominal order of y for primitive orbitals.

        Returns Polynominal order of y for primitive orbitals

        Returns:
            jax.Array: Polynominal order of y for primitive orbitals
        """
        return jnp.array(self._polynominal_order_y_prim_np, dtype=np.int32)

    @property
    def _polynominal_order_z_prim_np(self) -> npt.NDArray[np.int32]:
        """Polynominal order of z for primitive orbitals.

        Returns Polynominal order of z for primitive orbitals

        Returns:
            jax.Array: Polynominal order of z for primitive orbitals
        """
        return np.array(self.polynominal_order_z, dtype=np.int32)[self._orbital_indices_np]

    @property
    def _polynominal_order_z_prim_jnp(self) -> jax.Array:
        """Polynominal order of z for primitive orbitals.

        Returns Polynominal order of z for primitive orbitals

        Returns:
            jax.Array: Polynominal order of z for primitive orbitals
        """
        return jnp.array(self._polynominal_order_z_prim_np, dtype=np.int32)

    @property
    def _normalization_factorial_ratio_prim_jnp(self) -> jax.Array:
        """Return factorial ratio used in AO normalization (primitive-wise)."""
        nx = self._polynominal_order_x_prim_np
        ny = self._polynominal_order_y_prim_np
        nz = self._polynominal_order_z_prim_np
        num = (
            scipy.special.factorial(nx, exact=True)
            * scipy.special.factorial(ny, exact=True)
            * scipy.special.factorial(nz, exact=True)
        )
        den = (
            scipy.special.factorial(2 * nx, exact=True)
            * scipy.special.factorial(2 * ny, exact=True)
            * scipy.special.factorial(2 * nz, exact=True)
        )
        ratio = np.asarray(num / den, dtype=np.float64)
        return jnp.array(ratio, dtype=jnp.float64)

    @property
    def _exponents_jnp(self) -> jax.Array:
        """Return exponents."""
        return jnp.array(self.exponents, dtype=jnp.float64)

    @property
    def _coefficients_jnp(self) -> jax.Array:
        """Return coefficients."""
        return jnp.array(self.coefficients, dtype=jnp.float64)

    @property
    def _num_orb(self) -> int:
        """Return the number of orbitals."""
        return self.num_ao


@struct.dataclass
class AOs_sphe_data:
    r"""Atomic orbital definitions in real spherical-harmonic form.

    Stores contracted Gaussian basis data for atomic orbitals (AOs) whose angular part is
    a real spherical harmonic :math:`Y_{l}^{m}` with :math:`m \in \{-l,\dots,+l\}`.

    Attributes:
        structure_data (Structure_data): Molecular structure and atomic positions used to place AOs.
        nucleus_index (list[int] | tuple[int]): AO -> atom index mapping (``len == num_ao``).
        num_ao (int): Number of contracted AOs.
        num_ao_prim (int): Number of primitive Gaussians.
        orbital_indices (list[int] | tuple[int]): For each primitive, the parent AO index (``len == num_ao_prim``).
        exponents (list[float] | tuple[float]): Gaussian exponents for primitives (``len == num_ao_prim``).
        coefficients (list[float] | tuple[float]): Contraction coefficients per primitive (``len == num_ao_prim``).
        angular_momentums (list[int] | tuple[int]): Angular momentum quantum numbers ``l`` per AO (``len == num_ao``).
        magnetic_quantum_numbers (list[int] | tuple[int]): Magnetic quantum numbers ``m`` per AO (``len == num_ao``),
            satisfying ``-l <= m <= l``.

    Examples:
        Hydrogen dimer (bohr) with all-electron cc-pVTZ (Gaussian format), real spherical harmonics::

            from jqmc.structure import Structure_data
            from jqmc.atomic_orbital import AOs_sphe_data

            structure = Structure_data(
                positions=[[0.0, 0.0, -0.70], [0.0, 0.0, 0.70]],
                pbc_flag=False,
                atomic_numbers=[1, 1],
                element_symbols=["H", "H"],
                atomic_labels=["H1", "H2"],
            )

            # Per atom: 14 AOs (3 S, 2×P shells -> 6, 1×D shell -> 5); 18 primitives.
            # Two atoms -> num_ao=28, num_ao_prim=36.
            exponents = [
                # atom 1
                0.3258,
                33.87, 5.095, 1.159, 0.3258, 0.1027,
                0.1027,
                1.407, 1.407, 1.407,
                0.388, 0.388, 0.388,
                1.057, 1.057, 1.057, 1.057, 1.057,
                # atom 2 (same order)
                0.3258,
                33.87, 5.095, 1.159, 0.3258, 0.1027,
                0.1027,
                1.407, 1.407, 1.407,
                0.388, 0.388, 0.388,
                1.057, 1.057, 1.057, 1.057, 1.057,
            ]

            coefficients = [
                # atom 1
                1.0,
                0.006068, 0.045308, 0.202822, 0.503903, 0.383421,
                1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0,
                # atom 2 (same order)
                1.0,
                0.006068, 0.045308, 0.202822, 0.503903, 0.383421,
                1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0,
            ]

            orbital_indices = [
                # atom 1 AOs 0-13
                0,
                1, 1, 1, 1, 1,
                2,
                3, 4, 5,
                6, 7, 8,
                9, 10, 11, 12, 13,
                # atom 2 AOs 14-27
                14,
                15, 15, 15, 15, 15,
                16,
                17, 18, 19,
                20, 21, 22,
                23, 24, 25, 26, 27,
            ]

            nucleus_index = [
                *([0] * 14),
                *([1] * 14),
            ]

            angular_momentums = [
                # atom 1
                0, 0, 0,                # S shells
                1, 1, 1, 1, 1, 1,       # two P shells (3 each)
                2, 2, 2, 2, 2,          # one D shell (5)
                # atom 2
                0, 0, 0,
                1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2,
            ]

            magnetic_quantum_numbers = [
                # atom 1 (S: m=0; P: -1,0,1; D: -2..2)
                0, 0, 0,
                -1, 0, 1,
                -1, 0, 1,
                -2, -1, 0, 1, 2,
                # atom 2
                0, 0, 0,
                -1, 0, 1,
                -1, 0, 1,
                -2, -1, 0, 1, 2,
            ]

            aos = AOs_sphe_data(
                structure_data=structure,
                nucleus_index=nucleus_index,
                num_ao=len(angular_momentums),
                num_ao_prim=len(exponents),
                orbital_indices=orbital_indices,
                exponents=exponents,
                coefficients=coefficients,
                angular_momentums=angular_momentums,
                magnetic_quantum_numbers=magnetic_quantum_numbers,
            )

            aos.sanity_check()
    """

    #: Molecular structure and atomic positions used to place AOs.
    structure_data: Structure_data = struct.field(pytree_node=True, default_factory=lambda: Structure_data())
    #: AO -> atom index mapping (``len == num_ao``).
    nucleus_index: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)
    #: Number of contracted AOs.
    num_ao: int = struct.field(pytree_node=False, default=0)
    #: Number of primitive Gaussians.
    num_ao_prim: int = struct.field(pytree_node=False, default=0)
    #: For each primitive, the parent AO index (``len == num_ao_prim``).
    orbital_indices: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)
    #: Gaussian exponents for primitives (``len == num_ao_prim``).
    exponents: list[float] | tuple[float] = struct.field(pytree_node=False, default_factory=tuple)
    #: Contraction coefficients per primitive (``len == num_ao_prim``).
    coefficients: list[float] | tuple[float] = struct.field(pytree_node=False, default_factory=tuple)
    #: Angular momentum quantum numbers ``l`` per AO (``len == num_ao``).
    angular_momentums: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)
    #: Magnetic quantum numbers ``m`` per AO (``len == num_ao``; ``-l <= m <= l``).
    magnetic_quantum_numbers: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)

    def sanity_check(self) -> None:
        """Validate AO array shapes and basic types.

        Ensures that array lengths match declared counts (``num_ao``, ``num_ao_prim``) and that
        inputs are provided as lists/tuples (ints for counts). Call this after constructing
        ``AOs_sphe_data`` and before AO evaluation routines.

        Raises:
            ValueError: When any of the following holds:
                - ``len(nucleus_index) != num_ao``
                - ``len(unique(orbital_indices)) != num_ao``
                - ``len(exponents) != num_ao_prim`` or ``len(coefficients) != num_ao_prim``
                - ``len(angular_momentums) != num_ao``
                - ``len(magnetic_quantum_numbers) != num_ao``
                - any attribute has an unexpected Python type (e.g., non-list/tuple arrays, non-int counts)
        """
        if len(self.nucleus_index) != self.num_ao:
            raise ValueError("dim. of self.nucleus_index is wrong")
        if len(np.unique(self.orbital_indices)) != self.num_ao:
            raise ValueError(f"num_ao={self.num_ao} and/or num_ao_prim={self.num_ao_prim} is wrong")
        if len(self.exponents) != self.num_ao_prim:
            raise ValueError("dim. of self.exponents is wrong")
        if len(self.coefficients) != self.num_ao_prim:
            raise ValueError("dim. of self.coefficients is wrong")
        if len(self.angular_momentums) != self.num_ao:
            raise ValueError("dim. of self.angular_momentums is wrong")
        if len(self.magnetic_quantum_numbers) != self.num_ao:
            raise ValueError("dim. of self.magnetic_quantum_numbers is wrong")

        if not isinstance(self.nucleus_index, (list, tuple)):
            raise ValueError(f"nucleus_index = {type(self.nucleus_index)} must be a list or tuple.")
        if not isinstance(self.num_ao, (int, np.integer)):
            raise ValueError(f"num_ao = {type(self.num_ao)} must be an int.")
        if not isinstance(self.num_ao_prim, (int, np.integer)):
            raise ValueError(f"num_ao_prim = {type(self.num_ao_prim)} must be an int.")
        if not isinstance(self.orbital_indices, (list, tuple)):
            raise ValueError(f"orbital_indices = {type(self.orbital_indices)} must be a list or tuple.")
        if not isinstance(self.exponents, (list, tuple)):
            raise ValueError(f"exponents = {type(self.exponents)} must be a list or tuple.")
        if not isinstance(self.coefficients, (list, tuple)):
            raise ValueError(f"coefficients = {type(self.coefficients)} must be a list or tuple.")
        if not isinstance(self.angular_momentums, (list, tuple)):
            raise ValueError(f"angular_momentums = {type(self.angular_momentums)} must be a list or tuple.")
        if not isinstance(self.magnetic_quantum_numbers, (list, tuple)):
            raise ValueError(f"magnetic_quantum_numbers = {type(self.magnetic_quantum_numbers)} must be a list or tuple.")

        """ It works for practical cases, but it is not good for the test cases!!
        # For each nucleus_index:
        # 1) cluster primitives by (exponent, coefficient, l) within tol,
        # 2) assert each cluster has exactly 2*l+1 entries,
        # 3) assert m covers all integers from -l to +l.
        primitive_info = []
        for prim_idx, ao_idx in enumerate(self.orbital_indices):
            # validate AO index
            if not (0 <= ao_idx < self.num_ao):
                logger.error(f"Primitive {prim_idx}: AO index {ao_idx} out of range [0, {self.num_ao})")
                raise ValueError(f"AO index {ao_idx} out of range")
            exp = self.exponents[prim_idx]
            coef = self.coefficients[prim_idx]
            l = self.angular_momentums[ao_idx]
            m = self.magnetic_quantum_numbers[ao_idx]
            primitive_info.append(
                {
                    "prim_index": prim_idx,
                    "ao_index": ao_idx,
                    "exponent": exp,
                    "coefficient": coef,
                    "l": l,
                    "m": m,
                }
            )

        # 2) attach nucleus to each primitive
        for info in primitive_info:
            ao_idx = info["ao_index"]
            info["nucleus"] = self.nucleus_index[ao_idx]

        # 3) loop over each nucleus
        for nucleus in set(self.nucleus_index):
            infos_nuc = [info for info in primitive_info if info["nucleus"] == nucleus]
            if not infos_nuc:
                continue  # nothing to check for this nucleus

            # --- cluster by (exp, coef, l) ---
            clusters: list[list] = []
            for info in infos_nuc:
                exp, coef, l = info["exponent"], info["coefficient"], info["l"]
                for c_exp, c_coef, c_l, c_infos in clusters:
                    if (
                        c_l == l
                        and np.isclose(exp, c_exp, atol=atol, rtol=rtol)
                        and np.isclose(coef, c_coef, atol=atol, rtol=rtol)
                    ):
                        c_infos.append(info)
                        break
                else:
                    # no matching cluster → create new
                    clusters.append([exp, coef, l, [info]])

            # --- validate each cluster ---
            for c_exp, c_coef, l, c_infos in clusters:
                expected_count = 2 * l + 1
                actual_ms = {i["m"] for i in c_infos}
                expected_ms = set(range(-l, l + 1))

                # 3.1 count check
                if len(c_infos) != expected_count:
                    logger.error(
                        f"[nucleus={nucleus}] "
                        f"(exp≈{c_exp:.5g}, coef≈{c_coef:.5g}, l={l}): "
                        f"found {len(c_infos)} entries, expected {expected_count}"
                    )
                    raise ValueError(f"Spherical completeness count failed for nucleus {nucleus}")

                # 3.2 coverage check
                missing = expected_ms - actual_ms
                extra = actual_ms - expected_ms
                if missing or extra:
                    logger.error(
                        f"[nucleus={nucleus}] "
                        f"(exp≈{c_exp:.5g}, coef≈{c_coef:.5g}, l={l}):\n"
                        f"  missing m-values:   {sorted(missing)}\n"
                        f"  unexpected m-values:{sorted(extra)}"
                    )
                    raise ValueError(f"Spherical completeness m-coverage failed for nucleus {nucleus}")
        """

        self.structure_data.sanity_check()

    def _get_info(self) -> list[str]:
        """Return a list of strings containing information about the class attributes."""
        info_lines = []
        info_lines.extend(["**" + self.__class__.__name__])
        info_lines.extend([f"  Number of AOs = {self.num_ao}"])
        info_lines.extend([f"  Number of primitive AOs = {self.num_ao_prim}"])
        info_lines.extend(["  Angular part is the real spherical (solid) Harmonics."])

        # Map angular momentum quantum number to NWChem shell label
        l_map = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h", 6: "i"}

        # Build mapping from AO index to its list of primitive indices
        prim_per_ao: dict[int, list[int]] = {}
        for prim_idx, ao_idx in enumerate(self.orbital_indices):
            prim_per_ao.setdefault(ao_idx, []).append(prim_idx)

        # Build mapping from atom index to its list of AO indices
        ao_per_atom: dict[int, list[int]] = {}
        for ao_idx, atom_idx in enumerate(self.nucleus_index):
            ao_per_atom.setdefault(atom_idx, []).append(ao_idx)

        # Loop over atoms in sorted order
        for atom_idx in sorted(ao_per_atom):
            symbol = self.structure_data.atomic_labels[atom_idx]
            info_lines.append("  " + "-" * 36)
            info_lines.append(f"  **basis set for atom index {atom_idx + 1}: {symbol}**")
            info_lines.append("  " + "-" * 36)

            # Collect unique shells with approximate comparison
            shell_groups: list[tuple[list[tuple[float, float]], int]] = []

            for ao_idx in ao_per_atom[atom_idx]:
                prim_idxs = prim_per_ao.get(ao_idx, [])
                l = self.angular_momentums[ao_idx]

                # Recover original coefficients and build (exp, coef) pairs
                ec_pairs = []
                for prim_idx in prim_idxs:
                    Z = self.exponents[prim_idx]
                    stored_coef = self.coefficients[prim_idx]
                    # consider the normalization factor
                    N_l_m = np.sqrt((2 * l + 1) / (4 * np.pi))
                    N_n = np.sqrt(
                        (2.0 ** (2 * l + 3) * scipy.special.factorial(l + 1) * (2 * Z) ** (l + 1.5))
                        / (scipy.special.factorial(2 * l + 2) * np.sqrt(np.pi))
                    )
                    Norm = N_l_m * N_n  # which is better for its output? Todo.
                    Norm = 1  # which is better for its output? Todo.
                    orig_coef = stored_coef * Norm
                    ec_pairs.append((Z, orig_coef))

                # Attempt to match existing group within tolerance
                matched = False
                for existing_ec, _ in shell_groups:
                    if len(existing_ec) == len(ec_pairs):
                        exps1, coefs1 = zip(*existing_ec, strict=True)
                        exps2, coefs2 = zip(*ec_pairs, strict=True)
                        if np.allclose(exps1, exps2, rtol=rtol, atol=atol) and np.allclose(
                            coefs1, coefs2, rtol=rtol, atol=atol
                        ):
                            matched = True
                            break
                if not matched:
                    shell_groups.append((ec_pairs, ao_idx))

            # Output one entry per unique shell
            for ec_pairs, rep_ao_idx in shell_groups:
                l = self.angular_momentums[rep_ao_idx]
                shell_label = l_map.get(l, "l > i")
                info_lines.append(f"  {symbol} {shell_label}")
                for Z, coef in ec_pairs:
                    info_lines.append(f"    {Z:.6f} {coef:.7f}")

        return info_lines

    def _logger_info(self) -> None:
        """Output the information from get_info using logger.info."""
        for line in self._get_info():
            logger.info(line)

    @property
    def _nucleus_index_np(self) -> npt.NDArray[np.int32]:
        """nucleus_index."""
        return np.array(self.nucleus_index, dtype=np.int32)

    @property
    def _nucleus_index_jnp(self) -> jax.Array:
        """nucleus_index."""
        return jnp.array(self.nucleus_index, dtype=jnp.int32)

    @property
    def _nucleus_index_prim_np(self) -> npt.NDArray[np.int32]:
        """nucleus_index."""
        return np.array(self.nucleus_index)[self._orbital_indices_np]

    @property
    def _nucleus_index_prim_jnp(self) -> jax.Array:
        """nucleus_index."""
        return jnp.array(self._nucleus_index_prim_np, dtype=jnp.int32)

    @property
    def _orbital_indices_np(self) -> npt.NDArray[np.int32]:
        """orbital_index."""
        return np.array(self.orbital_indices, dtype=np.int32)

    @property
    def _orbital_indices_jnp(self) -> jax.Array:
        """orbital_index."""
        return jnp.array(self.orbital_indices, dtype=jnp.int32)

    @property
    def _atomic_center_carts_np(self) -> npt.NDArray[np.float64]:
        """Atomic positions in cartesian.

        Returns atomic positions in cartesian

        Returns:
            npt.NDArray[np.float64]: atomic positions in cartesian
        """
        return self.structure_data._positions_cart_np[self._nucleus_index_np]

    @property
    def _atomic_center_carts_jnp(self) -> jax.Array:
        """Atomic positions in cartesian.

        Returns atomic positions in cartesian

        Returns:
            jax.Array: atomic positions in cartesian
        """
        # this is super slow!!! Do not use list comprehension.
        # return jnp.array([self.structure_data.positions_cart[i] for i in self.nucleus_index])
        return self.structure_data._positions_cart_jnp[self._nucleus_index_jnp]

    @property
    def _atomic_center_carts_unique_jnp(self) -> jax.Array:
        """Unique atomic positions in cartesian.

        Returns unique atomic positions in cartesian

        Returns:
            jax.Array: atomic positions in cartesian
        """
        return self.structure_data._positions_cart_jnp
        """ the same as above.
        _, first_indices = np.unique(self.nucleus_index_np, return_index=True)
        sorted_order = jnp.argsort(first_indices)
        return self.structure_data.positions_cart_jnp[sorted_order]
        """

    @property
    def _atomic_center_carts_prim_np(self) -> npt.NDArray[np.float64]:
        """Atomic positions in cartesian for primitve orbitals.

        Returns atomic positions in cartesian for primitive orbitals

        Returns:
            npt.NDArray[np.float]: atomic positions in cartesian for primitive orbitals
        """
        return self._atomic_center_carts_np[self.orbital_indices]

    @property
    def _atomic_center_carts_prim_jnp(self) -> jax.Array:
        """Atomic positions in cartesian for primitve orbitals.

        Returns atomic positions in cartesian for primitive orbitals

        Returns:
            jax.Array: atomic positions in cartesian for primitive orbitals
        """
        # this is super slow!!! Do not use list comprehension.
        # return jnp.array([self.atomic_center_carts_jnp[i] for i in self.orbital_indices])
        return self._atomic_center_carts_jnp[self._orbital_indices_jnp]

    @property
    def _angular_momentums_prim_np(self) -> npt.NDArray[np.int32]:
        """Angular momentums for primitive orbitals.

        Returns angular momentums for primitive orbitals

        Returns:
            npt.NDArray[np.float64]: angular momentums for primitive orbitals
        """
        return np.array(self.angular_momentums, dtype=np.int32)[self._orbital_indices_np]

    @property
    def _angular_momentums_prim_jnp(self) -> jax.Array:
        """Angular momentums for primitive orbitals.

        Returns angular momentums for primitive orbitals

        Returns:
            jax.Array: angular momentums for primitive orbitals
        """
        return jnp.array(self._angular_momentums_prim_np, dtype=jnp.int32)

    @property
    def _magnetic_quantum_numbers_prim_np(self) -> npt.NDArray[np.int32]:
        """Magnetic quantum numbers for primitive orbitals.

        Returns magnetic quantum numbers for primitive orbitals

        Returns:
            jax.Array: magnetic quantum numbers for primitive orbitals
        """
        return np.array(self.magnetic_quantum_numbers, dtype=np.int32)[self._orbital_indices_np]

    @property
    def _magnetic_quantum_numbers_prim_jnp(self) -> jax.Array:
        """Magnetic quantum numbers for primitive orbitals.

        Returns magnetic quantum numbers for primitive orbitals

        Returns:
            npt.NDArray[np.int64]: magnetic quantum numbers for primitive orbitals
        """
        return jnp.array(self._magnetic_quantum_numbers_prim_np, dtype=jnp.int32)

    @property
    def _exponents_jnp(self) -> jax.Array:
        """Return exponents."""
        return jnp.array(self.exponents, dtype=jnp.float64)

    @property
    def _coefficients_jnp(self) -> jax.Array:
        """Return coefficients."""
        return jnp.array(self.coefficients, dtype=jnp.float64)

    @property
    def _num_orb(self) -> int:
        """Return the number of orbitals."""
        return self.num_ao


def compute_AOs(aos_data: AOs_sphe_data | AOs_cart_data, r_carts: jax.Array) -> jax.Array:
    """Evaluate contracted atomic orbitals (AOs) at electron coordinates.

    Dispatches to Cartesian or real-spherical backends and returns float64 JAX arrays
    (ensure ``jax_enable_x64=True``). Call ``aos_data.sanity_check()`` before use.

    Args:
        aos_data: ``AOs_cart_data`` or ``AOs_sphe_data`` describing centers, primitive parameters,
            angular data, and contraction mapping.
        r_carts (jax.Array): Electron Cartesian coordinates, shape ``(N_e, 3)`` in Bohr. Casts to
            ``float64`` internally via ``jnp.asarray``.

    Returns:
        jax.Array: AO values, shape ``(num_ao, N_e)``.

    Raises:
        NotImplementedError: If ``aos_data`` is neither Cartesian nor spherical.
    """
    r_carts = jnp.asarray(r_carts, dtype=jnp.float64)

    if isinstance(aos_data, AOs_sphe_data):
        AOs = _compute_AOs_sphe(aos_data, r_carts)

    elif isinstance(aos_data, AOs_cart_data):
        AOs = _compute_AOs_cart(aos_data, r_carts)
    else:
        raise NotImplementedError
    return AOs


def _compute_AOs_debug(aos_data: AOs_sphe_data | AOs_cart_data, r_carts: jnpt.ArrayLike) -> jax.Array:
    """Debug function."""
    if isinstance(aos_data, AOs_sphe_data):
        AOs = _compute_AOs_sphe_debug(aos_data, r_carts)

    elif isinstance(aos_data, AOs_cart_data):
        AOs = _compute_AOs_cart_debug(aos_data, r_carts)
    else:
        raise NotImplementedError
    return AOs


def _compute_AOs_sphe_debug(aos_data: AOs_sphe_data, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute AO values at the given r_carts.

    The method is for computing the value of the given atomic orbital at r_carts
    for debugging purpose. See compute_AOs_api.
    """
    aos_values = []

    for ao_index in range(aos_data.num_ao):
        atomic_center_cart = aos_data._atomic_center_carts_np[ao_index]
        shell_indices = [i for i, v in enumerate(aos_data.orbital_indices) if v == ao_index]
        exponents = [aos_data.exponents[i] for i in shell_indices]
        coefficients = [aos_data.coefficients[i] for i in shell_indices]
        angular_momentum = aos_data.angular_momentums[ao_index]
        magnetic_quantum_number = aos_data.magnetic_quantum_numbers[ao_index]
        ao_value = []
        for r_cart in r_carts:
            # radial part
            R_n = np.array(
                [
                    coefficient * np.exp(-1.0 * exponent * LA.norm(np.array(r_cart) - np.array(atomic_center_cart)) ** 2)
                    for coefficient, exponent in zip(coefficients, exponents, strict=True)
                ]
            )
            # normalization part
            N_n_l = np.array(
                [
                    np.sqrt(
                        (
                            2.0 ** (2 * angular_momentum + 3)
                            * scipy.special.factorial(angular_momentum + 1)
                            * (2 * Z) ** (angular_momentum + 1.5)
                        )
                        / (scipy.special.factorial(2 * angular_momentum + 2) * np.sqrt(np.pi))
                    )
                    for Z in exponents
                ]
            )
            # angular part
            S_l_m = _compute_S_l_m_debug(
                atomic_center_cart=atomic_center_cart,
                angular_momentum=angular_momentum,
                magnetic_quantum_number=magnetic_quantum_number,
                r_cart=r_cart,
            )

            ao_value.append(np.sum(N_n_l * R_n) * np.sqrt((2 * angular_momentum + 1) / (4 * np.pi)) * S_l_m)

        aos_values.append(ao_value)

    return aos_values


def _compute_AOs_cart_debug(aos_data: AOs_cart_data, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute AO values at the given r_carts.

    The method is for computing the value of the given atomic orbital at r_carts
    for debugging purpose. See compute_AOs_api.
    """
    aos_values = []

    for ao_index in range(aos_data.num_ao):
        R_cart = aos_data._atomic_center_carts_np[ao_index]
        l = aos_data.angular_momentums[ao_index]
        shell_indices = [i for i, v in enumerate(aos_data.orbital_indices) if v == ao_index]
        exponents = [aos_data.exponents[i] for i in shell_indices]
        coefficients = [aos_data.coefficients[i] for i in shell_indices]
        nx = aos_data.polynominal_order_x[ao_index]
        ny = aos_data.polynominal_order_y[ao_index]
        nz = aos_data.polynominal_order_z[ao_index]

        ao_value = []
        for r_cart in r_carts:
            # radial part
            R_n = np.array(
                [
                    coefficient * np.exp(-1.0 * exponent * LA.norm(np.array(r_cart) - np.array(R_cart)) ** 2)
                    for coefficient, exponent in zip(coefficients, exponents, strict=True)
                ]
            )
            # normalization part
            N_n_l = np.array(
                [
                    np.sqrt(
                        (2.0 * Z / np.pi) ** (3.0 / 2.0)
                        * (8.0 * Z) ** l
                        * scipy.special.factorial(nx)
                        * scipy.special.factorial(ny)
                        * scipy.special.factorial(nz)
                        / (scipy.special.factorial(2 * nx) * scipy.special.factorial(2 * ny) * scipy.special.factorial(2 * nz))
                    )
                    for Z in exponents
                ]
            )
            # angular part
            x, y, z = np.array(r_cart) - np.array(R_cart)
            P_l_nx_ny_nz = x**nx * y**ny * z**nz

            ao_value.append(np.sum(N_n_l * R_n) * P_l_nx_ny_nz)

        aos_values.append(ao_value)

    return aos_values


@jit
def _compute_AOs_cart(aos_data: AOs_cart_data, r_carts: jnpt.ArrayLike) -> jax.Array:
    """Compute AO values at the given r_carts.

    See compute_AOs_api

    """
    # Indices with respect to the contracted AOs
    R_carts_jnp = aos_data._atomic_center_carts_prim_jnp
    c_jnp = aos_data._coefficients_jnp
    Z_jnp = aos_data._exponents_jnp
    l_jnp = aos_data._angular_momentums_prim_jnp
    nx_jnp = aos_data._polynominal_order_x_prim_jnp
    ny_jnp = aos_data._polynominal_order_y_prim_jnp
    nz_jnp = aos_data._polynominal_order_z_prim_jnp

    N_n_dup_fuctorial_part = aos_data._normalization_factorial_ratio_prim_jnp
    N_n_dup_Z_part = (2.0 * Z_jnp / jnp.pi) ** (3.0 / 2.0) * (8.0 * Z_jnp) ** l_jnp
    N_n_dup = jnp.sqrt(N_n_dup_Z_part * N_n_dup_fuctorial_part)
    r_R_diffs = r_carts[None, :, :] - R_carts_jnp[:, None, :]
    r_squared = jnp.sum(r_R_diffs**2, axis=-1)
    R_n_dup = c_jnp[:, None] * jnp.exp(-Z_jnp[:, None] * r_squared)

    x, y, z = r_R_diffs[..., 0], r_R_diffs[..., 1], r_R_diffs[..., 2]
    eps = 1.0e-16  # This is quite important to avoid some numerical instability in JAX!!
    P_l_nx_ny_nz_dup = (x + eps) ** (nx_jnp[:, None]) * (y + eps) ** (ny_jnp[:, None]) * (z + eps) ** (nz_jnp[:, None])

    """
    logger.info(f"Z_jnp={Z_jnp}.")
    logger.info(f"l_jnp={l_jnp}.")
    logger.info(f"nx_jnp={nx_jnp}.")
    logger.info(f"ny_jnp={ny_jnp}.")
    logger.info(f"nz_jnp={nz_jnp}.")
    logger.info(f"N_n_dup={N_n_dup.shape}, R_n_dup={R_n_dup.shape}")
    logger.info(f"N_n_dup={N_n_dup.shape}, R_n_dup={R_n_dup.shape}")
    logger.info(f"l_jnp={l_jnp.shape}, Z_jnp={Z_jnp.shape}.")
    logger.info(f"nx_jnp={nx_jnp.shape}, ny_jnp={ny_jnp.shape}, nz_jnp={nz_jnp.shape}")
    """

    AOs_dup = N_n_dup[:, None] * R_n_dup * P_l_nx_ny_nz_dup

    orbital_indices = aos_data._orbital_indices_jnp
    num_segments = aos_data.num_ao
    AOs = jax.ops.segment_sum(AOs_dup, orbital_indices, num_segments=num_segments)
    return AOs


@jit
def _compute_AOs_sphe(aos_data: AOs_sphe_data, r_carts: jnpt.ArrayLike) -> jax.Array:
    """Compute AO values at the given r_carts.

    See compute_AOs_api

    """
    # Indices with respect to the contracted AOs
    # compute R_n inc. the whole normalization factor
    nucleus_index_prim_jnp = aos_data._nucleus_index_prim_jnp
    R_carts_jnp = aos_data._atomic_center_carts_prim_jnp
    R_carts_unique_jnp = aos_data._atomic_center_carts_unique_jnp
    c_jnp = aos_data._coefficients_jnp
    Z_jnp = aos_data._exponents_jnp
    l_jnp = aos_data._angular_momentums_prim_jnp
    m_jnp = aos_data._magnetic_quantum_numbers_prim_jnp

    # use float64 gamma-based factorials to avoid float32 drift vs debug implementation
    l_f64 = l_jnp.astype(jnp.float64)
    Z_f64 = Z_jnp.astype(jnp.float64)
    factorial_l_plus_1 = jnp.exp(jscipy.special.gammaln(l_f64 + 2.0))
    factorial_2l_plus_2 = jnp.exp(jscipy.special.gammaln(2.0 * l_f64 + 3.0))

    N_n_dup = jnp.sqrt(
        (2.0 ** (2 * l_f64 + 3) * factorial_l_plus_1 * (2 * Z_f64) ** (l_f64 + 1.5)) / (factorial_2l_plus_2 * jnp.sqrt(jnp.pi))
    )
    N_l_m_dup = jnp.sqrt((2 * l_f64 + 1) / (4 * jnp.pi))
    r_R_diffs = r_carts[None, :, :] - R_carts_jnp[:, None, :]
    r_squared = jnp.sum(r_R_diffs**2, axis=-1)
    R_n_dup = c_jnp[:, None] * jnp.exp(-Z_jnp[:, None] * r_squared)
    r_R_diffs_uq = r_carts[None, :, :] - R_carts_unique_jnp[:, None, :]

    max_ml, S_l_m_dup_all_l_m = _compute_S_l_m(r_R_diffs_uq)
    S_l_m_dup_all_l_m_reshaped = S_l_m_dup_all_l_m.reshape(
        (S_l_m_dup_all_l_m.shape[0] * S_l_m_dup_all_l_m.shape[1], S_l_m_dup_all_l_m.shape[2]), order="F"
    )
    global_l_m_index = l_jnp**2 + (m_jnp + l_jnp)
    global_R_l_m_index = nucleus_index_prim_jnp * max_ml + global_l_m_index
    S_l_m_dup = S_l_m_dup_all_l_m_reshaped[global_R_l_m_index]

    AOs_dup = N_n_dup[:, None] * R_n_dup * N_l_m_dup[:, None] * S_l_m_dup

    orbital_indices = aos_data._orbital_indices_jnp
    num_segments = aos_data.num_ao
    AOs = jax.ops.segment_sum(AOs_dup, orbital_indices, num_segments=num_segments)
    return AOs


@jit
def _compute_S_l_m(
    r_R_diffs: jnpt.ArrayLike,
) -> jax.Array:
    r"""Solid harmonics part of a primitve AO.

    Compute the solid harmonics, i.e., r^l * spherical hamonics part (c.f., regular solid harmonics) of a given AO

    Args:
        r_R_diffs ( jnpt.ArrayLike): Cartesian coordinate of N electrons - Cartesian corrdinates of M nuclei. dim: (N,M,3)

    Returns:
        jax.Array: dim:(49,N,M) arrays of the spherical harmonics part * r^l (i.e., regular solid harmonics) for all (l,m) pairs.
    """
    x, y, z = r_R_diffs[..., 0], r_R_diffs[..., 1], r_R_diffs[..., 2]
    r_norm = jnp.sqrt(x**2 + y**2 + z**2)

    def lnorm(l):
        return jnp.sqrt((4 * jnp.pi) / (2 * l + 1))

    """see https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics (l=0-4)"""
    """Useful tool to generate spherical harmonics generator [https://github.com/elerac/sh_table]"""
    max_ml = 49
    # s orbital
    s_0 = lnorm(l=0) * 1.0 / 2.0 * jnp.sqrt(1.0 / jnp.pi) * r_norm**0.0  # (l, m) == (0, 0)
    # p orbitals
    p_m1 = lnorm(l=1) * jnp.sqrt(3.0 / (4 * jnp.pi)) * y  # (l, m) == (1, -1)
    p_0 = lnorm(l=1) * jnp.sqrt(3.0 / (4 * jnp.pi)) * z  # (l, m) == (1, 0)
    p_p1 = lnorm(l=1) * jnp.sqrt(3.0 / (4 * jnp.pi)) * x  # (l, m) == (1, 1)
    # d orbitals
    d_m2 = lnorm(l=2) * 1.0 / 2.0 * jnp.sqrt(15.0 / (jnp.pi)) * x * y  # (l, m) == (2, -2)
    d_m1 = lnorm(l=2) * 1.0 / 2.0 * jnp.sqrt(15.0 / (jnp.pi)) * y * z  # (l, m) == (2, -1)
    d_0 = lnorm(l=2) * 1.0 / 4.0 * jnp.sqrt(5.0 / (jnp.pi)) * (3 * z**2 - r_norm**2)  # (l, m) == (2, 0):
    d_p1 = lnorm(l=2) * 1.0 / 2.0 * jnp.sqrt(15.0 / (jnp.pi)) * x * z  # (l, m) == (2, 1)
    d_p2 = lnorm(l=2) * 1.0 / 4.0 * jnp.sqrt(15.0 / (jnp.pi)) * (x**2 - y**2)  # (l, m) == (2, 2)
    # f orbitals
    f_m3 = lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(35.0 / (2 * jnp.pi)) * y * (3 * x**2 - y**2)  # (l, m) == (3, -3)
    f_m2 = lnorm(l=3) * 1.0 / 2.0 * jnp.sqrt(105.0 / (jnp.pi)) * x * y * z  # (l, m) == (3, -2)
    f_m1 = lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(21.0 / (2 * jnp.pi)) * y * (5 * z**2 - r_norm**2)  # (l, m) == (3, -1)
    f_0 = lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(7.0 / (jnp.pi)) * (5 * z**3 - 3 * z * r_norm**2)  # (l, m) == (3, 0)
    f_p1 = lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(21.0 / (2 * jnp.pi)) * x * (5 * z**2 - r_norm**2)  # (l, m) == (3, 1)
    f_p2 = lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(105.0 / (jnp.pi)) * (x**2 - y**2) * z  # (l, m) == (3, 2)
    f_p3 = lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(35.0 / (2 * jnp.pi)) * x * (x**2 - 3 * y**2)  # (l, m) == (3, 3)
    # g orbitals
    g_m4 = lnorm(l=4) * 3.0 / 4.0 * jnp.sqrt(35.0 / (jnp.pi)) * x * y * (x**2 - y**2)  # (l, m) == (4, -4)
    g_m3 = lnorm(l=4) * 3.0 / 4.0 * jnp.sqrt(35.0 / (2 * jnp.pi)) * y * z * (3 * x**2 - y**2)  # (l, m) == (4, -3)
    g_m2 = lnorm(l=4) * 3.0 / 4.0 * jnp.sqrt(5.0 / (jnp.pi)) * x * y * (7 * z**2 - r_norm**2)  # (l, m) == (4, -2)
    g_m1 = lnorm(l=4) * 3.0 / 4.0 * jnp.sqrt(5.0 / (2 * jnp.pi)) * y * (7 * z**3 - 3 * z * r_norm**2)  # (l, m) == (4, -1)
    g_0 = (
        lnorm(l=4) * 3.0 / 16.0 * jnp.sqrt(1.0 / (jnp.pi)) * (35 * z**4 - 30 * z**2 * r_norm**2 + 3 * r_norm**4)
    )  # (l, m) == (4, 0)
    g_p1 = lnorm(l=4) * 3.0 / 4.0 * jnp.sqrt(5.0 / (2 * jnp.pi)) * x * (7 * z**3 - 3 * z * r_norm**2)  # (l, m) == (4, 1)
    g_p2 = lnorm(l=4) * 3.0 / 8.0 * jnp.sqrt(5.0 / (jnp.pi)) * (x**2 - y**2) * (7 * z**2 - r_norm**2)  # (l, m) == (4, 2)
    g_p3 = lnorm(l=4) * (3.0 / 4.0 * jnp.sqrt(35.0 / (2 * jnp.pi)) * x * z * (x**2 - 3 * y**2))  # (l, m) == (4, 3)
    g_p4 = (
        lnorm(l=4) * 3.0 / 16.0 * jnp.sqrt(35.0 / (jnp.pi)) * (x**2 * (x**2 - 3 * y**2) - y**2 * (3 * x**2 - y**2))
    )  # (l, m) == (4, 4)
    # h orbitals
    h_m5 = lnorm(5) * 3.0 / 16.0 * jnp.sqrt(77.0 / (2 * jnp.pi)) * (5 * x**4 * y - 10 * x**2 * y**3 + y**5)  # (l, m) == (5, -5)
    h_m4 = lnorm(5) * 3.0 / 16.0 * jnp.sqrt(385.0 / jnp.pi) * 4 * x * y * z * (x**2 - y**2)  # (l, m) == (5, -4)
    h_m3 = (
        lnorm(5) * 1.0 / 16.0 * jnp.sqrt(385.0 / (2 * jnp.pi)) * -1 * (y**3 - 3 * x**2 * y) * (9 * z**2 - (x**2 + y**2 + z**2))
    )  # (l, m) == (5, -3)
    h_m2 = (
        lnorm(5) * 1.0 / 8.0 * jnp.sqrt(1155 / jnp.pi) * 2 * x * y * (3 * z**3 - z * (x**2 + y**2 + z**2))
    )  # (l, m) == (5, -2)
    h_m1 = (
        lnorm(5)
        * 1.0
        / 16.0
        * jnp.sqrt(165 / jnp.pi)
        * y
        * (21 * z**4 - 14 * z**2 * (x**2 + y**2 + z**2) + (x**2 + y**2 + z**2) ** 2)
    )  # (l, m) == (5, -1)
    h_0 = (
        lnorm(5)
        * 1.0
        / 16.0
        * jnp.sqrt(11 / jnp.pi)
        * (63 * z**5 - 70 * z**3 * (x**2 + y**2 + z**2) + 15 * z * (x**2 + y**2 + z**2) ** 2)
    )  # (l, m) == (5, 0)
    h_p1 = (
        lnorm(5)
        * 1.0
        / 16.0
        * jnp.sqrt(165 / jnp.pi)
        * x
        * (21 * z**4 - 14 * z**2 * (x**2 + y**2 + z**2) + (x**2 + y**2 + z**2) ** 2)
    )  # (l, m) == (5, 1)
    h_p2 = (
        lnorm(5) * 1.0 / 8.0 * jnp.sqrt(1155 / jnp.pi) * (x**2 - y**2) * (3 * z**3 - z * (x**2 + y**2 + z**2))
    )  # (l, m) == (5, 2)
    h_p3 = (
        lnorm(5) * 1.0 / 16.0 * jnp.sqrt(385.0 / (2 * jnp.pi)) * (x**3 - 3 * x * y**2) * (9 * z**2 - (x**2 + y**2 + z**2))
    )  # (l, m) == (5, 3)
    h_p4 = (
        lnorm(5) * 3.0 / 16.0 * jnp.sqrt(385.0 / jnp.pi) * (x**2 * z * (x**2 - 3 * y**2) - y**2 * z * (3 * x**2 - y**2))
    )  # (l, m) == (5, 4)
    h_p5 = lnorm(5) * 3.0 / 16.0 * jnp.sqrt(77.0 / (2 * jnp.pi)) * (x**5 - 10 * x**3 * y**2 + 5 * x * y**4)  # (l, m) == (5, 5)
    # i orbitals
    i_m6 = (
        lnorm(6) * 1.0 / 64.0 * jnp.sqrt(6006.0 / jnp.pi) * (6 * x**5 * y - 20 * x**3 * y**3 + 6 * x * y**5)
    )  # (l, m) == (6, -6)
    i_m5 = lnorm(6) * 3.0 / 32.0 * jnp.sqrt(2002.0 / jnp.pi) * z * (5 * x**4 * y - 10 * x**2 * y**3 + y**5)  # (l, m) == (6, -5)
    i_m4 = (
        lnorm(6) * 3.0 / 32.0 * jnp.sqrt(91.0 / jnp.pi) * 4 * x * y * (11 * z**2 - (x**2 + y**2 + z**2)) * (x**2 - y**2)
    )  # (l, m) == (6, -4)
    i_m3 = (
        lnorm(6)
        * 1.0
        / 32.0
        * jnp.sqrt(2730.0 / jnp.pi)
        * -1
        * (11 * z**3 - 3 * z * (x**2 + y**2 + z**2))
        * (y**3 - 3 * x**2 * y)
    )  # (l, m) == (6, -3)
    i_m2 = (
        lnorm(6)
        * 1.0
        / 64.0
        * jnp.sqrt(2730.0 / jnp.pi)
        * 2
        * x
        * y
        * (33 * z**4 - 18 * z**2 * (x**2 + y**2 + z**2) + (x**2 + y**2 + z**2) ** 2)
    )  # (l, m) == (6, -2)
    i_m1 = (
        lnorm(6)
        * 1.0
        / 16.0
        * jnp.sqrt(273.0 / jnp.pi)
        * y
        * (33 * z**5 - 30 * z**3 * (x**2 + y**2 + z**2) + 5 * z * (x**2 + y**2 + z**2) ** 2)
    )  # (l, m) == (6, -1)
    i_0 = (
        lnorm(6)
        * 1.0
        / 32.0
        * jnp.sqrt(13.0 / jnp.pi)
        * (
            231 * z**6
            - 315 * z**4 * (x**2 + y**2 + z**2)
            + 105 * z**2 * (x**2 + y**2 + z**2) ** 2
            - 5 * (x**2 + y**2 + z**2) ** 3
        )
    )  # (l, m) == (6, 0)
    i_p1 = (
        lnorm(6)
        * 1.0
        / 16.0
        * jnp.sqrt(273.0 / jnp.pi)
        * x
        * (33 * z**5 - 30 * z**3 * (x**2 + y**2 + z**2) + 5 * z * (x**2 + y**2 + z**2) ** 2)
    )  # (l, m) == (6, 1)
    i_p2 = (
        lnorm(6)
        * 1.0
        / 64.0
        * jnp.sqrt(2730.0 / jnp.pi)
        * (x**2 - y**2)
        * (33 * z**4 - 18 * z**2 * (x**2 + y**2 + z**2) + (x**2 + y**2 + z**2) ** 2)
    )  # (l, m) == (6, 2)
    i_p3 = (
        lnorm(6) * 1.0 / 32.0 * jnp.sqrt(2730.0 / jnp.pi) * (11 * z**3 - 3 * z * (x**2 + y**2 + z**2)) * (x**3 - 3 * x * y**2)
    )  # (l, m) == (6, 3)
    i_p4 = (
        lnorm(6)
        * 3.0
        / 32.0
        * jnp.sqrt(91.0 / jnp.pi)
        * (11 * z**2 - (x**2 + y**2 + z**2))
        * (x**2 * (x**2 - 3 * y**2) + y**2 * (y**2 - 3 * x**2))
    )  # (l, m) == (6, 4)
    i_p5 = lnorm(6) * 3.0 / 32.0 * jnp.sqrt(2002.0 / jnp.pi) * z * (x**5 - 10 * x**3 * y**2 + 5 * x * y**4)  # (l, m) == (6, 5)
    i_p6 = (
        lnorm(6) * 1.0 / 64.0 * jnp.sqrt(6006.0 / jnp.pi) * (x**6 - 15 * x**4 * y**2 + 15 * x**2 * y**4 - y**6)
    )  # (l, m) == (6, 6)

    S_l_m_values = jnp.stack(
        [
            s_0,
            p_m1,
            p_0,
            p_p1,
            d_m2,
            d_m1,
            d_0,
            d_p1,
            d_p2,
            f_m3,
            f_m2,
            f_m1,
            f_0,
            f_p1,
            f_p2,
            f_p3,
            g_m4,
            g_m3,
            g_m2,
            g_m1,
            g_0,
            g_p1,
            g_p2,
            g_p3,
            g_p4,
            h_m5,
            h_m4,
            h_m3,
            h_m2,
            h_m1,
            h_0,
            h_p1,
            h_p2,
            h_p3,
            h_p4,
            h_p5,
            i_m6,
            i_m5,
            i_m4,
            i_m3,
            i_m2,
            i_m1,
            i_0,
            i_p1,
            i_p2,
            i_p3,
            i_p4,
            i_p5,
            i_p6,
        ],
        axis=0,
    )
    return max_ml, S_l_m_values


@jit
def _compute_S_l_m_and_grad_lap(r_R_diffs_uq: jnp.ndarray) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Vectorized solid harmonics values, gradients, and Laplacians.

    Returns:
        tuple: (values, grads, laps) where values has shape (49, num_R, num_r), grads has shape (49, num_R, num_r, 3),
        and laps has shape (49, num_R, num_r).
    """
    S_L_M_COEFFS = (
        jnp.array([1.0], dtype=jnp.float64),
        jnp.array([1.0], dtype=jnp.float64),
        jnp.array([1.0], dtype=jnp.float64),
        jnp.array([1.0], dtype=jnp.float64),
        jnp.array([1.7320508075688774], dtype=jnp.float64),
        jnp.array([1.7320508075688774], dtype=jnp.float64),
        jnp.array([1.0, -0.5, -0.5], dtype=jnp.float64),
        jnp.array([1.7320508075688774], dtype=jnp.float64),
        jnp.array([-0.8660254037844387, 0.8660254037844387], dtype=jnp.float64),
        jnp.array([-0.7905694150420949, 2.3717082451262845], dtype=jnp.float64),
        jnp.array([3.8729833462074166], dtype=jnp.float64),
        jnp.array([2.4494897427831783, -0.6123724356957946, -0.6123724356957946], dtype=jnp.float64),
        jnp.array([1.0, -1.5, -1.5], dtype=jnp.float64),
        jnp.array([2.4494897427831783, -0.6123724356957946, -0.6123724356957946], dtype=jnp.float64),
        jnp.array([-1.9364916731037083, 1.9364916731037083], dtype=jnp.float64),
        jnp.array([-2.3717082451262845, 0.7905694150420949], dtype=jnp.float64),
        jnp.array([-2.958039891549808, 2.958039891549808], dtype=jnp.float64),
        jnp.array([-2.091650066335189, 6.274950199005566], dtype=jnp.float64),
        jnp.array([6.708203932499371, -1.1180339887498951, -1.1180339887498951], dtype=jnp.float64),
        jnp.array([3.1622776601683795, -2.3717082451262845, -2.3717082451262845], dtype=jnp.float64),
        jnp.array([1.0, -3.0, 0.375, -3.0, 0.75, 0.375], dtype=jnp.float64),
        jnp.array([3.1622776601683795, -2.3717082451262845, -2.3717082451262845], dtype=jnp.float64),
        jnp.array([-3.3541019662496856, 0.5590169943749476, 3.3541019662496856, -0.5590169943749476], dtype=jnp.float64),
        jnp.array([-6.274950199005566, 2.091650066335189], dtype=jnp.float64),
        jnp.array([0.739509972887452, -4.437059837324712, 0.739509972887452], dtype=jnp.float64),
        jnp.array([0.7015607600201141, -7.015607600201141, 3.5078038001005707], dtype=jnp.float64),
        jnp.array([-8.874119674649426, 8.874119674649426], dtype=jnp.float64),
        jnp.array(
            [-4.183300132670378, 0.5229125165837972, 12.549900398011133, -1.0458250331675945, -1.5687375497513916],
            dtype=jnp.float64,
        ),
        jnp.array([10.2469507659596, -5.1234753829798, -5.1234753829798], dtype=jnp.float64),
        jnp.array(
            [
                3.872983346207417,
                -5.809475019311126,
                0.4841229182759271,
                -5.809475019311126,
                0.9682458365518543,
                0.4841229182759271,
            ],
            dtype=jnp.float64,
        ),
        jnp.array([1.0, -5.0, 1.875, -5.0, 3.75, 1.875], dtype=jnp.float64),
        jnp.array(
            [
                3.872983346207417,
                -5.809475019311126,
                0.4841229182759271,
                -5.809475019311126,
                0.9682458365518543,
                0.4841229182759271,
            ],
            dtype=jnp.float64,
        ),
        jnp.array([-5.1234753829798, 2.5617376914899, 5.1234753829798, -2.5617376914899], dtype=jnp.float64),
        jnp.array(
            [-12.549900398011133, 1.5687375497513916, 4.183300132670378, 1.0458250331675945, -0.5229125165837972],
            dtype=jnp.float64,
        ),
        jnp.array([2.2185299186623566, -13.311179511974139, 2.2185299186623566], dtype=jnp.float64),
        jnp.array([3.5078038001005707, -7.015607600201141, 0.7015607600201141], dtype=jnp.float64),
        jnp.array([4.030159736288377, -13.433865787627923, 4.030159736288377], dtype=jnp.float64),
        jnp.array([2.3268138086232857, -23.268138086232856, 11.634069043116428], dtype=jnp.float64),
        jnp.array([-19.843134832984433, 1.9843134832984433, 19.843134832984433, -1.9843134832984433], dtype=jnp.float64),
        jnp.array(
            [-7.245688373094719, 2.7171331399105196, 21.737065119284157, -5.434266279821039, -8.15139941973156],
            dtype=jnp.float64,
        ),
        jnp.array(
            [
                14.491376746189438,
                -14.491376746189438,
                0.9057110466368399,
                -14.491376746189438,
                1.8114220932736798,
                0.9057110466368399,
            ],
            dtype=jnp.float64,
        ),
        jnp.array(
            [4.58257569495584, -11.4564392373896, 2.8641098093474, -11.4564392373896, 5.7282196186948, 2.8641098093474],
            dtype=jnp.float64,
        ),
        jnp.array([1.0, -7.5, 5.625, -0.3125, -7.5, 11.25, -0.9375, 5.625, -0.9375, -0.3125], dtype=jnp.float64),
        jnp.array(
            [4.58257569495584, -11.4564392373896, 2.8641098093474, -11.4564392373896, 5.7282196186948, 2.8641098093474],
            dtype=jnp.float64,
        ),
        jnp.array(
            [
                -7.245688373094719,
                7.245688373094719,
                -0.45285552331841994,
                7.245688373094719,
                -0.45285552331841994,
                -7.245688373094719,
                0.45285552331841994,
                0.45285552331841994,
            ],
            dtype=jnp.float64,
        ),
        jnp.array(
            [-21.737065119284157, 8.15139941973156, 7.245688373094719, 5.434266279821039, -2.7171331399105196],
            dtype=jnp.float64,
        ),
        jnp.array(
            [
                4.960783708246108,
                -0.4960783708246108,
                -29.76470224947665,
                2.480391854123054,
                4.960783708246108,
                2.480391854123054,
                -0.4960783708246108,
            ],
            dtype=jnp.float64,
        ),
        jnp.array([11.634069043116428, -23.268138086232856, 2.3268138086232857], dtype=jnp.float64),
        jnp.array([-0.6716932893813962, 10.075399340720942, -10.075399340720942, 0.6716932893813962], dtype=jnp.float64),
    )

    S_L_M_EXPS = (
        jnp.array([[0, 0, 0]], dtype=jnp.int32),
        jnp.array([[0, 1, 0]], dtype=jnp.int32),
        jnp.array([[0, 0, 1]], dtype=jnp.int32),
        jnp.array([[1, 0, 0]], dtype=jnp.int32),
        jnp.array([[1, 1, 0]], dtype=jnp.int32),
        jnp.array([[0, 1, 1]], dtype=jnp.int32),
        jnp.array([[0, 0, 2], [0, 2, 0], [2, 0, 0]], dtype=jnp.int32),
        jnp.array([[1, 0, 1]], dtype=jnp.int32),
        jnp.array([[0, 2, 0], [2, 0, 0]], dtype=jnp.int32),
        jnp.array([[0, 3, 0], [2, 1, 0]], dtype=jnp.int32),
        jnp.array([[1, 1, 1]], dtype=jnp.int32),
        jnp.array([[0, 1, 2], [0, 3, 0], [2, 1, 0]], dtype=jnp.int32),
        jnp.array([[0, 0, 3], [0, 2, 1], [2, 0, 1]], dtype=jnp.int32),
        jnp.array([[1, 0, 2], [1, 2, 0], [3, 0, 0]], dtype=jnp.int32),
        jnp.array([[0, 2, 1], [2, 0, 1]], dtype=jnp.int32),
        jnp.array([[1, 2, 0], [3, 0, 0]], dtype=jnp.int32),
        jnp.array([[1, 3, 0], [3, 1, 0]], dtype=jnp.int32),
        jnp.array([[0, 3, 1], [2, 1, 1]], dtype=jnp.int32),
        jnp.array([[1, 1, 2], [1, 3, 0], [3, 1, 0]], dtype=jnp.int32),
        jnp.array([[0, 1, 3], [0, 3, 1], [2, 1, 1]], dtype=jnp.int32),
        jnp.array([[0, 0, 4], [0, 2, 2], [0, 4, 0], [2, 0, 2], [2, 2, 0], [4, 0, 0]], dtype=jnp.int32),
        jnp.array([[1, 0, 3], [1, 2, 1], [3, 0, 1]], dtype=jnp.int32),
        jnp.array([[0, 2, 2], [0, 4, 0], [2, 0, 2], [4, 0, 0]], dtype=jnp.int32),
        jnp.array([[1, 2, 1], [3, 0, 1]], dtype=jnp.int32),
        jnp.array([[0, 4, 0], [2, 2, 0], [4, 0, 0]], dtype=jnp.int32),
        jnp.array([[0, 5, 0], [2, 3, 0], [4, 1, 0]], dtype=jnp.int32),
        jnp.array([[1, 3, 1], [3, 1, 1]], dtype=jnp.int32),
        jnp.array([[0, 3, 2], [0, 5, 0], [2, 1, 2], [2, 3, 0], [4, 1, 0]], dtype=jnp.int32),
        jnp.array([[1, 1, 3], [1, 3, 1], [3, 1, 1]], dtype=jnp.int32),
        jnp.array([[0, 1, 4], [0, 3, 2], [0, 5, 0], [2, 1, 2], [2, 3, 0], [4, 1, 0]], dtype=jnp.int32),
        jnp.array([[0, 0, 5], [0, 2, 3], [0, 4, 1], [2, 0, 3], [2, 2, 1], [4, 0, 1]], dtype=jnp.int32),
        jnp.array([[1, 0, 4], [1, 2, 2], [1, 4, 0], [3, 0, 2], [3, 2, 0], [5, 0, 0]], dtype=jnp.int32),
        jnp.array([[0, 2, 3], [0, 4, 1], [2, 0, 3], [4, 0, 1]], dtype=jnp.int32),
        jnp.array([[1, 2, 2], [1, 4, 0], [3, 0, 2], [3, 2, 0], [5, 0, 0]], dtype=jnp.int32),
        jnp.array([[0, 4, 1], [2, 2, 1], [4, 0, 1]], dtype=jnp.int32),
        jnp.array([[1, 4, 0], [3, 2, 0], [5, 0, 0]], dtype=jnp.int32),
        jnp.array([[1, 5, 0], [3, 3, 0], [5, 1, 0]], dtype=jnp.int32),
        jnp.array([[0, 5, 1], [2, 3, 1], [4, 1, 1]], dtype=jnp.int32),
        jnp.array([[1, 3, 2], [1, 5, 0], [3, 1, 2], [5, 1, 0]], dtype=jnp.int32),
        jnp.array([[0, 3, 3], [0, 5, 1], [2, 1, 3], [2, 3, 1], [4, 1, 1]], dtype=jnp.int32),
        jnp.array([[1, 1, 4], [1, 3, 2], [1, 5, 0], [3, 1, 2], [3, 3, 0], [5, 1, 0]], dtype=jnp.int32),
        jnp.array([[0, 1, 5], [0, 3, 3], [0, 5, 1], [2, 1, 3], [2, 3, 1], [4, 1, 1]], dtype=jnp.int32),
        jnp.array(
            [[0, 0, 6], [0, 2, 4], [0, 4, 2], [0, 6, 0], [2, 0, 4], [2, 2, 2], [2, 4, 0], [4, 0, 2], [4, 2, 0], [6, 0, 0]],
            dtype=jnp.int32,
        ),
        jnp.array([[1, 0, 5], [1, 2, 3], [1, 4, 1], [3, 0, 3], [3, 2, 1], [5, 0, 1]], dtype=jnp.int32),
        jnp.array([[0, 2, 4], [0, 4, 2], [0, 6, 0], [2, 0, 4], [2, 4, 0], [4, 0, 2], [4, 2, 0], [6, 0, 0]], dtype=jnp.int32),
        jnp.array([[1, 2, 3], [1, 4, 1], [3, 0, 3], [3, 2, 1], [5, 0, 1]], dtype=jnp.int32),
        jnp.array([[0, 4, 2], [0, 6, 0], [2, 2, 2], [2, 4, 0], [4, 0, 2], [4, 2, 0], [6, 0, 0]], dtype=jnp.int32),
        jnp.array([[1, 4, 1], [3, 2, 1], [5, 0, 1]], dtype=jnp.int32),
        jnp.array([[0, 6, 0], [2, 4, 0], [4, 2, 0], [6, 0, 0]], dtype=jnp.int32),
    )

    def _eval_poly_value_grad_lap(
        coeffs: jax.Array,
        exps: jax.Array,
        x_pows: jax.Array,
        y_pows: jax.Array,
        z_pows: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        ex = exps[:, 0]
        ey = exps[:, 1]
        ez = exps[:, 2]

        val_terms = coeffs * x_pows[ex] * y_pows[ey] * z_pows[ez]
        val = jnp.sum(val_terms)

        ex_m1 = jnp.where(ex > 0, ex - 1, 0)
        ey_m1 = jnp.where(ey > 0, ey - 1, 0)
        ez_m1 = jnp.where(ez > 0, ez - 1, 0)

        gx_terms = coeffs * ex * x_pows[ex_m1] * y_pows[ey] * z_pows[ez]
        gy_terms = coeffs * ey * x_pows[ex] * y_pows[ey_m1] * z_pows[ez]
        gz_terms = coeffs * ez * x_pows[ex] * y_pows[ey] * z_pows[ez_m1]
        grad = jnp.stack([jnp.sum(gx_terms), jnp.sum(gy_terms), jnp.sum(gz_terms)], axis=0)

        ex_m2 = jnp.where(ex > 1, ex - 2, 0)
        ey_m2 = jnp.where(ey > 1, ey - 2, 0)
        ez_m2 = jnp.where(ez > 1, ez - 2, 0)

        lap_x = coeffs * ex * (ex - 1) * x_pows[ex_m2] * y_pows[ey] * z_pows[ez]
        lap_y = coeffs * ey * (ey - 1) * x_pows[ex] * y_pows[ey_m2] * z_pows[ez]
        lap_z = coeffs * ez * (ez - 1) * x_pows[ex] * y_pows[ey] * z_pows[ez_m2]
        lap = jnp.sum(lap_x + lap_y + lap_z)

        return val, grad, lap

    def _single_val_grad_lap(diff: jnp.ndarray) -> tuple[jax.Array, jax.Array, jax.Array]:
        x, y, z = diff[0], diff[1], diff[2]
        x_pows = jnp.stack([x**0, x, x**2, x**3, x**4, x**5, x**6], axis=0)
        y_pows = jnp.stack([y**0, y, y**2, y**3, y**4, y**5, y**6], axis=0)
        z_pows = jnp.stack([z**0, z, z**2, z**3, z**4, z**5, z**6], axis=0)

        vals_list = []
        grads_list = []
        laps_list = []
        for coeffs, exps in zip(S_L_M_COEFFS, S_L_M_EXPS, strict=True):
            val, grad, lap = _eval_poly_value_grad_lap(coeffs, exps, x_pows, y_pows, z_pows)
            vals_list.append(val)
            grads_list.append(grad)
            laps_list.append(lap)

        vals = jnp.stack(vals_list, axis=0)
        grads = jnp.stack(grads_list, axis=0)
        laps = jnp.stack(laps_list, axis=0)
        return vals, grads, laps

    num_R, num_r, _ = r_R_diffs_uq.shape
    diffs_flat = r_R_diffs_uq.reshape((num_R * num_r, 3), order="C")

    vals_flat, grads_flat, lap_flat = vmap(_single_val_grad_lap)(diffs_flat)

    vals = vals_flat.T.reshape((vals_flat.shape[1], num_R, num_r), order="C")
    grads = grads_flat.transpose(1, 0, 2).reshape((grads_flat.shape[1], num_R, num_r, 3), order="C")
    laps = lap_flat.T.reshape((lap_flat.shape[1], num_R, num_r), order="C")
    return vals, grads, laps


@jit
def _compute_AOs_laplacian_analytic_cart(aos_data: AOs_cart_data, r_carts: jnp.ndarray) -> jax.Array:
    """Analytic Laplacian for Cartesian AOs (contracted)."""
    r_carts = jnp.asarray(r_carts)
    R_carts = aos_data._atomic_center_carts_prim_jnp
    c = aos_data._coefficients_jnp
    Z = aos_data._exponents_jnp
    l = aos_data._angular_momentums_prim_jnp
    nx = aos_data._polynominal_order_x_prim_jnp
    ny = aos_data._polynominal_order_y_prim_jnp
    nz = aos_data._polynominal_order_z_prim_jnp

    N_fact = aos_data._normalization_factorial_ratio_prim_jnp
    N_Z = (2.0 * Z / jnp.pi) ** (3.0 / 2.0) * (8.0 * Z) ** l
    N = jnp.sqrt(N_Z * N_fact)

    diff = r_carts[None, :, :] - R_carts[:, None, :]
    x, y, z = diff[..., 0], diff[..., 1], diff[..., 2]
    x = x + EPS_stabilizing_jax_AO_cart_deriv
    y = y + EPS_stabilizing_jax_AO_cart_deriv
    z = z + EPS_stabilizing_jax_AO_cart_deriv
    r2 = jnp.sum(diff**2, axis=-1)
    pref = c[:, None] * jnp.exp(-Z[:, None] * r2)

    def _pow(base, exp):
        return jnp.where(exp[:, None] == 0, 1.0, base ** exp[:, None])

    px, py, pz = _pow(x, nx), _pow(y, ny), _pow(z, nz)
    phi = N[:, None] * pref * px * py * pz

    def _second_component(base, n):
        safe_div = jnp.where(base != 0.0, n[:, None] / base, 0.0)
        safe_div2 = jnp.where(base != 0.0, n[:, None] / (base**2), 0.0)
        a = safe_div - 2.0 * Z[:, None] * base
        return phi * (a**2 - safe_div2 - 2.0 * Z[:, None])

    lap_dup = _second_component(x, nx) + _second_component(y, ny) + _second_component(z, nz)

    orbital_indices = aos_data._orbital_indices_jnp
    num_segments = aos_data.num_ao
    lap = jax.ops.segment_sum(lap_dup, orbital_indices, num_segments=num_segments)
    return lap


@jit
def _compute_AOs_laplacian_analytic_sphe(aos_data: AOs_sphe_data, r_carts: jnp.ndarray) -> jax.Array:
    """Analytic Laplacian for spherical AOs (contracted)."""
    r_carts = jnp.asarray(r_carts)
    nucleus_index_prim_jnp = aos_data._nucleus_index_prim_jnp
    R_carts_jnp = aos_data._atomic_center_carts_prim_jnp
    R_carts_unique_jnp = aos_data._atomic_center_carts_unique_jnp
    c_jnp = aos_data._coefficients_jnp
    Z_jnp = aos_data._exponents_jnp
    l_jnp = aos_data._angular_momentums_prim_jnp
    m_jnp = aos_data._magnetic_quantum_numbers_prim_jnp

    l_f64 = l_jnp.astype(jnp.float64)
    Z_f64 = Z_jnp.astype(jnp.float64)
    factorial_l_plus_1 = jnp.exp(jscipy.special.gammaln(l_f64 + 2.0))
    factorial_2l_plus_2 = jnp.exp(jscipy.special.gammaln(2.0 * l_f64 + 3.0))

    N_n_dup = jnp.sqrt(
        (2.0 ** (2 * l_f64 + 3) * factorial_l_plus_1 * (2 * Z_f64) ** (l_f64 + 1.5)) / (factorial_2l_plus_2 * jnp.sqrt(jnp.pi))
    )
    N_l_m_dup = jnp.sqrt((2 * l_f64 + 1) / (4 * jnp.pi))

    r_R_diffs = r_carts[None, :, :] - R_carts_jnp[:, None, :]
    r_squared = jnp.sum(r_R_diffs**2, axis=-1)
    R_n_dup = c_jnp[:, None] * jnp.exp(-Z_jnp[:, None] * r_squared)

    r_R_diffs_uq = r_carts[None, :, :] - R_carts_unique_jnp[:, None, :]
    S_l_m_vals_all, S_l_m_grads_all, S_l_m_laps_all = _compute_S_l_m_and_grad_lap(r_R_diffs_uq)
    max_ml = S_l_m_vals_all.shape[0]

    S_l_m_vals_flat = S_l_m_vals_all.reshape((max_ml * S_l_m_vals_all.shape[1], S_l_m_vals_all.shape[2]), order="F")
    S_l_m_grads_flat = S_l_m_grads_all.reshape((max_ml * S_l_m_grads_all.shape[1], S_l_m_grads_all.shape[2], 3), order="F")
    S_l_m_laps_flat = S_l_m_laps_all.reshape((max_ml * S_l_m_laps_all.shape[1], S_l_m_laps_all.shape[2]), order="F")

    global_l_m_index = l_jnp**2 + (m_jnp + l_jnp)
    global_R_l_m_index = nucleus_index_prim_jnp * max_ml + global_l_m_index
    S_l_m_dup = S_l_m_vals_flat[global_R_l_m_index]
    S_l_m_grad_dup = S_l_m_grads_flat[global_R_l_m_index]
    S_l_m_lap_dup = S_l_m_laps_flat[global_R_l_m_index]

    pref = N_n_dup[:, None] * R_n_dup * N_l_m_dup[:, None]
    AOs_dup = pref * S_l_m_dup

    grad_S_dot_r = jnp.sum(S_l_m_grad_dup * r_R_diffs, axis=-1)
    lap_dup = (
        pref * S_l_m_lap_dup
        + AOs_dup * (4.0 * (Z_jnp[:, None] ** 2) * r_squared - 6.0 * Z_jnp[:, None])
        - 4.0 * Z_jnp[:, None] * pref * grad_S_dot_r
    )

    orbital_indices = aos_data._orbital_indices_jnp
    num_segments = aos_data.num_ao
    lap = jax.ops.segment_sum(lap_dup, orbital_indices, num_segments=num_segments)
    return lap


def compute_AOs_laplacian(aos_data: AOs_sphe_data | AOs_cart_data, r_carts: jax.Array) -> jax.Array:
    """Return analytic Laplacians of contracted atomic orbitals.

    Dispatches to Cartesian or real-spherical implementations; returns float64 JAX arrays
    (ensure ``jax_enable_x64=True``).

    Args:
        aos_data: ``AOs_cart_data`` or ``AOs_sphe_data`` describing centers, primitive exponents/coefficients,
            angular data, and contraction mapping (run ``sanity_check()`` beforehand).
        r_carts (jax.Array): Electron Cartesian coordinates, shape ``(N_e, 3)`` (Bohr). Casts to ``float64``
            internally via ``jnp.asarray``.

    Returns:
        jax.Array: Laplacians of all contracted AOs, shape ``(num_ao, N_e)``.

    Raises:
        NotImplementedError: If ``aos_data`` is not Cartesian or spherical.
    """
    r_carts = jnp.asarray(r_carts, dtype=jnp.float64)

    if isinstance(aos_data, AOs_cart_data):
        return _compute_AOs_laplacian_analytic_cart(aos_data, r_carts)

    if isinstance(aos_data, AOs_sphe_data):
        return _compute_AOs_laplacian_analytic_sphe(aos_data, r_carts)

    raise NotImplementedError("Analytic Laplacian implemented for Cartesian and spherical AOs only.")


def _compute_S_l_m_debug(
    angular_momentum: int,
    magnetic_quantum_number: int,
    atomic_center_cart: list[float],
    r_cart: list[float],
) -> float:
    r"""Solid harmonics part of a primitve AO.

    Compute the solid harmonics, i.e., r^l * spherical hamonics part (c.f., regular solid harmonics) of a given AO

    Args:
        angular_momentum (int): Angular momentum of the AO, i.e., l
        magnetic_quantum_number (int): Magnetic quantum number of the AO, i.e m = -l .... +l
        atomic_center_cart (list[float]): Center of the nucleus associated to the AO.
        r_cart (list[float]): Cartesian coordinate of an electron

    Returns:
        float: Value of the spherical harmonics part * r^l (i.e., regular solid harmonics).

    Note:
        A real basis of spherical harmonics Y_{l,m} : S^2 -> R can be defined in terms of
        their complex analogues  Y_{l}^{m} : S^2 -> C by setting:
        Y_{l,m}(theta, phi) =
                sqrt(2) * (-1)^m * \Im[Y_l^{|m|}] (if m < 0)
                Y_l^{0} (if m = 0)
                sqrt(2) * (-1)^m * \Re[Y_l^{|m|}] (if m > 0)

        A conversion from cartesian to spherical coordinate is:
                r = sqrt(x**2 + y**2 + z**2)
                theta = arccos(z/r)
                phi = sgn(y)arccos(x/sqrt(x**2+y**2))

        It indicates that there are two singular points
                1) the origin (x,y,z) = (0,0,0)
                2) points on the z axis (0,0,z)

        Therefore, instead, the so-called solid harmonics function is computed, which is defined as
        S_{l,\pm|m|} = \sqrt(\cfrac{4 * np.pi}{2 * l + 1}) * |\vec{R} - \vec{r}|^l [Y_{l,m,\alpha}(\phi, \theta) +- Y_{l,-m,\alpha}(\phi, \theta)].

        The real solid harmonics function are tabulated in many textbooks and websites such as Wikipedia.
        They can be hardcoded into a code, or they can be computed analytically (e.g., https://en.wikipedia.org/wiki/Solid_harmonics).
        The latter one is the strategy employed in this code,
    """
    R_cart = atomic_center_cart
    x, y, z = np.array(r_cart) - np.array(R_cart)
    r_norm = LA.norm(np.array(r_cart) - np.array(R_cart))
    l, m = angular_momentum, magnetic_quantum_number
    m_abs = np.abs(m)

    # solid harmonics for (x,y) dependent part:
    def A_m(x: float, y: float) -> float:
        return np.sum(
            [
                scipy.special.binom(m_abs, p) * x ** (p) * y ** (m_abs - p) * np.cos((m_abs - p) * (np.pi / 2.0))
                for p in range(0, m_abs + 1)
            ]
        )

    def B_m(x: float, y: float) -> float:
        return np.sum(
            [
                scipy.special.binom(m_abs, p) * x ** (p) * y ** (m_abs - p) * np.sin((m_abs - p) * (np.pi / 2.0))
                for p in range(0, m_abs + 1)
            ]
        )

    # solid harmonics for (z) dependent part:
    def lambda_lm(k: int) -> float:
        # logger.devel(f"l={l}, type ={type(l)}")
        return (
            (-1.0) ** (k)
            * 2.0 ** (-l)
            * scipy.special.binom(l, k)
            * scipy.special.binom(2 * l - 2 * k, l)
            * scipy.special.factorial(l - 2 * k)
            / scipy.special.factorial(l - 2 * k - m_abs)
        )

    # solid harmonics for (z) dependent part:
    def Lambda_lm(r_norm: float, z: float) -> float:
        return np.sqrt(
            (2 - int(m_abs == 0)) * scipy.special.factorial(l - m_abs) / scipy.special.factorial(l + m_abs)
        ) * np.sum([lambda_lm(k) * r_norm ** (2 * k) * z ** (l - 2 * k - m_abs) for k in range(0, int((l - m_abs) / 2) + 1)])

    # solid harmonics eveluated in Cartesian coord. (x,y,z):
    if m >= 0:
        gamma = Lambda_lm(r_norm, z) * A_m(x, y)
    if m < 0:
        gamma = Lambda_lm(r_norm, z) * B_m(x, y)
    return gamma


@jit
def _compute_AOs_laplacian_autodiff(aos_data: AOs_sphe_data | AOs_cart_data, r_carts: jnpt.ArrayLike) -> jax.Array:
    """Compute laplacians of the give AOs at r_carts.

    See compute_AOs_laplacian_api

    """
    # not very fast, but it works.
    ao_matrix_hessian = hessian(compute_AOs, argnums=1)(aos_data, r_carts)
    ao_matrix_laplacian = jnp.einsum("m i i u i u -> mi", ao_matrix_hessian)

    return ao_matrix_laplacian


def _compute_AOs_laplacian_debug(
    aos_data: AOs_sphe_data | AOs_cart_data, r_carts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute laplacians of the give AOs at r_carts.

    The method is for computing the laplacians of the given atomic orbital at r_carts
    using the FDM method for debuging purpose.

    Args:
        ao_datas (AOs_data): an instance of AOs_data
        r_carts (npt.NDArray[np.float64]): Cartesian coordinates of electrons (dim: N_e, 3)
        debug_flag (bool): if True, numerical derivatives are computed for debuging purpose

    Returns:
        npt.NDArray[np.float64]:
            Array containing laplacians of the AOs at r_carts. The dim. is (num_ao, N_e)

    """
    # Laplacians of AOs (numerical)
    diff_h = 1.0e-5

    ao_matrix = compute_AOs(aos_data, r_carts)

    # laplacians x^2
    diff_p_x_r_carts = r_carts.copy()
    diff_p_x_r_carts[:, 0] += diff_h
    ao_matrix_diff_p_x = compute_AOs(aos_data, diff_p_x_r_carts)
    diff_m_x_r_carts = r_carts.copy()
    diff_m_x_r_carts[:, 0] -= diff_h
    ao_matrix_diff_m_x = compute_AOs(aos_data, diff_m_x_r_carts)

    # laplacians y^2
    diff_p_y_r_carts = r_carts.copy()
    diff_p_y_r_carts[:, 1] += diff_h
    ao_matrix_diff_p_y = compute_AOs(aos_data, diff_p_y_r_carts)
    diff_m_y_r_carts = r_carts.copy()
    diff_m_y_r_carts[:, 1] -= diff_h
    ao_matrix_diff_m_y = compute_AOs(aos_data, diff_m_y_r_carts)

    # laplacians z^2
    diff_p_z_r_carts = r_carts.copy()
    diff_p_z_r_carts[:, 2] += diff_h
    ao_matrix_diff_p_z = compute_AOs(aos_data, diff_p_z_r_carts)
    diff_m_z_r_carts = r_carts.copy()
    diff_m_z_r_carts[:, 2] -= diff_h
    ao_matrix_diff_m_z = compute_AOs(aos_data, diff_m_z_r_carts)

    ao_matrix_grad2_x = (ao_matrix_diff_p_x + ao_matrix_diff_m_x - 2 * ao_matrix) / (diff_h) ** 2
    ao_matrix_grad2_y = (ao_matrix_diff_p_y + ao_matrix_diff_m_y - 2 * ao_matrix) / (diff_h) ** 2
    ao_matrix_grad2_z = (ao_matrix_diff_p_z + ao_matrix_diff_m_z - 2 * ao_matrix) / (diff_h) ** 2

    ao_matrix_laplacian = ao_matrix_grad2_x + ao_matrix_grad2_y + ao_matrix_grad2_z

    if ao_matrix_laplacian.shape != (aos_data.num_ao, len(r_carts)):
        logger.error(
            f"ao_matrix_laplacian.shape = {ao_matrix_laplacian.shape} is \
                inconsistent with the expected one = {aos_data.num_ao, len(r_carts)}"
        )
        raise ValueError

    return ao_matrix_laplacian


@jit
def _compute_AOs_grad_analytic_cart(aos_data: AOs_cart_data, r_carts: jnp.ndarray) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Analytic gradients for Cartesian AOs (contracted)."""
    r_carts = jnp.asarray(r_carts)
    R_carts = aos_data._atomic_center_carts_prim_jnp
    c = aos_data._coefficients_jnp
    Z = aos_data._exponents_jnp
    l = aos_data._angular_momentums_prim_jnp
    nx = aos_data._polynominal_order_x_prim_jnp
    ny = aos_data._polynominal_order_y_prim_jnp
    nz = aos_data._polynominal_order_z_prim_jnp

    N_fact = aos_data._normalization_factorial_ratio_prim_jnp
    N_Z = (2.0 * Z / jnp.pi) ** (3.0 / 2.0) * (8.0 * Z) ** l
    N = jnp.sqrt(N_Z * N_fact)

    diff = r_carts[None, :, :] - R_carts[:, None, :]
    x, y, z = diff[..., 0], diff[..., 1], diff[..., 2]
    x = x + EPS_stabilizing_jax_AO_cart_deriv
    y = y + EPS_stabilizing_jax_AO_cart_deriv
    z = z + EPS_stabilizing_jax_AO_cart_deriv
    r2 = jnp.sum(diff**2, axis=-1)
    pref = c[:, None] * jnp.exp(-Z[:, None] * r2)

    def _pow(base, exp):
        return jnp.where(exp[:, None] == 0, 1.0, base ** exp[:, None])

    px, py, pz = _pow(x, nx), _pow(y, ny), _pow(z, nz)
    phi = N[:, None] * pref * px * py * pz

    def _grad_component(base, n):
        safe_div = jnp.where(base != 0.0, n[:, None] / base, 0.0)
        return phi * (safe_div - 2.0 * Z[:, None] * base)

    gx_dup = _grad_component(x, nx)
    gy_dup = _grad_component(y, ny)
    gz_dup = _grad_component(z, nz)

    orbital_indices = aos_data._orbital_indices_jnp
    num_segments = aos_data.num_ao
    gx = jax.ops.segment_sum(gx_dup, orbital_indices, num_segments=num_segments)
    gy = jax.ops.segment_sum(gy_dup, orbital_indices, num_segments=num_segments)
    gz = jax.ops.segment_sum(gz_dup, orbital_indices, num_segments=num_segments)

    return gx, gy, gz


@jit
def _compute_AOs_grad_analytic_sphe(aos_data: AOs_sphe_data, r_carts: jnp.ndarray) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Analytic gradients for spherical AOs (contracted)."""
    r_carts = jnp.asarray(r_carts)
    nucleus_index_prim_jnp = aos_data._nucleus_index_prim_jnp
    R_carts_jnp = aos_data._atomic_center_carts_prim_jnp
    R_carts_unique_jnp = aos_data._atomic_center_carts_unique_jnp
    c_jnp = aos_data._coefficients_jnp
    Z_jnp = aos_data._exponents_jnp
    l_jnp = aos_data._angular_momentums_prim_jnp
    m_jnp = aos_data._magnetic_quantum_numbers_prim_jnp

    l_f64 = l_jnp.astype(jnp.float64)
    Z_f64 = Z_jnp.astype(jnp.float64)
    factorial_l_plus_1 = jnp.exp(jscipy.special.gammaln(l_f64 + 2.0))
    factorial_2l_plus_2 = jnp.exp(jscipy.special.gammaln(2.0 * l_f64 + 3.0))

    N_n_dup = jnp.sqrt(
        (2.0 ** (2 * l_f64 + 3) * factorial_l_plus_1 * (2 * Z_f64) ** (l_f64 + 1.5)) / (factorial_2l_plus_2 * jnp.sqrt(jnp.pi))
    )
    N_l_m_dup = jnp.sqrt((2 * l_f64 + 1) / (4 * jnp.pi))

    r_R_diffs = r_carts[None, :, :] - R_carts_jnp[:, None, :]
    r_squared = jnp.sum(r_R_diffs**2, axis=-1)
    R_n_dup = c_jnp[:, None] * jnp.exp(-Z_jnp[:, None] * r_squared)

    r_R_diffs_uq = r_carts[None, :, :] - R_carts_unique_jnp[:, None, :]
    max_ml, S_l_m_dup_all_l_m = _compute_S_l_m(r_R_diffs_uq)
    _, S_l_m_grad_all_l_m, _ = _compute_S_l_m_and_grad_lap(r_R_diffs_uq)

    S_l_m_dup_all_l_m_reshaped = S_l_m_dup_all_l_m.reshape(
        (S_l_m_dup_all_l_m.shape[0] * S_l_m_dup_all_l_m.shape[1], S_l_m_dup_all_l_m.shape[2]), order="F"
    )
    S_l_m_grad_all_l_m_reshaped = S_l_m_grad_all_l_m.reshape(
        (
            S_l_m_grad_all_l_m.shape[0] * S_l_m_grad_all_l_m.shape[1],
            S_l_m_grad_all_l_m.shape[2],
            S_l_m_grad_all_l_m.shape[3],
        ),
        order="F",
    )

    global_l_m_index = l_jnp**2 + (m_jnp + l_jnp)
    global_R_l_m_index = nucleus_index_prim_jnp * max_ml + global_l_m_index
    S_l_m_dup = S_l_m_dup_all_l_m_reshaped[global_R_l_m_index]
    S_l_m_grad_dup = S_l_m_grad_all_l_m_reshaped[global_R_l_m_index]

    pref = N_n_dup[:, None] * R_n_dup * N_l_m_dup[:, None]
    AOs_dup = pref * S_l_m_dup

    grad_from_R = AOs_dup[..., None] * (-2.0 * Z_jnp[:, None, None] * r_R_diffs)
    grad_from_S = pref[..., None] * S_l_m_grad_dup
    grad_dup = grad_from_R + grad_from_S

    orbital_indices = aos_data._orbital_indices_jnp
    num_segments = aos_data.num_ao
    gx = jax.ops.segment_sum(grad_dup[..., 0], orbital_indices, num_segments=num_segments)
    gy = jax.ops.segment_sum(grad_dup[..., 1], orbital_indices, num_segments=num_segments)
    gz = jax.ops.segment_sum(grad_dup[..., 2], orbital_indices, num_segments=num_segments)

    return gx, gy, gz


def compute_AOs_grad(aos_data: AOs_sphe_data | AOs_cart_data, r_carts: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return analytic Cartesian gradients of contracted atomic orbitals.

    Public gradient API used for drift vectors and kinetic terms. Dispatches to Cartesian or real-spherical
    backends; returns float64 JAX arrays (ensure ``jax_enable_x64=True``).

    Args:
        aos_data: ``AOs_cart_data`` or ``AOs_sphe_data`` describing primitive parameters, angular info,
            contraction mapping, and centers (run ``sanity_check()`` beforehand).
        r_carts (jax.Array): Electron Cartesian coordinates, shape ``(N_e, 3)`` (Bohr). Casts to ``float64``
            internally via ``jnp.asarray``.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: Gradients w.r.t. x, y, z, each of shape ``(num_ao, N_e)``.
        Order is ``(gx, gy, gz)``.

    Raises:
        NotImplementedError: If ``aos_data`` is neither Cartesian nor spherical.
    """
    r_carts = jnp.asarray(r_carts, dtype=jnp.float64)

    if isinstance(aos_data, AOs_cart_data):
        return _compute_AOs_grad_analytic_cart(aos_data, r_carts)

    if isinstance(aos_data, AOs_sphe_data):
        return _compute_AOs_grad_analytic_sphe(aos_data, r_carts)

    raise NotImplementedError("Analytic gradients implemented for Cartesian and spherical AOs only.")


def _compute_AOs_grad_autodiff(
    aos_data: AOs_sphe_data | AOs_cart_data, r_carts: jnpt.ArrayLike
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute Cartesian Gradients of AOs.

    The method is for computing the Carteisan gradients (x,y,z) of
    the given atomic orbital at r_carts

    Args:
        ao_datas(AOs_data): an instance of AOs_data
        r_carts(jnpt.ArrayLike): Cartesian coordinates of electrons (dim: N_e, 3)

    Returns:
        tuple: tuple containing gradients of the AOs at r_carts. (grad_x, grad_y, grad_z).
        The dim. of each matrix is (num_ao, N_e)

    """
    grad_full = jacrev(compute_AOs, argnums=1)(aos_data, r_carts)
    grad_diag = jnp.diagonal(grad_full, axis1=1, axis2=2)
    grad_diag = jnp.swapaxes(grad_diag, 1, 2)
    ao_matrix_grad_x = grad_diag[..., 0]  # (M, N)
    ao_matrix_grad_y = grad_diag[..., 1]  # (M, N)
    ao_matrix_grad_z = grad_diag[..., 2]  # (M, N)

    return ao_matrix_grad_x, ao_matrix_grad_y, ao_matrix_grad_z


# no longer used in the main code
def _compute_AOs_grad_debug(
    aos_data: AOs_sphe_data,
    r_carts: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute Cartesian Gradients of AOs.

    The method is for computing the Carteisan gradients (x,y,z) of
    the given atomic orbital at r_carts using FDM for debugging JAX
    implementations. See compute_AOs_grad_api
    """
    # Gradients of AOs (numerical)
    diff_h = 1.0e-5

    # grad x
    diff_p_x_r_carts = r_carts.copy()
    diff_p_x_r_carts[:, 0] += diff_h
    ao_matrix_diff_p_x = compute_AOs(aos_data, diff_p_x_r_carts)
    diff_m_x_r_carts = r_carts.copy()
    diff_m_x_r_carts[:, 0] -= diff_h
    ao_matrix_diff_m_x = compute_AOs(aos_data, diff_m_x_r_carts)

    # grad y
    diff_p_y_r_carts = r_carts.copy()
    diff_p_y_r_carts[:, 1] += diff_h
    ao_matrix_diff_p_y = compute_AOs(aos_data, diff_p_y_r_carts)
    diff_m_y_r_carts = r_carts.copy()
    diff_m_y_r_carts[:, 1] -= diff_h
    ao_matrix_diff_m_y = compute_AOs(aos_data, diff_m_y_r_carts)

    # grad z
    diff_p_z_r_carts = r_carts.copy()
    diff_p_z_r_carts[:, 2] += diff_h
    ao_matrix_diff_p_z = compute_AOs(aos_data, diff_p_z_r_carts)
    diff_m_z_r_carts = r_carts.copy()
    diff_m_z_r_carts[:, 2] -= diff_h
    ao_matrix_diff_m_z = compute_AOs(aos_data, diff_m_z_r_carts)

    ao_matrix_grad_x = (ao_matrix_diff_p_x - ao_matrix_diff_m_x) / (2.0 * diff_h)
    ao_matrix_grad_y = (ao_matrix_diff_p_y - ao_matrix_diff_m_y) / (2.0 * diff_h)
    ao_matrix_grad_z = (ao_matrix_diff_p_z - ao_matrix_diff_m_z) / (2.0 * diff_h)

    return ao_matrix_grad_x, ao_matrix_grad_y, ao_matrix_grad_z


"""
if __name__ == "__main__":
    import os

    # from functools import partial
    # from jax.experimental import sparse
    from .trexio_wrapper import read_trexio_file

    log = getLogger("jqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)
"""
