"""Hamiltonian module."""

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
import collections.abc
import dataclasses
import importlib
from logging import getLogger
from typing import Any, Type, TypeVar, Union

import h5py

# JAX
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import jit
from jax import typing as jnpt

from .coulomb_potential import Coulomb_potential_data, compute_coulomb_potential
from .structure import Structure_data
from .wavefunction import (
    Wavefunction_data,
    _compute_kinetic_energy_auto,
    compute_kinetic_energy,
)

T = TypeVar("T")

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

# separator
num_sep_line = 66


@struct.dataclass
class Hamiltonian_data:
    """Hamiltonian dataclass.

    The class contains data for computing Kinetic and Potential energy terms.

    Args:
        structure_data (Structure_data): an instance of Structure_data
        coulomb_data (Coulomb_data): an instance of Coulomb_data
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data

    Notes:
        Heres are the differentiable arguments, i.e., pytree_node = True
        This information is a little bit tricky in terms of a principle of the object-oriented programming,
        'Don't ask, but tell' (i.e., the Hamiltonian_data knows the details of the other classes
        too much), but there is no other choice to dynamically switch on and off pytree_nodes depending
        on optimized variational parameters chosen by a user because @dataclass is statistically generated.

        WF parameters related:
            - lambda in wavefunction_data.geminal_data (determinant.py)
            - jastrow_2b_param in wavefunction_data.jastrow_data.jastrow_two_body_data (jastrow_factor.py)
            - j_matrix in wavefunction_data.jastrow_data.jastrow_three_body_data (jastrow_factor.py)

        Atomic positions related:
            - positions in hamiltonian_data.structure_data (this file)
            - positions in wavefunction_data.geminal_data.mos_data/aos_data.structure_data (molecular_orbital.py/atomic_orbital.py)
            - positions in wavefunction_data.jastrow_data.jastrow_three_body_data.mos_data/aos_data.structure_data (jastrow_factor.py)
            - positions in Coulomb_potential_data.structure_data (coulomb_potential.py)
    """

    structure_data: Structure_data = struct.field(pytree_node=True, default_factory=lambda: Structure_data())
    coulomb_potential_data: Coulomb_potential_data = struct.field(
        pytree_node=True, default_factory=lambda: Coulomb_potential_data()
    )
    wavefunction_data: Wavefunction_data = struct.field(pytree_node=True, default_factory=lambda: Wavefunction_data())

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        self.structure_data.sanity_check()
        self.coulomb_potential_data.sanity_check()
        self.wavefunction_data.sanity_check()

    def _get_info(self) -> list[str]:
        """Return a list of strings representing the logged information."""
        info_lines = []
        # Add the top separator line.
        info_lines.append("=" * num_sep_line)
        # Replace attribute logger_info() calls with their get_info() outputs.
        info_lines.extend(self.structure_data._get_info())
        info_lines.extend(self.coulomb_potential_data._get_info())
        info_lines.extend(self.wavefunction_data._get_info())
        # Add the bottom separator line.
        info_lines.append("=" * num_sep_line)
        return info_lines

    def _logger_info(self) -> None:
        """Log the information from get_info() using logger.info."""
        for line in self._get_info():
            logger.info(line)

    def accumulate_position_grad(self, grad_hamiltonian: "Hamiltonian_data"):
        """Aggregate position gradients from Hamiltonian components (structure + wavefunction)."""
        grad = grad_hamiltonian.structure_data.positions
        grad += grad_hamiltonian.coulomb_potential_data.structure_data.positions
        if self.wavefunction_data is not None and grad_hamiltonian.wavefunction_data is not None:
            grad += self.wavefunction_data.accumulate_position_grad(grad_hamiltonian.wavefunction_data)
        return grad

    def save_to_hdf5(self, filepath="jqmc.h5") -> None:
        """Save Hamiltonian data to an HDF5 file.

        Args:
            filepath (str, optional): file path
        """
        with h5py.File(filepath, "w") as f:
            _save_dataclass_to_hdf5(f, self)

    @staticmethod
    def load_from_hdf5(filepath="jqmc.h5") -> "Hamiltonian_data":
        """Load Hamiltonian data from an HDF5 file.

        Args:
            filepath (str, optional): file path

        Returns:
            Hamiltonian_data: An instance of Hamiltonian_data.
        """
        with h5py.File(filepath, "r") as f:
            return _load_dataclass_from_hdf5(Hamiltonian_data, f)


def compute_local_energy(
    hamiltonian_data: Hamiltonian_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
    RT: jnpt.ArrayLike,
) -> float:
    """Compute Local Energy.

    The method is for computing the local energy at (r_up_carts, r_dn_carts).

    Args:
        hamiltonian_data (Hamiltonian_data):
            an instance of Hamiltonian_data
        r_up_carts (jnpt.ArrayLike):
            Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (jnpt.ArrayLike):
            Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        RT (jnpt.ArrayLike):
            Rotation matrix. equiv R.T used for non-local part. It does not affect all-electron calculations.

    Returns:
        float: The value of local energy (e_L) with the given wavefunction (float)
    """
    T = compute_kinetic_energy(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    V = compute_coulomb_potential(
        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        RT=RT,
        wavefunction_data=hamiltonian_data.wavefunction_data,
    )

    return T + V


@jit
def _compute_local_energy_auto(
    hamiltonian_data: Hamiltonian_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
    RT: jnpt.ArrayLike,
) -> float:
    """Compute Local Energy.

    The method is for computing the local energy at (r_up_carts, r_dn_carts).

    Args:
        hamiltonian_data (Hamiltonian_data):
            an instance of Hamiltonian_data
        r_up_carts (jnpt.ArrayLike):
            Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (jnpt.ArrayLike):
            Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        RT (jnpt.ArrayLike):
            Rotation matrix. equiv R.T used for non-local part. It does not affect all-electron calculations.

    Returns:
        float: The value of local energy (e_L) with the given wavefunction (float)
    """
    T = _compute_kinetic_energy_auto(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    V = compute_coulomb_potential(
        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        RT=RT,
        wavefunction_data=hamiltonian_data.wavefunction_data,
    )

    return T + V


def _reconstruct_dataclass(cls, obj):
    """Restore a dataclass instance from an object.

    Reconstructs an instance of `cls` from `obj`, handling missing/extra fields.
    This is useful for backward compatibility when loading pickled objects
    where the class definition might have changed (e.g. new fields added).

    """
    if not dataclasses.is_dataclass(cls):
        return obj

    kwargs = {}
    for field in dataclasses.fields(cls):
        field_name = field.name
        if hasattr(obj, field_name):
            val = getattr(obj, field_name)

            # Recursively reconstruct if the field type is a dataclass
            # This handles nested dataclasses that might also have changed.
            # We check if field.type is a class and is a dataclass.
            # Note: This assumes field.type is the actual class, not a string.
            if isinstance(field.type, type) and dataclasses.is_dataclass(field.type):
                val = _reconstruct_dataclass(field.type, val)

            kwargs[field_name] = val
        # If field is missing in obj, we skip it so cls() uses its default value/factory.

    return cls(**kwargs)


def _save_item(group: h5py.Group, name: str, value: Any) -> None:
    """Helper to save an item to HDF5 group."""
    if value is None:
        return

    if hasattr(value, "device_buffer") or isinstance(value, jnp.ndarray):
        value = np.array(value)

    if isinstance(value, np.ndarray):
        group.create_dataset(name, data=value)
    elif dataclasses.is_dataclass(value):
        subgroup = group.create_group(name)
        _save_dataclass_to_hdf5(subgroup, value)
    elif isinstance(value, (dict, collections.abc.Mapping)):
        subgroup = group.create_group(name)
        subgroup.attrs["_is_dict"] = True
        for k, v in value.items():
            _save_item(subgroup, str(k), v)
    elif isinstance(value, (list, tuple)):
        if len(value) > 0:
            if all(isinstance(v, (int, float, bool, str, np.number, np.bool_)) for v in value):
                group.create_dataset(name, data=value)
            elif all(isinstance(v, (np.ndarray, jnp.ndarray)) for v in value):
                subgroup = group.create_group(name)
                subgroup.attrs["_is_list"] = True
                for i, v in enumerate(value):
                    v_np = np.array(v) if hasattr(v, "device_buffer") or isinstance(v, jnp.ndarray) else v
                    subgroup.create_dataset(str(i), data=v_np)
            else:
                subgroup = group.create_group(name)
                subgroup.attrs["_is_list"] = True
                for i, v in enumerate(value):
                    _save_item(subgroup, str(i), v)
        else:
            group.create_dataset(name, data=np.array([]))
    elif isinstance(value, (int, float, bool, str)):
        group.attrs[name] = value
    else:
        try:
            group.attrs[name] = value
        except Exception:
            pass


def _save_dataclass_to_hdf5(group: h5py.Group, obj: Any) -> None:
    """Recursively save a dataclass to an HDF5 group.

    Args:
        group (h5py.Group): The HDF5 group to save to.
        obj (Any): The dataclass instance to save.
    """
    if not dataclasses.is_dataclass(obj):
        raise ValueError(f"Object {obj} is not a dataclass.")

    # Save class name for verification/reconstruction
    group.attrs["_class_name"] = obj.__class__.__name__
    group.attrs["_module_name"] = obj.__class__.__module__

    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        _save_item(group, field.name, value)


def _load_item(item: Union[h5py.Group, h5py.Dataset, Any]) -> Any:
    """Helper to load an item from HDF5."""
    if isinstance(item, h5py.Dataset):
        val = item[()]
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        elif isinstance(val, np.ndarray):
            if val.dtype.kind == "S":
                val = np.char.decode(val, "utf-8")
            elif val.dtype.kind == "O" and val.size > 0 and isinstance(val.flat[0], bytes):
                val = np.array([v.decode("utf-8") for v in val.flat]).reshape(val.shape)
        return val
    elif isinstance(item, h5py.Group):
        if item.attrs.get("_is_list"):
            lst = []
            # Combine keys from subgroups/datasets and attributes
            all_keys = set(item.keys())
            for k in item.attrs.keys():
                if k.isdigit():
                    all_keys.add(k)

            sorted_keys = sorted(all_keys, key=int)
            for k in sorted_keys:
                if k in item:
                    lst.append(_load_item(item[k]))
                elif k in item.attrs:
                    lst.append(item.attrs[k])
            return lst
        elif item.attrs.get("_is_dict"):
            d = {}
            for k in item.keys():
                d[k] = _load_item(item[k])
            return d
        else:
            # Dataclass or generic group
            class_name = item.attrs.get("_class_name")
            module_name = item.attrs.get("_module_name")
            if class_name and module_name:
                module = importlib.import_module(module_name)
                sub_cls = getattr(module, class_name)
                return _load_dataclass_from_hdf5(sub_cls, item)
            else:
                # Fallback for dicts saved without _is_dict or unknown structures
                d = {}
                for k in item.keys():
                    d[k] = _load_item(item[k])
                return d
    return item


def _load_dataclass_from_hdf5(cls: Type[T], group: h5py.Group) -> T:
    """Recursively load a dataclass from an HDF5 group.

    Args:
        cls (Type[T]): The class to reconstruct.
        group (h5py.Group): The HDF5 group to load from.

    Returns:
        T: The reconstructed dataclass instance.
    """
    init_args = {}

    for field in dataclasses.fields(cls):
        if field.name in group:
            val = _load_item(group[field.name])

            # Type conversion for list/tuple/array
            if isinstance(val, np.ndarray) and val.size == 0:
                if field.default_factory is list:
                    val = []
                elif field.default_factory is tuple:
                    val = ()
            elif (
                isinstance(val, np.ndarray)
                and (field.type is list or field.type is tuple or "list" in str(field.type) or "tuple" in str(field.type))
                and not ("Array" in str(field.type) or "ndarray" in str(field.type))
            ):
                val = val.tolist()
                if isinstance(field.type, type) and issubclass(field.type, tuple):
                    val = tuple(val)
            elif isinstance(val, list) and (field.type is tuple or "tuple" in str(field.type)):
                val = tuple(val)

            init_args[field.name] = val
        elif field.name in group.attrs:
            val = group.attrs[field.name]
            if field.type is bool:
                val = bool(val)
            init_args[field.name] = val

    # Check for missing fields and fill with defaults
    for field in dataclasses.fields(cls):
        if field.name not in init_args:
            if field.default is not dataclasses.MISSING:
                init_args[field.name] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                init_args[field.name] = field.default_factory()

    obj = cls(**init_args)

    # Special handling for NN_Jastrow_data to reconstruct nn_def
    if cls.__name__ == "NN_Jastrow_data":
        # Reconstruct nn_def
        from .jastrow_factor import NNJastrow

        if (
            hasattr(obj, "species_lookup")
            and obj.species_lookup is not None
            and hasattr(obj, "num_species")
            and obj.num_species > 0
        ):
            species_lookup = tuple(int(x) for x in obj.species_lookup)

            object.__setattr__(
                obj,
                "nn_def",
                NNJastrow(
                    hidden_dim=obj.hidden_dim,
                    num_layers=obj.num_layers,
                    num_rbf=obj.num_rbf,
                    cutoff=obj.cutoff,
                    species_lookup=species_lookup,
                    num_species=obj.num_species,
                ),
            )
            obj.__post_init__()

    return obj


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
