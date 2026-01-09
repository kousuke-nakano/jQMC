"""collections of unit tests."""

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

import os
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from jax import numpy as jnp

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc.determinant import Geminal_data  # noqa: E402
from jqmc.hamiltonians import Hamiltonian_data  # noqa: E402
from jqmc.jastrow_factor import (  # noqa: E402
    Jastrow_data,
    Jastrow_NN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.trexio_wrapper import read_trexio_file  # noqa: E402
from jqmc.wavefunction import (  # noqa: E402
    Wavefunction_data,
)

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


test_trexio_files = ["H2_ecp_ccpvtz_cart.h5", "H_ecp_ccpvqz.h5"]


def assert_dataclasses_equal(d1, d2):
    """Helper to compare two dataclasses (or pytrees) with numpy arrays."""
    import dataclasses
    from collections.abc import Mapping

    if dataclasses.is_dataclass(d1) and dataclasses.is_dataclass(d2):
        assert d1.__class__ == d2.__class__
        for field in dataclasses.fields(d1):
            val1 = getattr(d1, field.name)
            val2 = getattr(d2, field.name)
            assert_dataclasses_equal(val1, val2)
    elif isinstance(d1, (list, tuple)) and isinstance(d2, (list, tuple)):
        assert len(d1) == len(d2)
        for v1, v2 in zip(d1, d2, strict=True):
            assert_dataclasses_equal(v1, v2)
    elif isinstance(d1, Mapping) and isinstance(d2, Mapping):
        assert d1.keys() == d2.keys()
        for k in d1:
            assert_dataclasses_equal(d1[k], d2[k])
    elif isinstance(d1, (np.ndarray, jnp.ndarray, jax.Array)) or isinstance(d2, (np.ndarray, jnp.ndarray, jax.Array)):
        # Handle JAX arrays by converting to numpy
        a1 = np.asarray(d1)
        a2 = np.asarray(d2)

        # Check for string arrays (which might be bytes vs str)
        if a1.dtype.kind in ("S", "U") and a2.dtype.kind in ("S", "U"):
            np.testing.assert_array_equal(a1.astype(str), a2.astype(str))
        else:
            np.testing.assert_array_equal(a1, a2)
    else:
        assert d1 == d2


@pytest.mark.parametrize("trexio_file", test_trexio_files)
@pytest.mark.parametrize("use_1b", [True, False])
@pytest.mark.parametrize("use_2b", [True, False])
@pytest.mark.parametrize("use_3b", [True, False])
@pytest.mark.parametrize("use_nn", [True, False])
@pytest.mark.parametrize("geminal_type", ["mo", "ao"])
def test_hamiltonian_hdf5(trexio_file, use_1b, use_2b, use_3b, use_nn, geminal_type, tmp_path):
    """Test HDF5 reading/writing for Hamiltonian_data."""
    (
        structure_data,
        aos_data,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_file), store_tuple=True
    )

    if geminal_type == "ao":
        geminal_data = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)
    else:
        geminal_data = geminal_mo_data

    jastrow_one_body_data = None
    if use_1b:
        # For ECPs, we need core_electrons.
        # The test files seem to be ECP files (H2_ecp..., H_ecp...).
        # Assuming 0 core electrons for Hydrogen if not specified, but let's check if we can deduce it.
        # Or just use dummy values for testing serialization.
        core_electrons = tuple([0.0] * len(structure_data.positions))
        jastrow_one_body_data = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0, structure_data=structure_data, core_electrons=core_electrons
        )

    jastrow_two_body_data = None
    if use_2b:
        jastrow_two_body_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)

    jastrow_three_body_data = None
    if use_3b:
        # We need orb_data. aos_data from read_trexio_file should work.
        # If aos_data is None or not compatible, we might need to create a dummy one.
        # But read_trexio_file should return valid aos_data.
        # However, Jastrow_three_body_data expects AOs_sphe_data, AOs_cart_data or MOs_data.
        # Let's assume aos_data is compatible.
        # If aos_data is None (which shouldn't happen for these files), we create a dummy.
        if aos_data is None:
            raise ValueError("aos_data is required for Jastrow_three_body_data but is None.")
        jastrow_three_body_data = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data)

    nn_jastrow_data = None
    if use_nn:
        nn_jastrow_data = Jastrow_NN_data.init_from_structure(structure_data=structure_data)

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_one_body_data,
        jastrow_two_body_data=jastrow_two_body_data,
        jastrow_three_body_data=jastrow_three_body_data,
        jastrow_nn_data=nn_jastrow_data,
    )

    jastrow_data.sanity_check()

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data)
    wavefunction_data.sanity_check()

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )
    hamiltonian_data.sanity_check()

    # Save to HDF5
    h5_file = tmp_path / "test_hamiltonian.h5"
    hamiltonian_data.save_to_hdf5(str(h5_file))

    # Load from HDF5
    loaded_hamiltonian_data = Hamiltonian_data.load_from_hdf5(str(h5_file))

    # Compare
    # Note: Direct equality (==) on dataclasses with numpy arrays is ambiguous in boolean context.
    # We use a helper to compare leaves.
    assert_dataclasses_equal(hamiltonian_data, loaded_hamiltonian_data)


if __name__ == "__main__":
    from logging import Formatter, StreamHandler, getLogger

    logger = getLogger("jqmc")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
