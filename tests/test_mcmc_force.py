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

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import get_tolerance  # noqa: E402
from jqmc.hamiltonians import Hamiltonian_data  # noqa: E402
from jqmc.jastrow_factor import (  # noqa: E402
    Jastrow_data,
    Jastrow_NN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.jqmc_mcmc import MCMC  # noqa: E402
from jqmc.trexio_wrapper import read_trexio_file  # noqa: E402
from jqmc.wavefunction import Wavefunction_data  # noqa: E402

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

test_mcmc_force_params = [
    pytest.param(
        "H2_ecp_ccpvtz_cart.h5",
        {
            "jastrow_1b_param": True,
            "jastrow_2b_param": True,
            "jastrow_3b_param": True,
            "jastrow_nn_param": False,
            "core_electrons": (0, 0),
        },
        id="ecp",
    ),
    pytest.param(
        "H2_ecp_ccpvtz_cart.h5",
        {
            "jastrow_1b_param": True,
            "jastrow_2b_param": True,
            "jastrow_3b_param": True,
            "jastrow_nn_param": True,
            "core_electrons": (0, 0),
        },
        id="ecp-nn",
    ),
    pytest.param(
        "H2_ae_ccpvtz_cart.h5",
        {
            "jastrow_1b_param": True,
            "jastrow_2b_param": True,
            "jastrow_3b_param": False,
            "jastrow_nn_param": False,
            "core_electrons": (0, 0),
        },
        id="ae",
    ),
    pytest.param(
        "H2_ae_ccpvtz_cart.h5",
        {
            "jastrow_1b_param": True,
            "jastrow_2b_param": True,
            "jastrow_3b_param": False,
            "jastrow_nn_param": True,
            "core_electrons": (0, 0),
        },
        id="ae-nn",
    ),
]


@pytest.mark.parametrize("trexio_file, jastrow_parameters", test_mcmc_force_params)
def test_mcmc_force_with_SWCT(trexio_file: str, jastrow_parameters: dict):
    """Test MCMC force with SWCT."""
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

    # Jastrow setup from parameters
    jastrow_1b_param = jastrow_parameters.get("jastrow_1b_param", False)
    jastrow_2b_param = jastrow_parameters.get("jastrow_2b_param", False)
    jastrow_3b_param = jastrow_parameters.get("jastrow_3b_param", False)

    if jastrow_1b_param:
        core_electrons = jastrow_parameters.get("core_electrons", (0, 0))
        jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=0.5, structure_data=structure_data, core_electrons=tuple(core_electrons), jastrow_1b_type="pade"
        )
    else:
        jastrow_onebody_data = None

    if jastrow_2b_param:
        jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="exp")
    else:
        jastrow_twobody_data = None

    if jastrow_3b_param:
        jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=aos_data, random_init=True, random_scale=1.0e-3
        )
    else:
        jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data)

    jastrow_nn_param = jastrow_parameters.get("jastrow_nn_param", False)
    if jastrow_nn_param:
        jastrow_nn_data = Jastrow_NN_data.init_from_structure(
            structure_data=structure_data, hidden_dim=5, num_layers=2, cutoff=5.0
        )
    else:
        jastrow_nn_data = None

    # define data
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
        jastrow_nn_data=jastrow_nn_data,
    )

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    # VMC parameters
    num_mcmc_warmup_steps = 5
    num_mcmc_bin_blocks = 5
    mcmc_seed = 34356

    # run VMC
    mcmc = MCMC(
        hamiltonian_data=hamiltonian_data,
        Dt=2.0,
        mcmc_seed=mcmc_seed,
        num_walkers=2,
        comput_position_deriv=True,
        comput_log_WF_param_deriv=False,
        comput_e_L_param_deriv=False,
        epsilon_AS=1.0e-2,
    )
    mcmc.run(num_mcmc_steps=20)
    mcmc.get_E(
        num_mcmc_warmup_steps=num_mcmc_warmup_steps,
        num_mcmc_bin_blocks=num_mcmc_bin_blocks,
    )
    force_mean, force_std = mcmc.get_aF(
        num_mcmc_warmup_steps=num_mcmc_warmup_steps,
        num_mcmc_bin_blocks=num_mcmc_bin_blocks,
    )

    # See [J. Chem. Phys. 156, 034101 (2022)]
    atol, rtol = get_tolerance("mcmc", "strict")
    assert not np.any(np.isnan(np.asarray(np.array(force_mean[0])))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(-1.0 * np.array(force_mean[1])))), "NaN detected in second argument"
    np.testing.assert_allclose(
        np.array(force_mean[0]),
        -1.0 * np.array(force_mean[1]),
        atol=atol,
        rtol=rtol,
    )
    assert not np.any(np.isnan(np.asarray(np.array(force_std[0])))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(np.array(force_std[1])))), "NaN detected in second argument"
    np.testing.assert_allclose(
        np.array(force_std[0]),
        np.array(force_std[1]),
        atol=atol,
        rtol=rtol,
    )


def test_mcmc_force_without_SWCT():
    """Test MCMC force without SWCT (use_swct=False)."""
    trexio_file = "H2_ecp_ccpvtz_cart.h5"
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

    jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=0.5, structure_data=structure_data, core_electrons=(0, 0), jastrow_1b_type="pade"
    )
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="exp")
    jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data)
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
        jastrow_nn_data=None,
    )
    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    mcmc = MCMC(
        hamiltonian_data=hamiltonian_data,
        Dt=2.0,
        mcmc_seed=34356,
        num_walkers=2,
        comput_position_deriv=True,
        comput_log_WF_param_deriv=False,
        comput_e_L_param_deriv=False,
        epsilon_AS=1.0e-2,
        use_swct=False,
    )
    mcmc.run(num_mcmc_steps=20)
    mcmc.get_E(num_mcmc_warmup_steps=5, num_mcmc_bin_blocks=5)
    force_mean, force_std = mcmc.get_aF(num_mcmc_warmup_steps=5, num_mcmc_bin_blocks=5)

    # Forces should be finite (no NaN/Inf)
    assert not np.any(np.isnan(np.array(force_mean))), "NaN detected in force_mean"
    assert not np.any(np.isnan(np.array(force_std))), "NaN detected in force_std"
    assert np.all(np.isfinite(np.array(force_mean))), "Inf detected in force_mean"


if __name__ == "__main__":
    from logging import Formatter, StreamHandler, getLogger

    logger = getLogger("jqmc")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
