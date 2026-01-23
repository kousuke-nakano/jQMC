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

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc.setting import (  # noqa: E402
    decimal_debug_vs_production,
)
from jqmc.swct import (  # noqa: E402
    SWCT_data,
    _evaluate_swct_domega_debug,
    _evaluate_swct_omega_debug,
    evaluate_swct_domega,
    evaluate_swct_omega,
)
from jqmc.trexio_wrapper import read_trexio_file  # noqa: E402

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


# @pytest.mark.skip
def test_debug_and_jax_SWCT_omega():
    """Test SWCT omega, compare debug and jax."""
    (
        structure_data,
        _,
        _,
        _,
        geminal_mo_data,
        _,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"), store_tuple=True
    )

    swct_data = SWCT_data(structure=structure_data)

    num_ele_up = geminal_mo_data.num_electron_up
    num_ele_dn = geminal_mo_data.num_electron_dn
    r_cart_min, r_cart_max = -5.0, +5.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_up, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_dn, 3) + r_cart_min

    omega_up_debug = _evaluate_swct_omega_debug(swct_data=swct_data, r_carts=r_up_carts)
    omega_dn_debug = _evaluate_swct_omega_debug(swct_data=swct_data, r_carts=r_dn_carts)
    omega_up_jax = evaluate_swct_omega(swct_data=swct_data, r_carts=r_up_carts)
    omega_dn_jax = evaluate_swct_omega(swct_data=swct_data, r_carts=r_dn_carts)

    np.testing.assert_almost_equal(omega_up_debug, omega_up_jax, decimal=decimal_debug_vs_production)
    np.testing.assert_almost_equal(omega_dn_debug, omega_dn_jax, decimal=decimal_debug_vs_production)

    domega_up_debug = _evaluate_swct_domega_debug(swct_data=swct_data, r_carts=r_up_carts)
    domega_dn_debug = _evaluate_swct_domega_debug(swct_data=swct_data, r_carts=r_dn_carts)
    domega_up_jax = evaluate_swct_domega(swct_data=swct_data, r_carts=r_up_carts)
    domega_dn_jax = evaluate_swct_domega(swct_data=swct_data, r_carts=r_dn_carts)

    np.testing.assert_almost_equal(domega_up_debug, domega_up_jax, decimal=decimal_debug_vs_production)
    np.testing.assert_almost_equal(domega_dn_debug, domega_dn_jax, decimal=decimal_debug_vs_production)

    jax.clear_caches()


if __name__ == "__main__":
    from logging import Formatter, StreamHandler, getLogger

    logger = getLogger("jqmc")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
