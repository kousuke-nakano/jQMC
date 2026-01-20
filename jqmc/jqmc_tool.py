"""jQMC tools."""


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

import gzip
import inspect
import os
import pickle
import re
import shutil
import sys
import zipfile
from enum import Enum
from logging import Formatter, StreamHandler, getLogger
from typing import List

import click
import matplotlib.pyplot as plt
import numpy as np
import tomlkit
import typer
from uncertainties import ufloat

from .atomic_orbital import AOs_cart_data, AOs_sphe_data
from .determinant import Geminal_data
from .hamiltonians import Hamiltonian_data
from .jastrow_factor import (
    Jastrow_data,
    Jastrow_NN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from .jqmc_miscs import cli_parameters
from .setting import (
    GFMC_MIN_BIN_BLOCKS,
    GFMC_MIN_COLLECT_STEPS,
    GFMC_MIN_WARMUP_STEPS,
    MCMC_MIN_BIN_BLOCKS,
    MCMC_MIN_WARMUP_STEPS,
    Bohr_to_Angstrom,
)
from .trexio_wrapper import read_trexio_file
from .wavefunction import Wavefunction_data

log = getLogger("jqmc")
log.setLevel("WARNING")
stream_handler = StreamHandler(sys.stdout)
stream_handler.setLevel("WARNING")
handler_format = Formatter("%(message)s")
stream_handler.setFormatter(handler_format)
log.addHandler(stream_handler)


@click.group()
def _cli():
    """The jQMC tools."""
    pass


# trexio_app
trexio_app = typer.Typer(help="Read and Convert TREXIO files.")


@trexio_app.command("show-info")
def trexio_show_info(
    filename: str = typer.Argument(..., help="TREXIO file name."),
):
    """Show information stored in the TREXIO file."""
    (structure_data, aos_data, mos_data_up, mos_data_dn, geminal_data, coulomb_potential_data) = read_trexio_file(filename)

    for line in structure_data._get_info():
        typer.echo(line)
    for line in aos_data._get_info():
        typer.echo(line)
    for line in mos_data_up._get_info():
        typer.echo(line)
    for line in mos_data_dn._get_info():
        typer.echo(line)
    for line in geminal_data._get_info():
        typer.echo(line)
    for line in coulomb_potential_data._get_info():
        typer.echo(line)


@trexio_app.command("show-detail")
def trexio_show_detail(
    filename: str = typer.Argument(..., help="TREXIO file name."),
):
    """Show information stored in the TREXIO file."""
    (structure_data, aos_data, mos_data_up, mos_data_dn, geminal_data, coulomb_potential_data) = read_trexio_file(filename)

    typer.echo(structure_data)
    typer.echo(aos_data)
    typer.echo(mos_data_up)
    typer.echo(mos_data_dn)
    typer.echo(geminal_data)
    typer.echo(coulomb_potential_data)


class orbital_type(str, Enum):
    """Orbital type."""

    ao = "ao"
    mo = "mo"
    ao_full = "ao-full"
    ao_small = "ao-small"
    ao_medium = "ao-medium"
    ao_large = "ao-large"
    none = "none"


def _get_nn_jastrow_help_msg() -> str:
    """Generate help message for NN Jastrow parameters dynamically."""
    try:
        sig = inspect.signature(Jastrow_NN_data.init_from_structure)
        params_list = []
        for name, param in sig.parameters.items():
            if name in ("structure_data", "key", "cls"):
                continue
            # Extract type name if possible
            type_name = getattr(param.annotation, "__name__", str(param.annotation)).replace("builtins.", "")
            # Format: name (type, default=value)
            params_list.append(f"{name} ({type_name}, default={param.default})")

        return (
            f"Parameters for NN Jastrow. Specify as 'key=value'. "
            f"Use multiple flags for multiple parameters (e.g. -jp hidden_dim=64 -jp num_layers=5). "
            f"Supported params for 'schnet': {', '.join(params_list)}."
        )
    except Exception:
        return "Parameters for NN Jastrow. Specify as 'key=value'. Can be used multiple times."


@trexio_app.command("convert-to")
def trexio_convert_to(
    trexio_file: str = typer.Argument(..., help="TREXIO filename."),
    hamiltonian_file: str = typer.Option("hamiltonian_data.h5", "-o", "--output", help="Output file name."),
    j1_parmeter: float = typer.Option(None, "-j1", "--jastrow-1b-parameter", help="Jastrow one-body parameter."),
    j2_parmeter: float = typer.Option(None, "-j2", "--jastrow-2b-parameter", help="Jastrow two-body parameter."),
    j3_basis_type: orbital_type = typer.Option(
        orbital_type.none,
        "-j3",
        "--jastrow-3b-basis-set-type",
        help="Jastrow three-body basis-set type (use 'none' to disable atomic/molecular-orbital-based J3 term).",
    ),
    j_nn_type: str = typer.Option(
        None,
        "-j-nn-type",
        "--jastrow-nn-type",
        help="NN Jastrow type (e.g. 'schnet'). If set, an NN-based Jastrow term is added.",
    ),
    j_nn_params: List[str] = typer.Option(
        None,
        "-jp",
        "--jastrow-nn-param",
        help=_get_nn_jastrow_help_msg(),
    ),
):
    """Convert a TREXIO file to hamiltonian_data."""
    # Allow direct string inputs when trexio_convert_to is called programmatically (e.g., in tests)
    if isinstance(j3_basis_type, str):
        try:
            j3_basis_type = orbital_type(j3_basis_type)
        except ValueError:
            # Leave as-is; downstream validation will raise a clearer error message.
            pass

    if isinstance(j_nn_type, typer.models.OptionInfo):
        j_nn_type = j_nn_type.default

    (structure_data, aos_data, mos_data, _, geminal_data, coulomb_potential_data) = read_trexio_file(
        trexio_file, store_tuple=True
    )

    if j1_parmeter is not None:
        if coulomb_potential_data.ecp_flag:
            core_electrons = coulomb_potential_data.z_cores
        else:
            core_electrons = [0] * structure_data.natom
        jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=j1_parmeter, structure_data=structure_data, core_electrons=core_electrons
        )
    else:
        jastrow_onebody_data = None
    if j2_parmeter is not None:
        jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=j2_parmeter)
    else:
        jastrow_twobody_data = None
    if j3_basis_type is None:
        j3_choice = None
    else:
        j3_choice = getattr(j3_basis_type, "value", j3_basis_type)

    if j3_choice == "none":
        j3_choice = None

    if j3_choice is not None:
        if j3_choice in {"ao", "ao-full", "ao-small", "ao-medium", "ao-large"}:
            selected_ao_indices_total = []

            # 1) Loop over each nucleus in the AO dataset
            for nucleus in set(aos_data.nucleus_index):
                # collect AO indices that belong to this nucleus
                ao_idxs = [i for i, nuc in enumerate(aos_data.nucleus_index) if nuc == nucleus]

                # 2) Within this nucleus, group AOs by their angular momentum quantum number l
                for l in set(aos_data.angular_momentums):
                    ao_idxs_l = [i for i in ao_idxs if aos_data.angular_momentums[i] == l]
                    if not ao_idxs_l:
                        continue

                    # 3) For each AO index, determine its “key exponent”
                    #    as the exponent of the primitive with the largest absolute coefficient
                    key_exps = []
                    for i in ao_idxs_l:
                        prims = [p for p, orb in enumerate(aos_data.orbital_indices) if orb == i]
                        max_p = max(prims, key=lambda p: abs(aos_data.coefficients[p]))
                        key_exps.append((i, aos_data.exponents[max_p]))

                    # 4) Extract the unique basis exponents (Z values) and sort them
                    basis_exps = sorted({exp for _, exp in key_exps})
                    B = len(basis_exps)

                    # 5) Exception-aware partitioning of the basis exponents
                    if j3_choice in ("ao-small", "ao-medium", "ao-large"):
                        # define desired number of equal splits depending on mode
                        desired = {"ao-small": 3, "ao-medium": 4, "ao-large": 5}[j3_choice]
                        # if the number of distinct basis exponents is too small,
                        # pick the single central exponent
                        if B <= desired - 1:
                            sel_basis = [basis_exps[B // 2]]
                        else:
                            # otherwise, split into `desired` parts
                            parts = np.array_split(basis_exps, desired)
                            if j3_choice == "ao-small":
                                # keep only the central part
                                idx = desired // 2
                                sel_basis = parts[idx]
                            elif j3_choice == "ao-medium":
                                # keep the two central parts
                                start = (desired - 2) // 2
                                sel_basis = np.concatenate(parts[start : start + 2])
                            else:  # ao-large
                                # keep the three central parts
                                start = (desired - 3) // 2
                                sel_basis = np.concatenate(parts[start : start + 3])
                    else:
                        # 'ao' or 'ao-full' → include all basis exponents
                        sel_basis = basis_exps

                    # 6) Select AO indices whose key exponent is in the chosen basis group
                    sel_ao = [i for (i, exp) in key_exps if exp in sel_basis]
                    selected_ao_indices_total.extend(sel_ao)

            # 7) Remove duplicates and sort the selected AO indices globally
            selected_ao_indices = sorted(set(selected_ao_indices_total))

            # 8) Build a mapping from old AO-index to new AO-index
            new_idx_map = {old: new for new, old in enumerate(selected_ao_indices)}

            # 9) Filter primitives: keep those whose orbital_indices appear in the new AO set
            new_prims = [p for p, orb in enumerate(aos_data.orbital_indices) if orb in new_idx_map]

            # 10) Reconstruct all common dataclass fields for the new AO object
            new_orbital_indices = [new_idx_map[aos_data.orbital_indices[p]] for p in new_prims]
            new_exponents = [aos_data.exponents[p] for p in new_prims]
            new_coefficients = [aos_data.coefficients[p] for p in new_prims]
            new_nucleus_index = [aos_data.nucleus_index[i] for i in selected_ao_indices]
            new_angular_momentums = [aos_data.angular_momentums[i] for i in selected_ao_indices]

            # 11) Reconstruct class-specific fields depending on data type
            if isinstance(aos_data, AOs_cart_data):
                new_polynominal_order_x = [aos_data.polynominal_order_x[i] for i in selected_ao_indices]
                new_polynominal_order_y = [aos_data.polynominal_order_y[i] for i in selected_ao_indices]
                new_polynominal_order_z = [aos_data.polynominal_order_z[i] for i in selected_ao_indices]
            elif isinstance(aos_data, AOs_sphe_data):
                new_magnetic_quantum_numbers = [aos_data.magnetic_quantum_numbers[i] for i in selected_ao_indices]
            else:
                raise ImportError(f"Invalid AOs data type: {type(aos_data)}")

            # 12) Assemble keyword arguments for the new dataclass constructor
            common_kwargs = {
                "structure_data": aos_data.structure_data,
                "nucleus_index": new_nucleus_index,
                "num_ao": len(selected_ao_indices),
                "num_ao_prim": len(new_orbital_indices),
                "orbital_indices": new_orbital_indices,
                "exponents": new_exponents,
                "coefficients": new_coefficients,
                "angular_momentums": new_angular_momentums,
            }

            if isinstance(aos_data, AOs_cart_data):
                common_kwargs.update(
                    {
                        "polynominal_order_x": new_polynominal_order_x,
                        "polynominal_order_y": new_polynominal_order_y,
                        "polynominal_order_z": new_polynominal_order_z,
                    }
                )
            else:
                common_kwargs["magnetic_quantum_numbers"] = new_magnetic_quantum_numbers

            # 13) Construct a new instance of the same class with trimmed AO basis
            aos_data = type(aos_data)(**common_kwargs)

            jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data)
        elif j3_choice == "mo":
            jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=mos_data)
        else:
            raise ImportError(f"Invalid j3_basis_type = {j3_choice}.")
    else:
        jastrow_threebody_data = None

    # NN three-body Jastrow (SchNet-like). If requested, initialize NN_Jastrow_data
    # from the structure information and attach it to Jastrow_data.
    if isinstance(j_nn_type, str):
        j_nn_choice = j_nn_type.lower()
    elif j_nn_type is None:
        j_nn_choice = None
    else:
        default_value = getattr(j_nn_type, "default", None)
        j_nn_choice = default_value.lower() if isinstance(default_value, str) else None

    nn_jastrow_data = None
    if j_nn_choice is not None:
        if j_nn_choice == "schnet":
            kwargs = {}
            if j_nn_params is not None:
                for param in j_nn_params:
                    if "=" in param:
                        key, value = param.split("=", 1)
                        try:
                            if "." in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            pass  # keep as string
                        kwargs[key] = value

            nn_jastrow_data = Jastrow_NN_data.init_from_structure(
                structure_data,
                **kwargs,
            )
        else:
            raise ImportError(f"Invalid j_nn_type = {j_nn_type}. Supported types: 'schnet'.")

    # define data
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
        jastrow_nn_data=nn_jastrow_data,
    )

    # geminal_data = geminal_mo_data
    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    hamiltonian_data.save_to_hdf5(hamiltonian_file)

    typer.echo(f"Hamiltonian data is saved in {hamiltonian_file}.")


typer_click_trexio = typer.main.get_command(trexio_app)

_cli.add_command(typer_click_trexio, "trexio")


# hamiltonian_app
hamiltonian_app = typer.Typer(help="Read and convert Hamiltonian data.")


@hamiltonian_app.command("show-info")
def hamiltonian_show_info(
    hamiltonian_data: str = typer.Argument(..., help="hamiltonian_data file, e.g. hamiltonian_data.chk"),
):
    """Show information stored in the Hamiltonian data."""
    hamiltonian = Hamiltonian_data.load_from_hdf5(hamiltonian_data)
    hamiltonian.sanity_check()
    for line in hamiltonian._get_info():
        typer.echo(line)


@hamiltonian_app.command("to-xyz")
def hamiltonian_to_xyz(
    hamiltonian_data: str = typer.Argument(..., help="hamiltonian_data file, e.g. hamiltonian_data.chk"),
    xyz_file: str = typer.Option("struct.xyz", "-o", "--output", help="Output file name."),
):
    """Show information stored in the Hamiltonian data."""
    hamiltonian = Hamiltonian_data.load_from_hdf5(hamiltonian_data)
    structure_data = hamiltonian.structure_data

    with open(xyz_file, "w") as f:
        f.write(f"{structure_data.natom}\n")
        f.write("\n")
    for atom, coord in zip(structure_data.atomic_labels, structure_data.positions, strict=True):
        f.write(f"{atom} {coord[0] * Bohr_to_Angstrom} {coord[1] * Bohr_to_Angstrom} {coord[2] * Bohr_to_Angstrom}\n")


class ansatz_type(str, Enum):
    """Orbital type."""

    jsd = "jsd"
    jagp = "jagp"


@hamiltonian_app.command("conv-wf")
def hamiltonian_convert_wavefunction(
    hamiltonian_data_org_file: str = typer.Argument(..., help="hamiltonian_data file, e.g. hamiltonian_data.chk"),
    convert_to: ansatz_type = typer.Option(None, "-c", "--convert-to", help="Convert to another type of anstaz."),
    hamiltonian_data_conv_file: str = typer.Option(
        "hamiltonian_data_conv.h5", "-o", "--output", help="Output hamiltonian_data file."
    ),
):
    """Convert wavefunction data in the Hamiltonian data."""
    hamiltonian_org = Hamiltonian_data.load_from_hdf5(hamiltonian_data_org_file)

    wavefunction_data = hamiltonian_org.wavefunction_data
    structure_data = hamiltonian_org.structure_data
    coulomb_potential_data = hamiltonian_org.coulomb_potential_data

    geminal_data = wavefunction_data.geminal_data
    Jastrow_data = wavefunction_data.jastrow_data

    if convert_to == "jsd":
        raise NotImplementedError("Conversion to JSD is not implemented yet.")
    elif convert_to == "jagp":
        # conversion of SD to AGP
        typer.echo("Convert SD to AGP.")
        geminal_data = Geminal_data.convert_from_MOs_to_AOs(geminal_data)
    else:
        raise ImportError(f"Invalid convert_to = {convert_to}.")

    wavefunction_data = Wavefunction_data(jastrow_data=Jastrow_data, geminal_data=geminal_data)

    hamiltonian_conv_data = Hamiltonian_data(
        structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wavefunction_data
    )

    hamiltonian_conv_data.save_to_hdf5(hamiltonian_data_conv_file)

    typer.echo(f"Hamiltonian data is saved in {hamiltonian_data_conv_file}.")


typer_click_hamiltonian = typer.main.get_command(hamiltonian_app)

_cli.add_command(typer_click_hamiltonian, "hamiltonian")

# VMC app
vmc_app = typer.Typer(help="Pre- and Post-Processing for VMC calculations.")


# This should be removed in future release since it will be no longer useful.
@vmc_app.command("fix")
def vmc_chk_fix(
    restart_chk: str = typer.Argument(..., help="old chk file, e.g. vmc.chk"),
):
    """VMCopt chk file fix."""
    typer.echo(f"Fix checkpoint file(s) from {restart_chk}.")
    typer.echo(f"Backup to checkpoint file(s) bak_{restart_chk}.")
    shutil.copy(restart_chk, f"bak_{restart_chk}")

    basename_restart_chk = os.path.basename(restart_chk)
    pattern = re.compile(rf"(\d+)_{basename_restart_chk}")

    mpi_ranks = []
    with zipfile.ZipFile(restart_chk, "r") as z:
        for file_name in z.namelist():
            match = pattern.match(os.path.basename(file_name))
            if match:
                mpi_ranks.append(int(match.group(1)))

    typer.echo(f"Found {len(mpi_ranks)} MPI ranks.")

    filenames = [f"{mpi_rank}_{basename_restart_chk}.pkl.gz" for mpi_rank in mpi_ranks]

    for filename, mpi_rank in zip(filenames, mpi_ranks, strict=True):
        with zipfile.ZipFile(restart_chk, "r") as zipf:
            data = zipf.read(filename)
            vmc = pickle.loads(data)
            tmp_gz_filename = f".{mpi_rank}.pkl.gz"
            with gzip.open(tmp_gz_filename, "wb") as gz:
                pickle.dump(vmc, gz, protocol=pickle.HIGHEST_PROTOCOL)

    with zipfile.ZipFile(restart_chk, "w", zipfile.ZIP_DEFLATED) as zipf:
        for mpi_rank in mpi_ranks:
            gz_name = f".{mpi_rank}.pkl.gz"
            arcname = gz_name.lstrip(".")
            zipf.write(gz_name, arcname=arcname)
            os.remove(gz_name)


@vmc_app.command("generate-input")
def vmc_generate_input(
    flag: bool = typer.Option(False, "-g", "--generate", help="Generate input file for VMCopt calculations."),
    filename: str = typer.Option("vmc.toml", "-f", "--filename", help="Filename for the input file."),
    exclude_comment: bool = typer.Option(False, "-nc", "--without-comment", help="Exclude comments in the input file."),
):
    """Generate an input file for VMCopt calculations."""
    if flag:
        doc = tomlkit.document()

        control_table = tomlkit.table()
        for key, value in cli_parameters["control"].items():
            if value is None:
                control_table[key] = str(value)
            else:
                control_table[key] = value
            if not exclude_comment and not isinstance(value, bool):  # due to a bug of tomlkit
                control_table[key].comment(cli_parameters["control_comments"][key])
        control_table["job_type"] = "vmc"
        doc.add("control", control_table)

        vmc_table = tomlkit.table()
        for key, value in cli_parameters["vmc"].items():
            if value is None:
                vmc_table[key] = str(value)
            else:
                vmc_table[key] = value
            if not exclude_comment and not isinstance(value, bool):
                vmc_table[key].comment(cli_parameters["vmc_comments"][key])
        doc.add("vmc", vmc_table)

        with open(filename, "w") as f:
            f.write(tomlkit.dumps(doc))
        typer.echo(f"Input file is generated: {filename}")

    else:
        typer.echo("Activate the flag (-g) to generate an input file. See --help for more information.")


@vmc_app.command("analyze-output")
def vmc_analyze_output(
    filenames: List[str] = typer.Argument(..., help="Output files of vmc optimizations."),
    plot_graph: bool = typer.Option(False, "-p", "--plot_graph", help="Plot a graph summerizing the result using matplotlib."),
    save_graph: str = typer.Option(None, "-s", "--save-graph", help="Specify a graph filename."),
):
    """Analyze the output files of vmc optimizations."""
    iter_list = []
    E_list = []
    max_f_list = []
    signal_to_noise_list = []

    iter_pattern = re.compile(r"Optimization\sstep\s*=\s*(\d+)/\d+")
    E_pattern = re.compile(r"E\s*=\s*([-+]?\d+(?:\.\d+)?)(?:\s*\+\-\s*([-+]?\d+(?:\.\d+)?))\s*Ha")
    max_f_pattern = re.compile(r"Max\sf\s=\s(-?\d+(?:\.\d+)?)\s*\+\-\s*(\d+(?:\.\d+)?)")
    signal_to_noise_pattern = re.compile(r"Max of signal-to-noise of f = max\(\|f\|/\|std f\|\) = ([-+]?\d+(?:\.\d+)?)(?:\.)?")

    for filename in filenames:
        with open(filename, "r") as f:
            for line in f:
                # iter
                iter_match = iter_pattern.search(line)
                if iter_match:
                    main_value = int(iter_match.group(1))
                    iter_list.append(main_value)

                # E
                E_match = E_pattern.search(line)
                if E_match:
                    main_value = float(E_match.group(1))
                    uncertainty = float(E_match.group(2))
                    E_list.append(ufloat(main_value, uncertainty))

                # max_f
                max_f_match = max_f_pattern.search(line)
                if max_f_match:
                    main_value = float(max_f_match.group(1))
                    uncertainty = float(max_f_match.group(2))
                    max_f_list.append(ufloat(main_value, uncertainty))

                # signal_to_noise
                signal_to_noise_match = signal_to_noise_pattern.search(line)
                if signal_to_noise_match:
                    main_value = float(signal_to_noise_match.group(1))
                    signal_to_noise_list.append(main_value)

    sep = 54
    typer.echo("-" * sep)
    typer.echo(f"{'Iter':<8} {'E (Ha)':<10} {'Max f (Ha)':<12} {'Max of signal to noise of f':<16}")
    typer.echo("-" * sep)
    for iter, E, max_f, signal_to_noise in zip(iter_list, E_list, max_f_list, signal_to_noise_list, strict=False):
        typer.echo(f"{iter:4}  {E:8.2uS}  {max_f:+10.2uS}  {signal_to_noise:8.3f}")
    typer.echo("-" * sep)

    # plot graphs
    if plot_graph or save_graph is not None:
        iters = []
        E_means = []
        E_errs = []
        max_f_means = []
        max_f_errs = []

        for iter, E, max_f, _ in zip(iter_list, E_list, max_f_list, signal_to_noise_list, strict=True):
            iters.append(iter)
            E_means.append(E.n)
            E_errs.append(E.s)
            max_f_means.append(max_f.n)
            max_f_errs.append(max_f.s)

        plt.rcParams["font.size"] = 8
        plt.rcParams["font.family"] = "sans-serif"

        fig = plt.figure(figsize=(8, 4), facecolor="white", dpi=300, tight_layout=True)

        ax11 = fig.add_subplot(1, 2, 1)
        ax12 = ax11.twinx()

        ax11.tick_params(axis="both", which="both", direction="in")
        ax11.errorbar(iters, E_means, yerr=E_errs, fmt="o-", markersize=3, capsize=3, color="blue", label="Energy")
        ax11.set_xlabel("Iteration")
        ax11.set_ylabel("Energy (Ha)")

        ax12.errorbar(iters, max_f_means, yerr=max_f_errs, fmt="s-", markersize=3, capsize=3, color="red", label="Max |f|")
        ax12.set_ylabel("max of |f|")

        lines11, labels11 = ax11.get_legend_handles_labels()
        lines12, labels12 = ax12.get_legend_handles_labels()
        ax11.legend(lines11 + lines12, labels11 + labels12, loc="best")

        ax21 = fig.add_subplot(1, 2, 2)
        ax22 = ax21.twinx()

        ax21.tick_params(axis="both", which="both", direction="in")
        ax21.errorbar(iters, E_means, yerr=E_errs, fmt="o-", markersize=3, capsize=3, color="blue", label="Energy")
        ax21.set_xlabel("Iteration")
        ax21.set_ylabel("Energy (Ha)")

        ax22.plot(iters, signal_to_noise_list, marker="s", linestyle="-", markersize=3, color="red", label="max of |f|/|std f|")
        ax22.set_ylabel("max of signal to noise = |f|/|std f|")

        # Combine legend handles and labels for the second subplot
        lines21, labels21 = ax21.get_legend_handles_labels()
        lines22, labels22 = ax22.get_legend_handles_labels()
        ax21.legend(lines21 + lines22, labels21 + labels22, loc="best")

        if save_graph is not None:
            plt.savefig(save_graph)
            typer.echo(f"Graph is saved in {save_graph}.")

        if plot_graph:
            plt.show()


typer_click_vmc = typer.main.get_command(vmc_app)

_cli.add_command(typer_click_vmc, "vmc")


# mcmc app
mcmc_app = typer.Typer(help="Pre- and Post-Processing for MCMC calculations.")


# This should be removed in future release since it will be no longer useful.
@mcmc_app.command("fix")
def mcmc_chk_fix(
    restart_chk: str = typer.Argument(..., help="old chk file, e.g. mcmc.chk"),
):
    """VMC chk file fix."""
    typer.echo(f"Fix checkpoint file(s) from {restart_chk}.")
    typer.echo(f"Backup to checkpoint file(s) bak_{restart_chk}.")
    shutil.copy(restart_chk, f"bak_{restart_chk}")

    basename_restart_chk = os.path.basename(restart_chk)
    pattern = re.compile(rf"(\d+)_{basename_restart_chk}")

    mpi_ranks = []
    with zipfile.ZipFile(restart_chk, "r") as z:
        for file_name in z.namelist():
            match = pattern.match(os.path.basename(file_name))
            if match:
                mpi_ranks.append(int(match.group(1)))

    typer.echo(f"Found {len(mpi_ranks)} MPI ranks.")

    filenames = [f"{mpi_rank}_{basename_restart_chk}.pkl.gz" for mpi_rank in mpi_ranks]

    for filename, mpi_rank in zip(filenames, mpi_ranks, strict=True):
        with zipfile.ZipFile(restart_chk, "r") as zipf:
            data = zipf.read(filename)
            mcmc = pickle.loads(data)
            tmp_gz_filename = f".{mpi_rank}.pkl.gz"
            with gzip.open(tmp_gz_filename, "wb") as gz:
                pickle.dump(mcmc, gz, protocol=pickle.HIGHEST_PROTOCOL)

    with zipfile.ZipFile(restart_chk, "w", zipfile.ZIP_DEFLATED) as zipf:
        for mpi_rank in mpi_ranks:
            gz_name = f".{mpi_rank}.pkl.gz"
            arcname = gz_name.lstrip(".")
            zipf.write(gz_name, arcname=arcname)
            os.remove(gz_name)


@mcmc_app.command("compute-energy")
def mcmc_compute_energy(
    restart_chk: str = typer.Argument(..., help="Restart checkpoint file, e.g. mcmc.rchk"),
    num_mcmc_bin_blocks: int = typer.Option(
        1,
        "-b",
        "--num_mcmc_bin_blocks",
        help="Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.",
    ),
    num_mcmc_warmup_steps: int = typer.Option(
        0, "-w", "--num_mcmc_warmup_steps", help="Number of observable measurement steps for warmup (i.e., discarged)."
    ),
):
    """VMC energy calculation."""
    typer.echo(f"Read restart checkpoint file(s) from {restart_chk}.")

    if num_mcmc_warmup_steps < MCMC_MIN_WARMUP_STEPS:
        typer.echo(f"num_mcmc_warmup_steps should be larger than {MCMC_MIN_WARMUP_STEPS}.")
    if num_mcmc_bin_blocks < MCMC_MIN_BIN_BLOCKS:
        typer.echo(f"num_mcmc_bin_blocks should be larger than {MCMC_MIN_BIN_BLOCKS}.")

    """Unzip the checkpoint file for each process and load them."""
    pattern = re.compile(r"(\d+).pkl.gz")

    mpi_ranks = []
    with zipfile.ZipFile(restart_chk, "r") as z:
        for file_name in z.namelist():
            match = pattern.match(os.path.basename(file_name))
            if match:
                mpi_ranks.append(int(match.group(1)))

    typer.echo(f"Found {len(mpi_ranks)} MPI ranks.")

    filenames = [f"{mpi_rank}.pkl.gz" for mpi_rank in mpi_ranks]

    w_L_binned_list = []
    w_L_e_L_binned_list = []

    for filename in filenames:
        with zipfile.ZipFile(restart_chk, "r") as zipf:
            with zipf.open(filename) as zipped_gz_fobj:
                with gzip.open(zipped_gz_fobj, "rb") as gz:
                    mcmc = pickle.load(gz)
                    e_L = mcmc.e_L[num_mcmc_warmup_steps:]
                    w_L = mcmc.w_L[num_mcmc_warmup_steps:]
                    w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
                    w_L_binned = list(np.ravel([np.mean(arr, axis=0) for arr in w_L_split]))
                    w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
                    w_L_e_L_binned = list(np.ravel([np.mean(arr, axis=0) for arr in w_L_e_L_split]))
                    w_L_binned_list += w_L_binned
                    w_L_e_L_binned_list += w_L_e_L_binned

    w_L_binned = np.array(w_L_binned_list)
    w_L_e_L_binned = np.array(w_L_e_L_binned_list)

    # jackknife implementation
    w_L_binned_sum = np.sum(w_L_binned)
    w_L_e_L_binned_sum = np.sum(w_L_e_L_binned)

    M = w_L_binned.size
    typer.echo(f"Total number of binned samples = {M}")

    E_jackknife_binned = np.array(
        [(w_L_e_L_binned_sum - w_L_e_L_binned[m]) / (w_L_binned_sum - w_L_binned[m]) for m in range(M)]
    )

    E_mean = np.average(E_jackknife_binned)
    E_std = np.sqrt(M - 1) * np.std(E_jackknife_binned)

    typer.echo(f"E = {E_mean} +- {E_std} Ha.")


@mcmc_app.command("generate-input")
def mcmc_generate_input(
    flag: bool = typer.Option(False, "-g", "--generate", help="Generate input file for VMC calculations."),
    filename: str = typer.Option("mcmc.toml", "-f", "--filename", help="Filename for the input file."),
    exclude_comment: bool = typer.Option(False, "-nc", "--without-comment", help="Exclude comments in the input file."),
):
    """Generate an input file for VMC calculations."""
    if flag:
        doc = tomlkit.document()

        control_table = tomlkit.table()
        for key, value in cli_parameters["control"].items():
            if value is None:
                control_table[key] = str(value)
            else:
                control_table[key] = value
            if not exclude_comment and not isinstance(value, bool):
                control_table[key].comment(cli_parameters["control_comments"][key])
        control_table["job_type"] = "mcmc"
        doc.add("control", control_table)

        mcmc_table = tomlkit.table()
        for key, value in cli_parameters["mcmc"].items():
            if value is None:
                mcmc_table[key] = str(value)
            else:
                mcmc_table[key] = value
            if not exclude_comment and not isinstance(value, bool):
                mcmc_table[key].comment(cli_parameters["mcmc_comments"][key])
        doc.add("mcmc", mcmc_table)

        with open(filename, "w") as f:
            f.write(tomlkit.dumps(doc))
        typer.echo(f"Input file is generated: {filename}")

    else:
        typer.echo("Activate the flag (-g) to generate an input file. See --help for more information.")


typer_click_mcmc = typer.main.get_command(mcmc_app)

_cli.add_command(typer_click_mcmc, "mcmc")


# LRDMC_app
lrdmc_app = typer.Typer(help="Pre- and Post-Processing for LRDMC calculations.")


# This should be removed in future release since it will be no longer useful.
@lrdmc_app.command("fix")
def lrdmc_chk_fix(
    restart_chk: str = typer.Argument(..., help="old chk file, e.g. lrdmc.chk"),
):
    """LRDMC chk file fix."""
    typer.echo(f"Fix checkpoint file(s) from {restart_chk}.")
    typer.echo(f"Backup to checkpoint file(s) bak_{restart_chk}.")
    shutil.copy(restart_chk, f"bak_{restart_chk}")

    basename_restart_chk = os.path.basename(restart_chk)
    pattern = re.compile(rf"(\d+)_{basename_restart_chk}")

    mpi_ranks = []
    with zipfile.ZipFile(restart_chk, "r") as z:
        for file_name in z.namelist():
            match = pattern.match(os.path.basename(file_name))
            if match:
                mpi_ranks.append(int(match.group(1)))

    typer.echo(f"Found {len(mpi_ranks)} MPI ranks.")

    filenames = [f"{mpi_rank}_{basename_restart_chk}.pkl.gz" for mpi_rank in mpi_ranks]

    for filename, mpi_rank in zip(filenames, mpi_ranks, strict=True):
        with zipfile.ZipFile(restart_chk, "r") as zipf:
            data = zipf.read(filename)
            lrdmc = pickle.loads(data)
            tmp_gz_filename = f".{mpi_rank}.pkl.gz"
            with gzip.open(tmp_gz_filename, "wb") as gz:
                pickle.dump(lrdmc, gz, protocol=pickle.HIGHEST_PROTOCOL)

    with zipfile.ZipFile(restart_chk, "w", zipfile.ZIP_DEFLATED) as zipf:
        for mpi_rank in mpi_ranks:
            gz_name = f".{mpi_rank}.pkl.gz"
            arcname = gz_name.lstrip(".")
            zipf.write(gz_name, arcname=arcname)
            os.remove(gz_name)


@lrdmc_app.command("compute-energy")
def lrdmc_compute_energy(
    restart_chk: str = typer.Argument(..., help="Restart checkpoint file, e.g. lrdmc.rchk"),
    num_gfmc_bin_block: int = typer.Option(
        5,
        "-b",
        "--num_gfmc_bin_blocks",
        help="Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_gfmc_bin_blocks, not num_gfmc_bin_blocks * mpi_size * number_of_walkers.",
    ),
    num_gfmc_warmup_steps: int = typer.Option(
        0, "-w", "--num_gfmc_warmup_steps", help="Number of observable measurement steps for warmup (i.e., discarged)."
    ),
    num_gfmc_collect_steps: int = typer.Option(
        5, "-c", "--num_gfmc_collect_steps", help="Number of measurement (before binning) for collecting the weights."
    ),
):
    """LRDMC energy calculation."""
    typer.echo(f"Read restart checkpoint file(s) from {restart_chk}.")

    if num_gfmc_warmup_steps < GFMC_MIN_WARMUP_STEPS:
        typer.echo(f"num_gfmc_warmup_steps should be larger than {GFMC_MIN_WARMUP_STEPS}.")
    if num_gfmc_bin_block < GFMC_MIN_BIN_BLOCKS:
        typer.echo(f"num_mcmc_bin_blocks should be larger than {GFMC_MIN_BIN_BLOCKS}.")
    if num_gfmc_collect_steps < GFMC_MIN_COLLECT_STEPS:
        typer.echo(f"num_gfmc_collect_steps should be larger than {GFMC_MIN_COLLECT_STEPS}.")

    pattern = re.compile(r"(\d+).pkl.gz")

    mpi_ranks = []
    with zipfile.ZipFile(restart_chk, "r") as z:
        for file_name in z.namelist():
            match = pattern.match(os.path.basename(file_name))
            if match:
                mpi_ranks.append(int(match.group(1)))

    typer.echo(f"Found {len(mpi_ranks)} MPI ranks.")

    filenames = [f"{mpi_rank}.pkl.gz" for mpi_rank in mpi_ranks]

    w_L_binned_list = []
    w_L_e_L_binned_list = []

    num_mcmc_warmup_steps = num_gfmc_warmup_steps
    num_mcmc_bin_blocks = num_gfmc_bin_block

    for filename in filenames:
        with zipfile.ZipFile(restart_chk, "r") as zipf:
            with zipf.open(filename) as zipped_gz_fobj:
                with gzip.open(zipped_gz_fobj, "rb") as gz:
                    lrdmc = pickle.load(gz)
            lrdmc.num_gfmc_collect_steps = num_gfmc_collect_steps

            if lrdmc.e_L.size != 0:
                e_L = lrdmc.e_L[num_mcmc_warmup_steps:]
                w_L = lrdmc.w_L[num_mcmc_warmup_steps:]
                w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
                w_L_binned = list(np.ravel([np.mean(arr, axis=0) for arr in w_L_split]))
                w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
                w_L_e_L_binned = list(np.ravel([np.mean(arr, axis=0) for arr in w_L_e_L_split]))
                w_L_binned_list += w_L_binned
                w_L_e_L_binned_list += w_L_e_L_binned

    w_L_binned = np.array(w_L_binned_list)
    w_L_e_L_binned = np.array(w_L_e_L_binned_list)

    # jackknife implementation
    w_L_binned_sum = np.sum(w_L_binned)
    w_L_e_L_binned_sum = np.sum(w_L_e_L_binned)

    M = w_L_binned.size
    typer.echo(f"Total number of binned samples = {M}")

    E_jackknife_binned = np.array(
        [(w_L_e_L_binned_sum - w_L_e_L_binned[m]) / (w_L_binned_sum - w_L_binned[m]) for m in range(M)]
    )

    E_mean = np.average(E_jackknife_binned)
    E_std = np.sqrt(M - 1) * np.std(E_jackknife_binned)

    typer.echo(f"E = {E_mean} +- {E_std} Ha.")


@lrdmc_app.command("extrapolate-energy")
def lrdmc_extrapolate_energy(
    restart_chks: List[str] = typer.Argument(..., help="Restart checkpoint files, e.g. lrdmc.rchk"),
    polynomial_order: int = typer.Option(
        2,
        "-p",
        "--polynomial-order",
        help="Polynomial order with respect to a^2 for extrapolation. Default is 2. i.e., E_0 + a^2 * E_2 + a^4 * E_4.",
    ),
    plot_graph: bool = typer.Option(False, "-g", "--plot-graph", help="Plot a graph summerizing the result using matplotlib."),
    save_graph: str = typer.Option(None, "-s", "--save-graph", help="Specify a graph filename."),
    num_gfmc_bin_block: int = typer.Option(
        5,
        "-b",
        "--num_gfmc_bin_blocks",
        help="Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_gfmc_bin_blocks, not num_gfmc_bin_blocks * mpi_size * number_of_walkers.",
    ),
    num_gfmc_warmup_steps: int = typer.Option(
        0, "-w", "--num_gfmc_warmup_steps", help="Number of observable measurement steps for warmup (i.e., discarged)."
    ),
    num_gfmc_collect_steps: int = typer.Option(
        5, "-c", "--num_gfmc_collect_steps", help="Number of measurement (before binning) for collecting the weights."
    ),
):
    """LRDMC energy calculation."""
    sep = 72
    typer.echo("-" * sep)
    typer.echo(f"Read restart checkpoint files from {restart_chks}.")

    alat_list = []
    energy_list = []
    energy_error_list = []

    for restart_chk in restart_chks:
        pattern = re.compile(r"(\d+).pkl.gz")

        mpi_ranks = []
        with zipfile.ZipFile(restart_chk, "r") as z:
            for file_name in z.namelist():
                match = pattern.match(os.path.basename(file_name))
                if match:
                    mpi_ranks.append(int(match.group(1)))

        typer.echo(f"Found {len(mpi_ranks)} MPI ranks.")

        filenames = [f"{mpi_rank}.pkl.gz" for mpi_rank in mpi_ranks]

        w_L_binned_list = []
        w_L_e_L_binned_list = []

        num_mcmc_warmup_steps = num_gfmc_warmup_steps
        num_mcmc_bin_blocks = num_gfmc_bin_block

        for filename in filenames:
            with zipfile.ZipFile(restart_chk, "r") as zipf:
                with zipf.open(filename) as zipped_gz_fobj:
                    with gzip.open(zipped_gz_fobj, "rb") as gz:
                        lrdmc = pickle.load(gz)
                lrdmc.num_gfmc_collect_steps = num_gfmc_collect_steps

                if lrdmc.e_L.size != 0:
                    e_L = lrdmc.e_L[num_mcmc_warmup_steps:]
                    w_L = lrdmc.w_L[num_mcmc_warmup_steps:]
                    w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
                    w_L_binned = list(np.ravel([np.mean(arr, axis=0) for arr in w_L_split]))
                    w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
                    w_L_e_L_binned = list(np.ravel([np.mean(arr, axis=0) for arr in w_L_e_L_split]))
                    w_L_binned_list += w_L_binned
                    w_L_e_L_binned_list += w_L_e_L_binned
                    alat = lrdmc.alat

        w_L_binned = np.array(w_L_binned_list)
        w_L_e_L_binned = np.array(w_L_e_L_binned_list)

        # jackknife implementation
        w_L_binned_sum = np.sum(w_L_binned)
        w_L_e_L_binned_sum = np.sum(w_L_e_L_binned)

        M = w_L_binned.size
        typer.echo(f"  Total number of binned samples = {M}")

        E_jackknife_binned = np.array(
            [(w_L_e_L_binned_sum - w_L_e_L_binned[m]) / (w_L_binned_sum - w_L_binned[m]) for m in range(M)]
        )

        E_mean = np.average(E_jackknife_binned)
        E_std = np.sqrt(M - 1) * np.std(E_jackknife_binned)
        a = alat

        alat_list.append(a)
        energy_list.append(E_mean)
        energy_error_list.append(E_std)

        typer.echo(f"  For a = {a} bohr: E = {E_mean} +- {E_std} Ha.")

    typer.echo("-" * sep)

    typer.echo("Extrapolation of the energy with respect to a^2.")
    typer.echo(f"  Polynomial order = {polynomial_order}.")

    # Prepare data for polynomial fitting
    monte_carlo_loop = 1000
    interpolate_num = 1000
    order_fit = polynomial_order
    x = np.array(alat_list) ** 2
    y = np.array(energy_list)
    y_err = np.array(energy_error_list)
    xs = np.linspace(0.0, np.max(x) * 1.2, interpolate_num)

    # Lists to store evaluated polynomial values for each Monte Carlo iteration at x and xs
    vals = []  # Stores np.polyval(w, x) for each iteration
    vals_plot = []  # Stores np.polyval(w, xs) for each iteration

    # Lists to store local minima results
    y0_list = []

    # Pre-generate Monte Carlo random deviations for each data point
    sigma_list = [np.random.randn(monte_carlo_loop) for _ in y]

    # Monte Carlo loop: iterate over each random realization
    for m in range(monte_carlo_loop):
        # Generate y values with added noise
        y_gen = [y[i] + y_err[i] * sigma_list[i][m] for i in range(len(y))]

        # Fit a polynomial of degree order_fit
        w = np.polyfit(x, y_gen, order_fit)

        # Evaluate the fitted polynomial at points xs
        ys_plot = np.polyval(w, xs)

        # Record the polynomial evaluations
        vals_plot.append(ys_plot)

        # a -> 0
        poly = np.poly1d(w)
        y0_list.append(poly(0.0))

    # Convert lists to NumPy arrays and calculate the mean and standard deviation
    vals = np.array(vals)  # shape: (monte_calro_loop, len(x))
    vals_plot = np.array(vals_plot)  # shape: (monte_calro_loop, len(xs))

    y0_list = np.array(y0_list)  # shape: (monte_calro_loop,)
    y_mean_plot = np.mean(vals_plot, axis=0)

    E_0_mean = np.mean(y0_list)
    E_0_std = np.std(y0_list)

    typer.echo(f"  For a -> 0 bohr: E = {E_0_mean} +- {E_0_std} Ha.")
    typer.echo("-" * sep)

    # plot graphs
    if plot_graph or save_graph is not None:
        plt.rcParams["font.size"] = 8
        plt.rcParams["font.family"] = "sans-serif"

        fig = plt.figure(figsize=(4, 4), facecolor="white", dpi=300, tight_layout=True)

        ax1 = fig.add_subplot(1, 1, 1)
        ax1.tick_params(axis="both", which="both", direction="in")
        ax1.errorbar(x, y, yerr=y_err, fmt="o", markersize=4, capsize=3, color="blue", label="Data")
        ax1.errorbar(0.0, E_0_mean, yerr=E_0_std, fmt="s", markersize=4, capsize=3, color="red", label="Extrapolated")
        ax1.plot(xs, y_mean_plot, ls="--", color="blue")
        ax1.set_xlabel(r"Lattice discretization $a^2$ (bohr$^2$)")
        ax1.set_ylabel("LRDMC Energy (Ha)")
        ax1.legend(loc="best")

        if save_graph is not None:
            plt.savefig(save_graph)
            typer.echo(f"Graph is saved in {save_graph}.")

        if plot_graph:
            plt.show()
    typer.echo("-" * sep)
    typer.echo("Extrapolation is finished.")


@lrdmc_app.command("generate-input")
def lrdmc_generate_input(
    flag: bool = typer.Option(False, "-g", "--generate", help="Generate input file for VMC calculations."),
    filename: str = typer.Option("lrdmc.toml", "-f", "--filename", help="Filename for the input file."),
    exclude_comment: bool = typer.Option(False, "-nc", "--without-comment", help="Exclude comments in the input file."),
):
    """Generate an input file for LRDMC calculations."""
    if flag:
        doc = tomlkit.document()

        control_table = tomlkit.table()
        for key, value in cli_parameters["control"].items():
            if value is None:
                control_table[key] = str(value)
            else:
                control_table[key] = value
            if not exclude_comment and not isinstance(value, bool):
                control_table[key].comment(cli_parameters["control_comments"][key])
        control_table["job_type"] = "lrdmc"
        doc.add("control", control_table)

        lrdmc_table = tomlkit.table()
        for key, value in cli_parameters["lrdmc"].items():
            if value is None:
                lrdmc_table[key] = str(value)
            else:
                lrdmc_table[key] = value
            if not exclude_comment and not isinstance(value, bool):
                lrdmc_table[key].comment(cli_parameters["lrdmc_comments"][key])
        doc.add("lrdmc", lrdmc_table)

        with open(filename, "w") as f:
            f.write(tomlkit.dumps(doc))
        typer.echo(f"Input file is generated: {filename}")

    else:
        typer.echo("Activate the flag (-g) to generate an input file. See --help for more information.")


typer_click_lrdmc = typer.main.get_command(lrdmc_app)

_cli.add_command(typer_click_lrdmc, "lrdmc")
