"""jQMC tools.

Todo:
    vmcopt
        generate input file
        get energy and devmax
        plot energy and devmax
        average hamiltonians

    lrdmc
        get extrapolated energy and forces
            polynomial order
            num_blocks
            warmup_steps
            collections steps
"""

import pickle
import re
import zipfile
from enum import Enum

import click
import numpy as np
import tomlkit
import typer

from .determinant import Geminal_data
from .hamiltonians import Hamiltonian_data
from .jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_three_body_data, Jastrow_two_body_data
from .jqmc_miscs import cli_parameters
from .trexio_wrapper import read_trexio_file
from .wavefunction import Wavefunction_data


@click.group()
def cli():
    """A useful command-line tool set for jQMC package."""
    pass


# trexio_app
trexio_app = typer.Typer(help="Read and Convert TREXIO files.")


@trexio_app.command("show-info")
def trexio_show_info(
    filename: str = typer.Argument(..., help="TREXIO file name."),
):
    """Show information stored in the TREXIO file."""
    (structure_data, aos_data, mos_data_up, mos_data_dn, geminal_data, coulomb_potential_data) = read_trexio_file(filename)

    for line in structure_data.get_info():
        typer.echo(line)
    for line in aos_data.get_info():
        typer.echo(line)
    for line in mos_data_up.get_info():
        typer.echo(line)
    for line in mos_data_dn.get_info():
        typer.echo(line)
    for line in geminal_data.get_info():
        typer.echo(line)
    for line in coulomb_potential_data.get_info():
        typer.echo(line)


typer_click_trexio = typer.main.get_command(trexio_app)

cli.add_command(typer_click_trexio, "trexio")


class orbital_type(str, Enum):
    """Orbital type."""

    ao = "ao"
    mo = "mo"


@trexio_app.command("convert-to")
def trexio_convert_to(
    trexio_file: str = typer.Argument(..., help="TREXIO filename."),
    hamiltonian_file: str = typer.Option("hamiltonian_data.chk", "-o", "--output", help="Output file name."),
    j1_parmeter: float = typer.Option(None, "-j1", "--jastrow-1b-parameter", help="Jastrow one-body parameter."),
    j2_parmeter: float = typer.Option(None, "-j2", "--jastrow-2b-parameter", help="Jastrow two-body parameter."),
    j3_basis_type: orbital_type = typer.Option(
        None, "-j3", "--jastrow-3b-basis-set-type", help="Jastrow three-body basis-set type"
    ),
):
    """Convert a TREXIO file to hamiltonian_data."""
    (structure_data, aos_data, mos_data, _, geminal_data, coulomb_potential_data) = read_trexio_file(trexio_file)

    if j1_parmeter is not None:
        if coulomb_potential_data.ecp_flag:
            core_electrons = coulomb_potential_data.z_cores
        else:
            core_electrons = [0] * coulomb_potential_data.n_atom
        jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=j1_parmeter, structure_data=structure_data, core_electrons=core_electrons
        )
    else:
        jastrow_onebody_data = None
    if j2_parmeter is not None:
        jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=j2_parmeter)
    else:
        jastrow_twobody_data = None
    if j3_basis_type is not None:
        if j3_basis_type == "ao":
            jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data)
        elif j3_basis_type == "mo":
            jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=mos_data)
        else:
            raise ImportError(f"Invalid j3_basis_type = {j3_basis_type}.")
    else:
        jastrow_threebody_data = None
    # define data
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
    )

    # geminal_data = geminal_mo_data
    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    with open(hamiltonian_file, "wb") as f:
        pickle.dump(hamiltonian_data, f)

    typer.echo(f"Hamiltonian data is saved in {hamiltonian_file}.")


typer_click_trexio = typer.main.get_command(trexio_app)

cli.add_command(typer_click_trexio, "trexio")


# hamiltonian_app
hamiltonian_app = typer.Typer()


@hamiltonian_app.command("show-info")
def hamiltonian_show_info(
    hamiltonian_data: str = typer.Argument(..., help="hamiltonian_data file, e.g. hamiltonian_data.chk"),
):
    """Show information stored in the Hamiltonian data."""
    with open(hamiltonian_data, "rb") as f:
        hamiltonian = pickle.load(f)
        for line in hamiltonian.get_info():
            typer.echo(line)


typer_click_hamiltonian = typer.main.get_command(hamiltonian_app)

cli.add_command(typer_click_hamiltonian, "hamiltonian")


class ansatz_type(str, Enum):
    """Orbital type."""

    jsd = "jsd"
    jagp = "jagp"


@hamiltonian_app.command("conv-wf")
def hamiltonian_convert_wavefunction(
    hamiltonian_data_org_file: str = typer.Argument(..., help="hamiltonian_data file, e.g. hamiltonian_data.chk"),
    convert_to: ansatz_type = typer.Option(None, "-c", "--convert-to", help="Convert to another type of anstaz."),
    hamiltonian_data_conv_file: str = typer.Option(
        "hamiltonian_data_conv.chk", "-o", "--output", help="Output hamiltonian_data file."
    ),
):
    """Convert wavefunction data in the Hamiltonian data."""
    with open(hamiltonian_data_org_file, "rb") as f:
        hamiltonian_org = pickle.load(f)

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

    with open(hamiltonian_data_conv_file, "wb") as f:
        pickle.dump(hamiltonian_conv_data, f)

    typer.echo(f"Hamiltonian data is saved in {hamiltonian_data_conv_file}.")


typer_click_hamiltonian = typer.main.get_command(hamiltonian_app)

cli.add_command(typer_click_hamiltonian, "hamiltonian")

# VMCopt_app
vmcopt_app = typer.Typer(help="Pre- and Post-Processing for VMC-opt calculations.")


@vmcopt_app.command("generate-input")
def vmcopt_generate_input(
    flag: bool = typer.Option(False, "-g", "--generate", help="Generate input file for VMCopt calculations."),
    filename: str = typer.Option("vmcopt.toml", "-f", "--filename", help="Filename for the input file."),
    exclude_comment: bool = typer.Option(False, "-nc", "--without-comment", help="Exclude comments in the input file."),
):
    """Generate an input file for VMCopt calculations."""
    if flag:
        doc = tomlkit.document()

        control_table = tomlkit.table()
        for key, value in cli_parameters["control"].items():
            if value is None:
                value = "None"
            control_table[key] = value
            if not exclude_comment:
                control_table[key].comment(cli_parameters["control_comments"][key])
        control_table["job_type"] = "vmcopt"
        doc.add("control", control_table)

        vmcopt_table = tomlkit.table()
        for key, value in cli_parameters["vmc"].items():
            if value is None:
                value = "None"
            vmcopt_table[key] = value
            if not exclude_comment:
                vmcopt_table[key].comment(cli_parameters["vmc_comments"][key])
        doc.add("vmcopt", vmcopt_table)

        with open(filename, "w") as f:
            f.write(tomlkit.dumps(doc))
        typer.echo(f"Input file is generated: {filename}")

    else:
        typer.echo("Activate the flag (-g) to generate an input file. See --help for more information.")


typer_click_vmcopt = typer.main.get_command(vmcopt_app)

cli.add_command(typer_click_vmcopt, "vmcopt")


# VMC_app
vmc_app = typer.Typer(help="Pre- and Post-Processing for VMC calculations.")


@vmc_app.command("compute-energy")
def vmc_compute_energy(
    restart_chk: str = typer.Argument(..., help="Restart checkpoint file, e.g. vmc.rchk"),
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

    """Unzip the checkpoint file for each process and load them."""
    pattern = re.compile(rf"(\d+)_{restart_chk}")

    mpi_ranks = []
    with zipfile.ZipFile(restart_chk, "r") as z:
        for file_name in z.namelist():
            match = pattern.match(file_name)
            if match:
                mpi_ranks.append(int(match.group(1)))

    typer.echo(f"Found {len(mpi_ranks)} MPI ranks.")

    filenames = [f"{mpi_rank}_{restart_chk}" for mpi_rank in mpi_ranks]

    w_L_binned_list = []
    w_L_e_L_binned_list = []

    for filename in filenames:
        with zipfile.ZipFile(restart_chk, "r") as zipf:
            data = zipf.read(filename)
            vmc = pickle.loads(data)

            if vmc.mcmc.e_L.size != 0:
                e_L = vmc.mcmc.e_L[num_mcmc_warmup_steps:]
                w_L = vmc.mcmc.w_L[num_mcmc_warmup_steps:]
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


@vmc_app.command("generate-input")
def vmc_generate_input(
    flag: bool = typer.Option(False, "-g", "--generate", help="Generate input file for VMC calculations."),
    filename: str = typer.Option("vmc.toml", "-f", "--filename", help="Filename for the input file."),
    exclude_comment: bool = typer.Option(False, "-nc", "--without-comment", help="Exclude comments in the input file."),
):
    """Generate an input file for VMC calculations."""
    if flag:
        doc = tomlkit.document()

        control_table = tomlkit.table()
        for key, value in cli_parameters["control"].items():
            if value is None:
                value = "None"
            control_table[key] = value
            if not exclude_comment:
                control_table[key].comment(cli_parameters["control_comments"][key])
        control_table["job_type"] = "vmc"
        doc.add("control", control_table)

        vmc_table = tomlkit.table()
        for key, value in cli_parameters["vmc"].items():
            if value is None:
                value = "None"
            vmc_table[key] = value
            if not exclude_comment:
                vmc_table[key].comment(cli_parameters["vmc_comments"][key])
        doc.add("vmc", vmc_table)

        with open(filename, "w") as f:
            f.write(tomlkit.dumps(doc))
        typer.echo(f"Input file is generated: {filename}")

    else:
        typer.echo("Activate the flag (-g) to generate an input file. See --help for more information.")


typer_click_vmc = typer.main.get_command(vmc_app)

cli.add_command(typer_click_vmc, "vmc")


# LRDMC_app
lrdmc_app = typer.Typer(help="Pre- and Post-Processing for LRDMC calculations.")


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

    pattern = re.compile(rf"(\d+)_{restart_chk}")

    mpi_ranks = []
    with zipfile.ZipFile(restart_chk, "r") as z:
        for file_name in z.namelist():
            match = pattern.match(file_name)
            if match:
                mpi_ranks.append(int(match.group(1)))

    typer.echo(f"Found {len(mpi_ranks)} MPI ranks.")

    filenames = [f"{mpi_rank}_{restart_chk}" for mpi_rank in mpi_ranks]

    w_L_binned_list = []
    w_L_e_L_binned_list = []

    num_mcmc_warmup_steps = num_gfmc_warmup_steps
    num_mcmc_bin_blocks = num_gfmc_bin_block

    for filename in filenames:
        with zipfile.ZipFile(restart_chk, "r") as zipf:
            data = zipf.read(filename)
            lrdmc = pickle.loads(data)
            lrdmc.mcmc.num_gfmc_collect_steps = num_gfmc_collect_steps

            if lrdmc.mcmc.e_L.size != 0:
                e_L = lrdmc.mcmc.e_L[num_mcmc_warmup_steps:]
                w_L = lrdmc.mcmc.w_L[num_mcmc_warmup_steps:]
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
                value = "None"
            control_table[key] = value
            if not exclude_comment:
                control_table[key].comment(cli_parameters["control_comments"][key])
        control_table["job_type"] = "lrdmc"
        doc.add("control", control_table)

        lrdmc_table = tomlkit.table()
        for key, value in cli_parameters["lrdmc"].items():
            if value is None:
                value = "None"
            lrdmc_table[key] = value
            if not exclude_comment:
                lrdmc_table[key].comment(cli_parameters["lrdmc_comments"][key])
        doc.add("lrdmc", lrdmc_table)

        with open(filename, "w") as f:
            f.write(tomlkit.dumps(doc))
        typer.echo(f"Input file is generated: {filename}")

    else:
        typer.echo("Activate the flag (-g) to generate an input file. See --help for more information.")


typer_click_lrdmc = typer.main.get_command(lrdmc_app)

cli.add_command(typer_click_lrdmc, "lrdmc")
