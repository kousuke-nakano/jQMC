"""jQMC tools.

Todo:
    trexio
        show info
        convert to hamiltonian_data
            jastrow-1b-flag = True (if AE), False (if ECP)
            jastrow-2b-flag = True
            jastrow_3b-flag = True
            jastrow-orbital-type 'ao' or 'mo', default 'ao'

    hamiltonian
        show info
        convert-to-xml
        convert-from-xml

    vmc
        generate input file
        get energy and forces
            num_blocks
            warmup_steps

    vmcopt
        generate input file
        get energy and devmax
        plot energy and devmax
        average hamiltonians

    lrdmc
        generate input file
        get energy and forces
            num_blocks
            warmup_steps
            collections steps
        get extrapolated energy and forces
            polynomial order
            num_blocks
            warmup_steps
            collections steps
"""

from typing import List, Optional, Tuple

import click
import tomlkit
import typer

from .jqmc_cli import default_parameters


@click.group()
def cli():
    pass


@cli.command()
def initdb():
    click.echo("Initialized the database")


@cli.command()
def dropdb():
    click.echo("Dropped the database")


vmc_app = typer.Typer()


@vmc_app.command("compute-energy")
def vmc_compute_energy(
    args: Tuple[int, int] = typer.Argument((1, 0), metavar="num_mcmc_bin, num_mcmc_warmups", help="xxx"),
    rchk: str = typer.Argument("vmc.rchk", metavar="vmc.rchk file", help="xxx"),
):
    """VMC energy calculation."""
    typer.echo("Typer is now below Click, the Click app is the top level")


@vmc_app.command("generate-input")
def vmc_generate_input():
    """Generate an input file for VMC calculations. Default filename is vmc.toml."""
    doc = tomlkit.document()

    control_table = tomlkit.table()
    for key, value in default_parameters["control"].items():
        if value is None:
            value = "None"
        control_table[key] = str(value)
        control_table[key].comment(default_parameters["control_comments"][key])
    doc.add("control", control_table)

    vmc_table = tomlkit.table()
    for key, value in default_parameters["vmc"].items():
        if value is None:
            value = "None"
        vmc_table[key] = value
        vmc_table[key].comment(default_parameters["vmc_comments"][key])
    doc.add("vmc", vmc_table)

    with open("vmc.toml", "w") as f:
        f.write(tomlkit.dumps(doc))


typer_click_vmc = typer.main.get_command(vmc_app)

cli.add_command(typer_click_vmc, "vmc")


def main():
    cli()
