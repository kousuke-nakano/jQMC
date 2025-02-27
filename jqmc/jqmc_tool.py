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

import typer

app = typer.Typer()


@app.command("hello")
def hello_world():
    typer.echo("Hello World")


def main():
    app()


if __name__ == "__main__":
    main()
