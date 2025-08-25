#!python
# -*- coding: utf-8 -*-

"""

pyturbo: pseudopotentials related classes and methods

Todo:
    * docstrings are not completed.
    * refactoring assert sentences. The assert should not be used for any on-the-fly check.
    * implementing __str__ method.
    * implementing sanity_check method.

"""

# python modules
import os
import re
import shutil
import random
import string
from typing import Optional

# turbo-genius modules
from turbogenius.pyturbo.utils.env import pyturbo_tmp_dir, turborvb_bin_root
from turbogenius.pyturbo.utils.utility import (
    pygetline,
    pygrep_lineno,
    return_element_symbol,
    return_orbchr,
)
from turbogenius.pyturbo.utils.execute import run

# Logger
from logging import getLogger, StreamHandler, Formatter

logger = getLogger("pyturbo").getChild(__name__)


class Pseudopotentials:
    def __init__(
        self,
        max_ang_mom_plus_1: Optional[list] = None,
        z_core: Optional[list] = None,
        cutoff: Optional[list] = None,
        nucleus_index: Optional[list] = None,
        element_list: Optional[list] = None,
        ang_mom: Optional[list] = None,
        exponent: Optional[list] = None,
        coefficient: Optional[list] = None,
        power: Optional[list] = None,
    ):
        if max_ang_mom_plus_1 is None:
            max_ang_mom_plus_1 = []
        if z_core is None:
            z_core = []
        if cutoff is None:
            cutoff = []
        if nucleus_index is None:
            nucleus_index = []
        if element_list is None:
            element_list = []
        if ang_mom is None:
            ang_mom = []
        if exponent is None:
            exponent = []
        if coefficient is None:
            coefficient = []
        if power is None:
            power = []

        # variables
        self.element_list = element_list
        self.max_ang_mom_plus_1 = max_ang_mom_plus_1
        self.z_core = z_core
        self.cutoff = cutoff
        self.nucleus_index = nucleus_index
        self.ang_mom = ang_mom
        self.exponent = exponent
        self.coefficient = coefficient
        self.power = power

        logger.debug(f"element_list={self.element_list}")
        logger.debug(f"max_ang_mom_plus_1={self.max_ang_mom_plus_1}")
        logger.debug(f"z_core={self.z_core}")
        logger.debug(f"cutoff={self.cutoff}")
        logger.debug(f"nucleus_index={self.nucleus_index}")
        logger.debug(f"ang_mom={self.ang_mom}")
        logger.debug(f"exponent={self.exponent}")
        logger.debug(f"coefficient={self.coefficient}")
        logger.debug(f"power={self.power}")

        # assertion!!
        assert len(set(self.nucleus_index)) == len(self.max_ang_mom_plus_1)
        assert len(set(self.nucleus_index)) == len(self.z_core)
        assert len(set(self.nucleus_index)) == len(self.cutoff)
        assert len(self.ang_mom) == len(self.exponent)
        assert len(self.ang_mom) == len(self.coefficient)
        assert len(self.ang_mom) == len(self.power)

    @property
    def nuclei_num(self):
        return len(set(self.nucleus_index))

    @property
    def ecp_num(self):
        return len(self.ang_mom)

    def write_pseudopotential_turborvb_file(self, file: str = "pseudo.dat"):
        with open(file, "w") as f:
            output = []
            output.append("ECP\n")

            for i, nuclei_i in enumerate(set(self.nucleus_index)):
                max_ang_mom_plus_1 = self.max_ang_mom_plus_1[i]  # ask to Sandro and Michele!!
                element = self.element_list[i]  # not used.
                z_core = self.z_core[i]  # not used.
                cutoff = self.cutoff[i]
                output.append(
                    "{:14d}  {:14f}  {:14d}\n".format(nuclei_i + 1, cutoff, max_ang_mom_plus_1 + 1)
                )  # ask to Sandro and Michele!!

                nindex_list = [i for i, x in enumerate(self.nucleus_index) if x == nuclei_i]
                ang_mom_n = [self.ang_mom[n] for n in nindex_list]
                exponent_n = [self.exponent[n] for n in nindex_list]
                coefficient_n = [self.coefficient[n] for n in nindex_list]
                power_n = [self.power[n] for n in nindex_list]

                logger.debug(nindex_list)
                logger.debug(ang_mom_n)
                logger.debug(exponent_n)
                logger.debug(coefficient_n)
                logger.debug(power_n)

                for l in range(max_ang_mom_plus_1 + 1):
                    output.append("{:14d} ".format(ang_mom_n.count(l)))
                output.append("\n")

                logger.debug(max_ang_mom_plus_1 + 1)
                for l in range(max_ang_mom_plus_1 + 1):
                    lindex_list = [i for i, x in enumerate(ang_mom_n) if x == l]
                    exponent_l = [exponent_n[n] for n in lindex_list]
                    coefficient_l = [coefficient_n[n] for n in lindex_list]
                    power_l = [power_n[n] for n in lindex_list]

                    logger.debug(lindex_list)
                    logger.debug(exponent_l)
                    logger.debug(coefficient_l)
                    logger.debug(power_l)

                    for exp, coeff, pow in zip(exponent_l, coefficient_l, power_l):
                        output.append("{:14f}  {:14f}  {:14f}\n".format(coeff, pow + 2, exp))

            f.writelines(output)

    def set_cutoffs(self, tollerance: float = 0.00001):
        logger.debug(self.cutoff)
        new_cutoff = []
        current_dir = os.getcwd()

        try:
            num_string = 15
            rand_string = "".join(random.choices(string.ascii_letters + string.digits, k=num_string))
            pyturbo_tmp_rand_dir = os.path.join(pyturbo_tmp_dir, rand_string)
            os.makedirs(pyturbo_tmp_rand_dir, exist_ok=True)
            os.chdir(pyturbo_tmp_rand_dir)
            for i, nuclei_i in enumerate(set(self.nucleus_index)):
                file = "pseudo.dat"
                with open(file, "x") as f:
                    output = []
                    output.append("ECP\n")

                    max_ang_mom_plus_1 = self.max_ang_mom_plus_1[i]
                    element = self.element_list[i]  # not used.
                    z_core = self.z_core[i]  # not used.
                    cutoff = self.cutoff[i]
                    output.append("{:14d}  {:14f}  {:14d}\n".format(1, cutoff, max_ang_mom_plus_1 + 1))

                    nindex_list = [i for i, x in enumerate(self.nucleus_index) if x == nuclei_i]
                    ang_mom_n = [self.ang_mom[n] for n in nindex_list]
                    exponent_n = [self.exponent[n] for n in nindex_list]
                    coefficient_n = [self.coefficient[n] for n in nindex_list]
                    power_n = [self.power[n] for n in nindex_list]
                    logger.debug(nindex_list)

                    for l in range(max_ang_mom_plus_1 + 1):
                        output.append("{:14d} ".format(ang_mom_n.count(l)))
                    output.append("\n")

                    for l in range(max_ang_mom_plus_1 + 1):
                        lindex_list = [i for i, x in enumerate(ang_mom_n) if x == l]
                        exponent_l = [exponent_n[n] for n in lindex_list]
                        coefficient_l = [coefficient_n[n] for n in lindex_list]
                        power_l = [power_n[n] for n in lindex_list]

                        for exp, coeff, pow in zip(exponent_l, coefficient_l, power_l):
                            output.append("{:14f}  {:14f}  {:14f}\n".format(coeff, pow + 2, exp))

                    f.writelines(output)

                out_pp = "out_pp"
                cmd = f"echo {tollerance} | {os.path.join(turborvb_bin_root, 'pseudo.x')}"
                run(binary=cmd, output_name=out_pp)
                lineno = pygrep_lineno(file=out_pp, keyword="Suggested cut-off pseudo")
                suggested_cutoff = float(pygetline(filename=out_pp, lineno=lineno, clearcache=True).split()[4])
                new_cutoff.append(suggested_cutoff)
                logger.info(f"suggested_cutoff for nuculei index = {i} is rc = {suggested_cutoff} Bohr")

                # clean files
                os.remove(file)
                os.remove(out_pp)
                os.remove("pseudo.plot")

            logger.info(f"suggested cutoff list {new_cutoff}")
            self.cutoff = new_cutoff

        finally:
            shutil.rmtree(pyturbo_tmp_rand_dir)
            os.chdir(current_dir)

    @classmethod
    def parse_pseudopotential_from_turborvb_pseudo_dat(cls, file: str = "pseudo.dat"):
        max_ang_mom_plus_1 = []
        z_core = []
        cutoff = []
        nucleus_index = []
        element_list = []
        ang_mom = []
        exponent = []
        coefficient = []
        power = []

        stride = 0
        with open(file, "rb") as f:
            lines = f.readlines()
            stride += 1  # dummy line

            while stride < len(lines):
                nuc_i, r_cutoff, max_ang_mom_plus_2 = lines[stride].split()
                nuc_i = int(nuc_i) - 1
                r_cutoff = float(r_cutoff)
                max_ang_mom_plus_2 = int(max_ang_mom_plus_2)
                max_ang_mom_plus_1.append(max_ang_mom_plus_2 - 1)
                z_core.append(None)
                cutoff.append(r_cutoff)
                element_list.append(None)
                stride += 1  # dummy line

                num_component_list = list(map(int, lines[stride].split()))
                stride += 1
                for l, num_component in enumerate(num_component_list):
                    for _ in range(num_component):
                        coeff, pow, exp = lines[stride].split()
                        nucleus_index.append(nuc_i)
                        ang_mom.append(l)
                        coefficient.append(float(coeff))
                        power.append(float(pow) - 2)
                        exponent.append(float(exp))
                        stride += 1

        return cls(
            max_ang_mom_plus_1=max_ang_mom_plus_1,
            z_core=z_core,
            cutoff=cutoff,
            nucleus_index=nucleus_index,
            element_list=element_list,
            ang_mom=ang_mom,
            exponent=exponent,
            coefficient=coefficient,
            power=power,
        )

    @classmethod
    def parse_pseudopotential_from_gamess_format_texts(cls, texts):
        max_ang_mom_plus_1 = []
        z_core = []
        cutoff = []
        nucleus_index = []
        element_list = []
        ang_mom = []
        exponent = []
        coefficient = []
        power = []

        for nuc_i, text in enumerate(texts):
            if text is None:
                continue

            pseudo_potential = Pseudopotential.parse_pseudopotential_from_gamess_format_text(text)

            # storing
            max_ang_mom_plus_1.append(pseudo_potential.max_ang_mom_plus_1)
            z_core.append(pseudo_potential.z_core)
            cutoff.append(pseudo_potential.cutoff)
            nucleus_index += [nuc_i] * pseudo_potential.ecp_num
            element_list.append(pseudo_potential.element)

            logger.debug(pseudo_potential.max_ang_mom_plus_1)

            ang_mom += pseudo_potential.ang_mom
            exponent += pseudo_potential.exponent
            coefficient += pseudo_potential.coefficient
            power += pseudo_potential.power

        return cls(
            max_ang_mom_plus_1=max_ang_mom_plus_1,
            z_core=z_core,
            cutoff=cutoff,
            nucleus_index=nucleus_index,
            element_list=element_list,
            ang_mom=ang_mom,
            exponent=exponent,
            coefficient=coefficient,
            power=power,
        )

    @classmethod
    def parse_pseudopotential_from_gamess_format_files(cls, files):
        texts = []
        for file in files:
            if file is None:
                texts.append(None)
            else:
                with open(file, "r") as f:
                    text = f.readlines()
                texts.append("".join(text))

        return cls.parse_pseudopotential_from_gamess_format_texts(texts=texts)

    @classmethod
    def parse_pseudopotential_from_turborvb_format_files(cls, files):
        max_ang_mom_plus_1 = []
        z_core = []
        cutoff = []
        nucleus_index = []
        element_list = []
        ang_mom = []
        exponent = []
        coefficient = []
        power = []

        for nuc_i, file in enumerate(files):
            if file is None:
                continue

            pseudo_potential = Pseudopotential.parse_pseudopotential_from_turborvb_format_file(file)

            # storing
            max_ang_mom_plus_1.append(pseudo_potential.max_ang_mom_plus_1)
            z_core.append(pseudo_potential.z_core)
            cutoff.append(pseudo_potential.cutoff)
            nucleus_index += [nuc_i] * pseudo_potential.ecp_num
            element_list.append(pseudo_potential.element)

            ang_mom += pseudo_potential.ang_mom
            exponent += pseudo_potential.exponent
            coefficient += pseudo_potential.coefficient
            power += pseudo_potential.power

        return cls(
            max_ang_mom_plus_1=max_ang_mom_plus_1,
            z_core=z_core,
            cutoff=cutoff,
            nucleus_index=nucleus_index,
            element_list=element_list,
            ang_mom=ang_mom,
            exponent=exponent,
            coefficient=coefficient,
            power=power,
        )


class Pseudopotential:
    def __init__(
        self,
        max_ang_mom_plus_1: int = 0,
        z_core: int = 0,
        element: Optional[str] = None,
        cutoff: float = 0.0,
        ang_mom: Optional[list] = None,
        exponent: Optional[list] = None,
        coefficient: Optional[list] = None,
        power: Optional[list] = None,
    ):
        if ang_mom is None:
            ang_mom = []
        if exponent is None:
            exponent = []
        if coefficient is None:
            coefficient = []
        if power is None:
            power = []

        self.element = element
        self.max_ang_mom_plus_1 = max_ang_mom_plus_1
        self.z_core = z_core
        self.cutoff = cutoff
        self.ang_mom = ang_mom
        self.exponent = exponent
        self.coefficient = coefficient
        self.power = power

        logger.debug(self.max_ang_mom_plus_1)
        logger.debug(self.z_core)
        logger.debug(self.cutoff)
        logger.debug(self.ang_mom)
        logger.debug(self.exponent)
        logger.debug(self.coefficient)
        logger.debug(self.power)

        # assertion!!
        assert len(self.ang_mom) == len(self.exponent)
        assert len(self.ang_mom) == len(self.coefficient)
        assert len(self.ang_mom) == len(self.power)

    @property
    def ecp_num(self):
        return len(self.ang_mom)

    @classmethod
    def parse_pseudopotential_from_gamess_format_file(cls, file: str):
        with open(file, "r") as f:
            text = f.readlines()
        text = "".join(text)
        return cls.parse_pseudopotential_from_gamess_format_text(text=text)

    @classmethod
    def parse_pseudopotential_from_gamess_format_text(cls, text: str):
        """ "
        http://myweb.liu.edu/~nmatsuna/gamess/input/ECP.html
        -card 1-    PNAME, PTYPE, IZCORE, LMAX+1
        IZCORE is the number of core electrons to be removed.
        Obviously IZCORE must be an even number, or in other
        words, all core orbitals being removed must be
        completely occupied.
        LMAX+1 is the one higher than the maximum angular momentum
        occupied in the core orbitals being removed:
        to remove s,p,d,f core orbitals  (LMAX=0,1,2,3)
        we use p,d,f,g core potentials (LMAX+1=1,2,3,4).
        LMAX+1 is not permitted to exceed 4.
        -card 2-    NGPOT
        NGPOT is the number of Gaussians in this part of the
        fit to the local effective potential.
        -card 3-    CLP,NLP,ZLP   (repeat this card NGPOT times)
        CLP is the coefficient of this Gaussian in the potential.
        NLP is the power of r for this Gaussian, 0 <= NLP <= 2.
        ZLP is the exponent of this Gaussian.
        The potential U(LMAX+1) is given first, followed by
        difference potentials U(L)-U(LMAX+1), for L=0,LMAX.

        """
        lines = text.split("\n")
        c = [line for line in lines if re.match(r"^\s*\S+.*", line)]  # remove blank lines

        ang_mom = []
        exponent_list = []
        coefficient_list = []
        power_list = []
        element, _, z_core, lmax_plus_1 = c[0].split()
        z_core = int(z_core)
        max_ang_mom_plus_1 = int(lmax_plus_1)
        lmax_plus_1_num_components = int(c[1].split()[0])
        cutoff = 0.0
        # the first component is the potential U(LMAX+1) is given first,
        # followed by difference potentials U(L)-U(LMAX+1), for L=0,LMAX.

        stride = 2
        for _ in range(lmax_plus_1_num_components):
            logger.debug(c[stride].split())
            coefficient, power, exponent = c[stride].split()
            coefficient = float(coefficient)
            power = int(power)
            exponent = float(exponent)
            ang_mom.append(max_ang_mom_plus_1)
            exponent_list.append(exponent)
            coefficient_list.append(coefficient)
            power_list.append(power - 2)
            stride += 1

        for l in range(max_ang_mom_plus_1):
            num_component = int(c[stride].split()[0])
            stride += 1
            for _ in range(num_component):
                coefficient, power, exponent = c[stride].split()
                exponent = float(exponent)
                power = int(power)
                coefficient = float(coefficient)
                ang_mom.append(l)
                exponent_list.append(exponent)
                coefficient_list.append(coefficient)
                power_list.append(power - 2)
                stride += 1

        logger.debug(ang_mom)
        logger.debug(exponent_list)
        logger.debug(coefficient_list)
        logger.debug(power_list)

        return cls(
            max_ang_mom_plus_1=max_ang_mom_plus_1,
            z_core=z_core,
            element=element,
            cutoff=cutoff,
            ang_mom=ang_mom,
            exponent=exponent_list,
            coefficient=coefficient_list,
            power=power_list,
        )

    @classmethod
    def parse_pseudopotential_from_turborvb_format_file(cls, file: str):
        with open(file, "r") as f:
            text = f.readlines()
        text = "".join(text)
        m = re.match(r"Z(\d+)_atomnumber(\d+)\..*", os.path.basename(file))
        assert len(m.groups()) == 2
        z_core = int(m.groups()[1]) - int(m.groups()[0])
        element = return_element_symbol(int(m.groups()[1]))
        return cls.parse_pseudopotential_from_turborvb_format_test(text=text, z_core=z_core, element=element)

    @classmethod
    def parse_pseudopotential_from_turborvb_format_test(cls, text: str, z_core: int, element: str):
        ang_mom = []
        exponent_list = []
        coefficient_list = []
        power_list = []

        lines = text.split("\n")
        c = [line for line in lines if re.match(r"^\s*\S+.*", line)]  # remove blank lines

        dummy_name = str(c[0])
        dummy_index, cutoff, lmax_plus_2 = c[1].split()
        dummy_index = int(dummy_index)
        cutoff = float(cutoff)
        max_ang_mom_plus_1 = int(lmax_plus_2)
        num_components = list(map(int, c[2].split()))
        # the last component is the local part.
        # l is in the ascending order from l=0 to l=lmax
        stride = 3
        for l, num_component in enumerate(num_components):
            for num in range(num_component):
                coefficient, power, exponent = c[stride].split()
                exponent = float(exponent)
                power = int(power)
                coefficient = float(coefficient)

                ang_mom.append(l)
                exponent_list.append(exponent)
                coefficient_list.append(coefficient)
                power_list.append(power - 2)

                stride += 1

        return cls(
            max_ang_mom_plus_1=max_ang_mom_plus_1,
            z_core=z_core,
            cutoff=cutoff,
            ang_mom=ang_mom,
            element=element,
            exponent=exponent_list,
            coefficient=coefficient_list,
            power=power_list,
        )

    @classmethod
    def parse_pseudopotential_from_eCEPP_format_file(cls, file: str):
        with open(file, "r") as f:
            text = f.readlines()
        text = "".join(text)
        return cls.parse_pseudopotential_from_eCEPP_format_text(text=text)

    @classmethod
    def parse_pseudopotential_from_eCEPP_format_text(cls, text: str):
        ang_mom = []
        exponent_list = []
        coefficient_list = []
        power_list = []

        lines = text.split("\n")
        lines = [line for line in lines if re.match(r"^\s*\S+.*", line)]  # remove blank lines
        logger.info(lines[0])
        _, element, z_core, max_ang_mom_plus_1, _ = lines[0].split(",")
        element = str(element)
        z_core = int(z_core)
        max_ang_mom_plus_1 = int(max_ang_mom_plus_1)
        cutoff = 0.0

        stride = 1
        local_part = True
        ang_mom_i = 0
        while stride < len(lines[1:]):
            num_component, _ = lines[stride].split()
            num_component = int(num_component.rstrip(";"))
            stride += 1
            if local_part:
                l = max_ang_mom_plus_1
                local_part = False
            else:
                l = ang_mom_i
                ang_mom_i += 1

            for i in range(num_component):
                r, exponent, coefficient = lines[stride].rstrip(";").split(",")
                r = int(r)
                exponent = float(exponent)
                coefficient = float(coefficient)
                stride += 1

                ang_mom.append(l)
                power_list.append(r - 2)
                exponent_list.append(exponent)
                coefficient_list.append(coefficient)

        return cls(
            max_ang_mom_plus_1=max_ang_mom_plus_1,
            z_core=z_core,
            cutoff=cutoff,
            ang_mom=ang_mom,
            element=element,
            exponent=exponent_list,
            coefficient=coefficient_list,
            power=power_list,
        )

    def to_text_gamess_format(self):
        text = ""
        text += f"{self.element:s}  GEN  {self.z_core:d}  {self.max_ang_mom_plus_1:d}\n"
        l_index = [i for i, x in enumerate(self.ang_mom) if x == self.max_ang_mom_plus_1]
        text += f"{len(l_index)}\n"
        for ll in l_index:
            text += f"{self.coefficient[ll]:.8f}  {self.power[ll] + 2:d}  {self.exponent[ll]:.8f}\n"

        for l in range(self.max_ang_mom_plus_1):
            l_index = [i for i, x in enumerate(self.ang_mom) if x == l]
            text += f"{len(l_index)}\n"
            for ll in l_index:
                text += f"{self.coefficient[ll]:.8f}  {self.power[ll] + 2:d}  {self.exponent[ll]:.8f}\n"

        return text

    def to_text_nwchem_format(self):
        text = ""
        text += f"{self.element:s} nelec {self.z_core:d}\n"
        l_index = [i for i, x in enumerate(self.ang_mom) if x == self.max_ang_mom_plus_1]
        text += f"{self.element:s} ul\n"
        for ll in l_index:
            text += f"  {self.power[ll] + 2:d}  {self.exponent[ll]:.8f}  {self.coefficient[ll]:.8f}\n"

        for l in range(self.max_ang_mom_plus_1):
            l_index = [i for i, x in enumerate(self.ang_mom) if x == l]
            text += f"{self.element:s} {return_orbchr(l).upper()}\n"
            for ll in l_index:
                text += f"  {self.power[ll] + 2:d}  {self.exponent[ll]:.8f}  {self.coefficient[ll]:.8f}\n"

        return text


def main():
    # parser.add_argument
    import argparse

    parser = argparse.ArgumentParser(
        description="This program is a python-based script for converting a TREXIO file to a TurboRVB Wavefunction file"
    )
    parser.add_argument("trexio_file", help="Name of TREXIO file")
    args = parser.parse_args()

    logger = getLogger("pyturbo").getChild(__name__)
    logger.setLevel(args.loglevel)
    stream_handler = StreamHandler()
    stream_handler.setLevel(args.loglevel)
    if args.loglevel in {"DEBUG"}:
        handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    else:
        handler_format = Formatter("%(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    logger.info("    ")
    logger.info("Program   : TREXIO-TurboRVB converter")
    logger.info("Author    : 2022 Kosuke Nakano (SISSA/JAIST)")
    logger.info("E-mail    : kousuke_1123@icloud.com")
    logger.info("Website   : https://www.kosuke-nakano-research.info")
    logger.info("License   : MIT lisence")
    logger.info("    ")
    logger.info("Please cite the following paper when publishing your work(s).")
    logger.info("    ")
    logger.info("TurboRVB:")
    logger.info("    K. Nakano et al. J. Chem. Phys., 152, 204121 (2020)")
    logger.info("Turbo-Genius:")
    logger.info("    K. Nakano et al. in preparation.")
    logger.info("TREXIO:")
    logger.info("    TREX project, in preparation.")
    logger.info("    ")


if __name__ == "__main__":
    logger = getLogger("pyturbo")
    logger.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    main()

    # moved to examples
    """
    ## basis set downloads
    pseudopotentials_test_dir = os.path.join(
        pyturbo_root, "tests", "pseudopotentials"
    )
    os.chdir(pseudopotentials_test_dir)
    file_name = "pp_eCEPP_H"
    pseudopotentials = (
        Pseudopotentials.parse_pseudopotential_from_turborvb_pseudo_dat(
            file="pseudo.dat"
        )
    )
    pseudopotentials.set_cutoffs()
    """
