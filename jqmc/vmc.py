"""VMC module"""

# python modules
import os
import random
from logging import Formatter, StreamHandler, getLogger

# import mpi4jax
# JAX
import jax
import numpy as np
from jax import grad

# MPI
from mpi4py import MPI

from .hamiltonians import Hamiltonian_data, compute_local_energy
from .jastrow_factor import Jastrow_data, Jastrow_two_body_data
from .structure import find_nearest_index
from .swct import SWCT_data, evaluate_swct_omega_api
from .trexio_wrapper import read_trexio_file
from .wavefunction import Wavefunction_data, evaluate_ln_wavefunction_api, evaluate_wavefunction_api

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


class MCMC:
    def __init__(
        self,
        hamiltonian_data: Hamiltonian_data = None,
        swct_data: SWCT_data = None,
        mcmc_seed: int = 34467,
        Dt: float = 2.0,
    ) -> None:
        """
        Initialize a MCMC class.

        Args:
            mcmc_seed (int): seed for the MCMC chain.
            hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data
        Returns:
            None
        """
        self.__hamiltonian_data = hamiltonian_data
        self.__swct_data = swct_data
        self.__mcmc_seed = mcmc_seed
        self.__Dt = Dt

        # mcmc counter
        self.__mcmc_counter = 0

        # latest electron positions
        self.__latest_r_up_carts = None
        self.__latest_r_dn_carts = None

        # stored electron positions
        self.__stored_r_up_carts = None
        self.__stored_r_dn_carts = None

        # stored local energy (e_L)
        self.__stored_e_L = []

        # stored de_L / dR
        self.__stored_grad_e_L_dR = []

        # stored de_L / dr_up
        self.__stored_grad_e_L_r_up = []

        # stored de_L / dr_dn
        self.__stored_grad_e_L_r_dn = []

        # stored ln_Psi
        self.__stored_ln_Psi = []

        # stored dln_Psi / dr_up
        self.__stored_grad_ln_Psi_r_up = []

        # stored dln_Psi / dr_dn
        self.__stored_grad_ln_Psi_r_dn = []

        # stored dln_Psi / dR
        self.__stored_grad_ln_Psi_dR = []

        # intialize all attributes
        self.__init__attributes()

    def __init__attributes(self):
        # set the initial electron configurations
        num_electron_up = self.__hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
        num_electron_dn = self.__hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn

        # Initialization
        r_carts_up = []
        r_carts_dn = []

        total_electrons = 0

        if self.__hamiltonian_data.coulomb_potential_data.ecp_flag:
            charges = np.array(self.__hamiltonian_data.structure_data.atomic_numbers) - np.array(
                self.__hamiltonian_data.coulomb_potential_data.z_cores
            )
        else:
            charges = np.array(self.__hamiltonian_data.structure_data.atomic_numbers)

        coords = self.__hamiltonian_data.structure_data.positions_cart

        # Place electrons around each nucleus
        for i in range(len(coords)):
            charge = charges[i]
            num_electrons = int(
                np.round(charge)
            )  # Number of electrons to place based on the charge

            # Retrieve the position coordinates
            x, y, z = coords[i]

            # Place electrons
            for _ in range(num_electrons):
                # Calculate distance range
                distance = np.random.uniform(0.1, 2.0)
                theta = np.random.uniform(0, np.pi)
                phi = np.random.uniform(0, 2 * np.pi)

                # Convert spherical to Cartesian coordinates
                dx = distance * np.sin(theta) * np.cos(phi)
                dy = distance * np.sin(theta) * np.sin(phi)
                dz = distance * np.cos(theta)

                # Position of the electron
                electron_position = np.array([x + dx, y + dy, z + dz])

                # Assign spin
                if len(r_carts_up) < num_electron_up:
                    r_carts_up.append(electron_position)
                else:
                    r_carts_dn.append(electron_position)

            total_electrons += num_electrons

        # Handle surplus electrons
        remaining_up = num_electron_up - len(r_carts_up)
        remaining_dn = num_electron_dn - len(r_carts_dn)

        # Randomly place any remaining electrons
        for _ in range(remaining_up):
            r_carts_up.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))
        for _ in range(remaining_dn):
            r_carts_dn.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))

        self.__latest_r_up_carts = np.array(r_carts_up)
        self.__latest_r_dn_carts = np.array(r_carts_dn)

        logger.info(f"initial r_up_carts = { self.__latest_r_up_carts}")
        logger.info(f"initial r_dn_carts = { self.__latest_r_dn_carts}")

        # reset mcmc counter
        self.__mcmc_counter = 0

        # stored local energy (e_L)
        self.__stored_e_L = []

        # stored de_L / dR
        self.__stored_grad_e_L_dR = []

        # stored de_L / dr_up
        self.__stored_grad_e_L_r_up = []

        # stored de_L / dr_dn
        self.__stored_grad_e_L_r_dn = []

        # stored ln_Psi
        self.__stored_ln_Psi = []

        # stored dln_Psi / dr_up
        self.__stored_grad_ln_Psi_r_up = []

        # stored dln_Psi / dr_dn
        self.__stored_grad_ln_Psi_r_dn = []

        # stored dln_Psi / dR
        self.__stored_grad_ln_Psi_dR = []

        # stored Omega_up (SWCT)
        self.__stored_omega_up = []

        # stored Omega_dn (SWCT)
        self.__stored_omega_dn = []

        # stored sum_i d omega/d r_i for up spins (SWCT)
        self.__stored_grad_omega_r_up = []

        # stored sum_i d omega/d r_i for dn spins (SWCT)
        self.__stored_grad_omega_r_dn = []

        # """
        # compiling methods
        logger.info("Compilation starts.")

        logger.info("  Compilation e_L starts.")
        _ = compute_local_energy(
            hamiltonian_data=self.__hamiltonian_data,
            r_up_carts=self.__latest_r_up_carts,
            r_dn_carts=self.__latest_r_dn_carts,
        )
        logger.info("  Compilation e_L is done.")

        logger.info("  Compilation de_L starts.")
        _, _, _ = grad(compute_local_energy, argnums=(0, 1, 2))(
            self.__hamiltonian_data,
            self.__latest_r_up_carts,
            self.__latest_r_dn_carts,
        )
        logger.info("  Compilation de_L is done.")

        logger.info("  Compilation dln_Psi starts.")
        _, _, _ = grad(evaluate_ln_wavefunction_api, argnums=(0, 1, 2))(
            self.__hamiltonian_data.wavefunction_data,
            self.__latest_r_up_carts,
            self.__latest_r_dn_carts,
        )
        logger.info("  Compilation dln_Psi is done.")

        logger.info("  Compilation domega starts.")
        _ = grad(evaluate_swct_omega_api, argnums=1)(
            self.__swct_data,
            self.__latest_r_up_carts,
        )
        logger.info("  Compilation domega is done.")

        logger.info("Compilation is done.")
        # """

    def run(self, num_mcmc_steps: int = 0, continuation: int = 0) -> None:
        """
        Args:
            num_mcmc_steps (int): the number of total mcmc steps
            continuation (int): 1 = VMC run from sctach, 0 = VMC run continuataion
        Returns:
            None
        """
        cpu_count = os.cpu_count()
        logger.info(f"cpu count = {cpu_count}")

        # Set the random seed. Use the Mersenne Twister generator
        accepted_moves = 0
        nbra = 16
        random.seed(self.__mcmc_seed)
        np.random.seed(self.__mcmc_seed)

        # MAIN MCMC loop from here !!!
        for i_mcmc_step in range(num_mcmc_steps):
            logger.info(f"Current MCMC step = {i_mcmc_step+1}/{num_mcmc_steps}.")

            # Determine the total number of electrons
            total_electrons = len(self.__latest_r_up_carts) + len(self.__latest_r_dn_carts)

            if self.__hamiltonian_data.coulomb_potential_data.ecp_flag:
                charges = np.array(
                    self.__hamiltonian_data.structure_data.atomic_numbers
                ) - np.array(self.__hamiltonian_data.coulomb_potential_data.z_cores)
            else:
                charges = np.array(self.__hamiltonian_data.structure_data.atomic_numbers)

            coords = self.__hamiltonian_data.structure_data.positions_cart

            for _ in range(nbra):
                # Choose randomly if the electron comes from up or dn
                if random.randint(0, total_electrons - 1) < len(self.__latest_r_up_carts):
                    selected_electron_spin = "up"
                    # Randomly select an electron from r_carts_up
                    selected_electron_index = random.randint(0, len(self.__latest_r_up_carts) - 1)

                    old_r_cart = self.__latest_r_up_carts[selected_electron_index]
                else:
                    selected_electron_spin = "dn"
                    # Randomly select an electron from r_carts_dn
                    selected_electron_index = random.randint(0, len(self.__latest_r_dn_carts) - 1)
                    old_r_cart = self.__latest_r_dn_carts[selected_electron_index]

                nearest_atom_index = find_nearest_index(
                    self.__hamiltonian_data.structure_data, old_r_cart
                )

                R_cart = coords[nearest_atom_index]
                Z = charges[nearest_atom_index]
                norm_r_R = np.linalg.norm(old_r_cart - R_cart)
                f_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)

                logger.debug(f"nearest_atom_index = {nearest_atom_index}")
                logger.debug(f"norm_r_R = {norm_r_R}")
                logger.debug(f"f_l  = {f_l }")

                sigma = f_l * self.__Dt
                g = float(np.random.normal(loc=0, scale=sigma))
                g_vector = np.zeros(3)
                random_index = np.random.randint(0, 3)
                g_vector[random_index] = g
                logger.debug(f"jn = {random_index}, g \equiv dstep  = {g_vector}")
                new_r_cart = old_r_cart + g_vector

                if selected_electron_spin == "up":
                    proposed_r_up_carts = self.__latest_r_up_carts.copy()
                    proposed_r_dn_carts = self.__latest_r_dn_carts.copy()
                    proposed_r_up_carts[selected_electron_index] = new_r_cart
                else:
                    proposed_r_up_carts = self.__latest_r_up_carts.copy()
                    proposed_r_dn_carts = self.__latest_r_dn_carts.copy()
                    proposed_r_dn_carts[selected_electron_index] = new_r_cart

                nearest_atom_index = find_nearest_index(
                    self.__hamiltonian_data.structure_data, new_r_cart
                )

                R_cart = coords[nearest_atom_index]
                Z = charges[nearest_atom_index]
                norm_r_R = np.linalg.norm(new_r_cart - R_cart)
                f_prime_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)
                logger.debug(f"nearest_atom_index = {nearest_atom_index}")
                logger.debug(f"norm_r_R = {norm_r_R}")
                logger.debug(f"f_prime_l  = {f_prime_l }")

                logger.debug(
                    f"The selected electron is {selected_electron_index+1}-th {selected_electron_spin} electron."
                )
                logger.debug(f"The selected electron position is {old_r_cart}.")
                logger.debug(f"The proposed electron position is {new_r_cart}.")

                T_ratio = (f_l / f_prime_l) * np.exp(
                    -(np.linalg.norm(new_r_cart - old_r_cart) ** 2)
                    * (
                        1.0 / (2.0 * f_prime_l**2 * self.__Dt**2)
                        - 1.0 / (2.0 * f_l**2 * self.__Dt**2)
                    )
                )

                R_ratio = (
                    evaluate_wavefunction_api(
                        wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                        r_up_carts=proposed_r_up_carts,
                        r_dn_carts=proposed_r_dn_carts,
                    )
                    / evaluate_wavefunction_api(
                        wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                        r_up_carts=self.__latest_r_up_carts,
                        r_dn_carts=self.__latest_r_dn_carts,
                    )
                ) ** 2.0

                logger.debug(f"R_ratio, T_ratio = {R_ratio}, {T_ratio}")
                acceptance_ratio = np.min([1.0, R_ratio * T_ratio])
                logger.debug(f"acceptance_ratio = {acceptance_ratio}")

                b = np.random.uniform(0, 1)

                if b < acceptance_ratio:
                    logger.debug("The proposed move is accepted!")
                    accepted_moves += 1
                    self.__latest_r_up_carts = proposed_r_up_carts
                    self.__latest_r_dn_carts = proposed_r_dn_carts
                else:
                    logger.debug("The proposed move is rejected!")

            # evaluate observables
            e_L = compute_local_energy(
                hamiltonian_data=self.__hamiltonian_data,
                r_up_carts=self.__latest_r_up_carts,
                r_dn_carts=self.__latest_r_dn_carts,
            )
            logger.info(f"e_L = {e_L}")
            self.__stored_e_L.append(e_L)

            # """
            grad_e_L_h, grad_e_L_r_up, grad_e_L_r_dn = grad(
                compute_local_energy, argnums=(0, 1, 2)
            )(
                self.__hamiltonian_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )
            self.__stored_grad_e_L_r_up.append(grad_e_L_r_up)
            self.__stored_grad_e_L_r_dn.append(grad_e_L_r_dn)

            grad_e_L_R = (
                grad_e_L_h.wavefunction_data.geminal_data.orb_data_up_spin.aos_data.structure_data.positions
                + grad_e_L_h.wavefunction_data.geminal_data.orb_data_dn_spin.aos_data.structure_data.positions
                + grad_e_L_h.coulomb_potential_data.structure_data.positions
            )
            self.__stored_grad_e_L_dR.append(grad_e_L_R)
            # """

            # """
            logger.debug(
                f"de_L_dR(AOs_data_up) = {grad_e_L_h.wavefunction_data.geminal_data.orb_data_up_spin.aos_data.structure_data.positions}"
            )
            logger.debug(
                f"de_L_dR(AOs_data_dn) = {grad_e_L_h.wavefunction_data.geminal_data.orb_data_dn_spin.aos_data.structure_data.positions}"
            )
            logger.debug(
                f"de_L_dR(coulomb_potential_data) = {grad_e_L_h.coulomb_potential_data.structure_data.positions}"
            )
            logger.info(f"de_L_dR = {grad_e_L_R}")
            logger.info(f"de_L_dr_up = {grad_e_L_r_up}")
            logger.info(f"de_L_dr_dn= {grad_e_L_r_dn}")
            # """

            ln_Psi = evaluate_ln_wavefunction_api(
                wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                r_up_carts=self.__latest_r_up_carts,
                r_dn_carts=self.__latest_r_dn_carts,
            )
            logger.info(f"ln_Psi = {ln_Psi}")
            self.__stored_ln_Psi.append(ln_Psi)

            # """
            grad_ln_Psi_h, grad_ln_Psi_r_up, grad_ln_Psi_r_dn = grad(
                evaluate_ln_wavefunction_api, argnums=(0, 1, 2)
            )(
                self.__hamiltonian_data.wavefunction_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )
            logger.info(f"dln_Psi_dr_up = {grad_ln_Psi_r_up}")
            logger.info(f"dln_Psi_dr_dn = {grad_ln_Psi_r_dn}")
            self.__stored_grad_ln_Psi_r_up.append(grad_ln_Psi_r_up)
            self.__stored_grad_ln_Psi_r_dn.append(grad_ln_Psi_r_dn)

            grad_ln_Psi_dR = (
                grad_ln_Psi_h.geminal_data.orb_data_up_spin.aos_data.structure_data.positions
                + grad_ln_Psi_h.geminal_data.orb_data_dn_spin.aos_data.structure_data.positions
            )

            # stored dln_Psi / dR
            logger.info(f"dln_Psi_dR = {grad_ln_Psi_dR}")
            self.__stored_grad_ln_Psi_dR.append(grad_ln_Psi_dR)
            # """

            omega_up = evaluate_swct_omega_api(
                swct_data=self.__swct_data,
                r_carts=self.__latest_r_up_carts,
            )

            omega_dn = evaluate_swct_omega_api(
                swct_data=self.__swct_data,
                r_carts=self.__latest_r_dn_carts,
            )

            logger.info(f"omega_up = {omega_up}")
            logger.info(f"omega_dn = {omega_dn}")

            self.__stored_omega_up.append(omega_up)
            self.__stored_omega_dn.append(omega_dn)

            grad_omega_dr_up = grad(evaluate_swct_omega_api, argnums=(1))(
                self.__swct_data,
                self.__latest_r_up_carts,
            )

            grad_omega_dr_dn = grad(evaluate_swct_omega_api, argnums=(1))(
                self.__swct_data,
                self.__latest_r_dn_carts,
            )

            logger.info(f"grad_omega_dr_up = {grad_omega_dr_up}")
            logger.info(f"grad_omega_dr_dn = {grad_omega_dr_dn}")

            self.__stored_grad_omega_r_up.append(grad_omega_dr_up)
            self.__stored_grad_omega_r_dn.append(grad_omega_dr_dn)

        logger.info(f"acceptance ratio is {accepted_moves/num_mcmc_steps/nbra*100} %")

    @property
    def e_L(self):
        return self.__stored_e_L

    @property
    def de_L_dR(self):
        return self.__stored_grad_e_L_dR

    @property
    def de_L_dr_up(self):
        return self.__stored_grad_e_L_r_up

    @property
    def de_L_dr_dn(self):
        return self.__stored_grad_e_L_r_dn

    @property
    def dln_Psi_dr_up(self):
        return self.__stored_grad_ln_Psi_r_up

    @property
    def dln_Psi_dr_dn(self):
        return self.__stored_grad_ln_Psi_r_dn

    @property
    def dln_Psi_dR(self):
        return self.__stored_grad_ln_Psi_dR


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    log = getLogger("jqmc")
    log.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter(
        f"MPI-rank={rank}: %(name)s - %(levelname)s - %(lineno)d - %(message)s"
    )
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    # water  cc-pVTZ with Mitas ccECP.
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "water_trexio.hdf5"))

    # define data
    jastrow_two_body_data = Jastrow_two_body_data(
        param_parallel_spin=0.0, param_anti_parallel_spin=0.0
    )
    jastrow_data = Jastrow_data(
        jastrow_two_body_data=jastrow_two_body_data, jastrow_two_body_type="off"
    )  # no jastrow for the time-being.

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    swct_data = SWCT_data(structure=structure_data)

    # VMC parameters
    num_mcmc_warmup_steps = 50
    num_mcmc_bin_blocks = 10
    mpi_seed = 34356 * (rank + 1)

    logger.info(f"mcmc_seed for MPI-rank={rank} is {mpi_seed}.")
    if rank == 0:
        logger.info(f"num_mcmc_warmup_steps={num_mcmc_warmup_steps}.")
        logger.info(f"num_mcmc_bin_blocks={num_mcmc_bin_blocks}.")

    # run VMC
    mcmc = MCMC(hamiltonian_data=hamiltonian_data, swct_data=swct_data, mcmc_seed=mpi_seed)
    mcmc.run(num_mcmc_steps=100)
    e_L = mcmc.e_L[num_mcmc_warmup_steps:]

    e_L_split = np.array_split(e_L, num_mcmc_bin_blocks)
    e_L_binned = [np.average(e_list) for e_list in e_L_split]

    logger.debug(f"e_L_binned for MPI-rank={rank} is {e_L_binned}.")

    e_L_binned = comm.reduce(e_L_binned, op=MPI.SUM, root=0)

    if rank == 0:
        logger.debug(f"e_L_binned = {e_L_binned}.")
        # jackknife implementation
        # https://www2.yukawa.kyoto-u.ac.jp/~etsuko.itou/old-HP/Notes/Jackknife-method.pdf
        e_L_jackknife_binned = [
            np.average(np.delete(e_L_binned, i)) for i in range(len(e_L_binned))
        ]

        logger.debug(f"e_L_jackknife_binned  = {e_L_jackknife_binned}.")

        e_L_mean = np.average(e_L_jackknife_binned)
        e_L_std = np.sqrt(len(e_L_binned) - 1) * np.std(e_L_jackknife_binned)

        logger.info(f"e_L = {e_L_mean} +- {e_L_std} Ha.")

    """

    old_r_up_carts = np.array(
        [
            [-0.64878536, -0.83275288, 0.33532629],
            [0.55271273, 0.72310605, 0.93443775],
            [0.66767275, 0.1206456, -0.36521208],
            [-0.93165236, -0.0120386, 0.33003036],
        ]
    )
    old_r_dn_carts = np.array(
        [
            [-1.0347816, 1.26162081, 0.42301735],
            [-0.57843435, 1.03651987, -0.55091542],
            [-1.56091964, -0.58952149, -0.99268141],
            [0.61863233, -0.14903326, 0.51962683],
        ]
    )
    new_r_up_carts = old_r_up_carts.copy()
    new_r_dn_carts = old_r_dn_carts.copy()

    old_r_cart = [0.61863233, -0.14903326, 0.51962683]
    new_r_cart = [0.618632327645002, -0.149033260668010, 0.131889254514777]
    new_r_dn_carts[3] = new_r_cart

    R_ratio = (
        evaluate_wavefunction(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=new_r_up_carts,
            r_dn_carts=new_r_dn_carts,
        )
        / evaluate_wavefunction(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=old_r_up_carts,
            r_dn_carts=old_r_dn_carts,
        )
    ) ** 2.0

    logger.debug(f"R_ratio = {R_ratio}")

    if hamiltonian_data.coulomb_potential_data.ecp_flag:
        charges = np.array(hamiltonian_data.structure_data.atomic_numbers) - np.array(
            hamiltonian_data.coulomb_potential_data.z_cores
        )
    else:
        charges = np.array(hamiltonian_data.structure_data.atomic_numbers)

    coords = hamiltonian_data.structure_data.positions_cart

    nearest_atom_index = (
        hamiltonian_data.structure_data.get_nearest_neigbhor_atom_index(old_r_cart)
    )

    R_cart = coords[nearest_atom_index]
    Z = charges[nearest_atom_index]
    norm_r_R = np.linalg.norm(old_r_cart - R_cart)
    f_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)
    logger.debug(f"f_l = {f_l}")

    nearest_atom_index = (
        hamiltonian_data.structure_data.get_nearest_neigbhor_atom_index(new_r_cart)
    )
    R_cart = coords[nearest_atom_index]
    Z = charges[nearest_atom_index]
    norm_r_R = np.linalg.norm(new_r_cart - R_cart)
    f_prime_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)
    logger.debug(f"f_prime_l  = {f_prime_l }")

    T_ratio = (f_l / f_prime_l) * np.exp(
        -np.linalg.norm(np.array(new_r_cart) - np.array(old_r_cart)) ** 2
        * (1.0 / (2.0 * f_prime_l**2 * 2.0**2) - 1.0 / (2.0 * f_l**2 * 2.0**2))
    )

    logger.debug(f"T_ratio = {T_ratio}")

    kinc = compute_kinetic_energy(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare = compute_bare_coulomb_potential(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_ecp_local = compute_ecp_local_parts(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_ecp_nonlocal = compute_ecp_nonlocal_parts(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    logger.debug(f"kinc={kinc} Ha")
    logger.debug(f"vpot_bare={vpot_bare} Ha")
    logger.debug(f"vpot_ecp_local={vpot_ecp_local} Ha")
    logger.debug(f"vpot_ecp_nonlocal={vpot_ecp_nonlocal} Ha")

    logger.debug(f"kinc={kinc} Ha")
    logger.debug(f"vpot={vpot_bare+vpot_ecp_local} Ha")
    logger.debug(f"vpotoff={vpot_ecp_nonlocal} Ha")
    """

    """
    e_L = compute_local_energy(
        hamiltonian_data=hamiltonian_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )
    print(f"e_L={e_L} Ha")
    """

    """
    from coulomb_potential import compute_ecp_local_parts, compute_ecp_nonlocal_parts

    num_electron_up = geminal_mo_data.num_electron_up
    num_electron_dn = geminal_mo_data.num_electron_dn

    # Initialization
    r_up_carts = []
    r_dn_carts = []

    total_electrons = 0

    if coulomb_potential_data.ecp_flag:
        charges = np.array(structure_data.atomic_numbers) - np.array(
            coulomb_potential_data.z_cores
        )
    else:
        charges = np.array(structure_data.atomic_numbers)

    coords = structure_data.positions_cart

    # Place electrons around each nucleus
    for i in range(len(coords)):
        charge = charges[i]
        num_electrons = int(
            np.round(charge)
        )  # Number of electrons to place based on the charge

        # Retrieve the position coordinates
        x, y, z = coords[i]

        # Place electrons
        for _ in range(num_electrons):
            # Calculate distance range
            distance = np.random.uniform(1.0 / charge, 2.0 / charge)
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)

            # Convert spherical to Cartesian coordinates
            dx = distance * np.sin(theta) * np.cos(phi)
            dy = distance * np.sin(theta) * np.sin(phi)
            dz = distance * np.cos(theta)

            # Position of the electron
            electron_position = np.array([x + dx, y + dy, z + dz])

            # Assign spin
            if len(r_up_carts) < num_electron_up:
                r_up_carts.append(electron_position)
            else:
                r_dn_carts.append(electron_position)

        total_electrons += num_electrons

    # Handle surplus electrons
    remaining_up = num_electron_up - len(r_up_carts)
    remaining_dn = num_electron_dn - len(r_dn_carts)

    # Randomly place any remaining electrons
    for _ in range(remaining_up):
        r_up_carts.append(
            np.random.choice(coords) + np.random.normal(scale=0.1, size=3)
        )
    for _ in range(remaining_dn):
        r_dn_carts.append(
            np.random.choice(coords) + np.random.normal(scale=0.1, size=3)
        )

    r_up_carts = np.array(r_up_carts)
    r_dn_carts = np.array(r_dn_carts)

    V_local = compute_ecp_local_parts(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    print(V_local)

    V_nonlocal = compute_ecp_nonlocal_parts(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    print(V_nonlocal)
    """
