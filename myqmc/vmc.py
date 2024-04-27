"""VMC module"""

# python modules
import random
import numpy as np

# import numpy.typing as npt

# set logger
from logging import getLogger, StreamHandler, Formatter

from hamiltonians import Hamiltonian_data, compute_local_energy
from wavefunction import compute_quantum_force, evaluate_wavefunction
from trexio_wrapper import read_trexio_file
from wavefunction import Wavefunction_data

logger = getLogger("myqmc").getChild(__name__)


class MCMC:
    def __init__(
        self,
        hamiltonian_data: Hamiltonian_data = None,
        mcmc_seed: int = 34467,
        Dt: float = 0.10,
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

        # stored local energy
        self.__stored_local_energy = None

        # intialize all attributes
        self.__init__attributes()

    def __init__attributes(self):

        # set the initial electron configurations
        num_electron_up = (
            self.__hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
        )
        num_electron_dn = (
            self.__hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
        )

        # Initialization
        r_carts_up = []
        r_carts_dn = []

        total_electrons = 0

        if self.__hamiltonian_data.coulomb_potential_data.ecp_flag:
            charges = np.array(
                self.__hamiltonian_data.structure_data.atomic_numbers
            ) - np.array(self.__hamiltonian_data.coulomb_potential_data.z_cores)
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
                distance = np.random.uniform(2.5 / charge, 3.5 / charge)
                distance = 1.0
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
            r_carts_up.append(
                np.random.choice(coords) + np.random.normal(scale=0.1, size=3)
            )
        for _ in range(remaining_dn):
            r_carts_dn.append(
                np.random.choice(coords) + np.random.normal(scale=0.1, size=3)
            )

        self.__latest_r_up_carts = np.array(r_carts_up)
        self.__latest_r_dn_carts = np.array(r_carts_dn)

        logger.info(f"initial r_up_carts = { self.__latest_r_up_carts}")
        logger.info(f"initial r_dn_carts = { self.__latest_r_dn_carts}")

        # reset mcmc counter
        self.__mcmc_counter = 0

        # reset stored electron positions
        self.__stored_r_up_carts = []
        self.__stored_r_dn_carts = []

        # stored local energy
        self.__stored_local_energy = []

    def run(self, num_mcmc_steps: int = 0, continuation: int = 0) -> None:
        """
        Args:
            num_mcmc_steps (int): the number of total mcmc steps
            continuation (int): 1 = VMC run from sctach, 0 = VMC run continuataion
        Returns:
            None
        """

        # Set the random seed and use the Mersenne Twister generator
        accepted_moves = 0
        random.seed(self.__mcmc_seed)
        np.random.seed(self.__mcmc_seed)

        # MAIN MCMC loop from here !!!
        for i_mcmc_step in range(num_mcmc_steps):
            logger.info(f"Current MCMC step = {i_mcmc_step+1}/{num_mcmc_steps}.")

            # Determine the total number of electrons
            total_electrons = len(self.__latest_r_up_carts) + len(
                self.__latest_r_dn_carts
            )

            # compute quantum forces
            qF_up, qF_dn = compute_quantum_force(
                wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                r_up_carts=self.__latest_r_up_carts,
                r_dn_carts=self.__latest_r_dn_carts,
            )

            # 3次元ガウス分布ベクトルを生成する
            # np.random.normalは指定された平均、標準偏差でランダムな値を生成する
            sigma = np.sqrt(2 * self.__Dt)
            g_vector = np.random.normal(loc=0, scale=sigma, size=3)

            # Choose randomly if the electron comes from up or dn
            if random.randint(0, total_electrons - 1) < len(self.__latest_r_up_carts):
                selected_electron_spin = "up"
                # Randomly select an electron from r_carts_up
                selected_electron_index = random.randint(
                    0, len(self.__latest_r_up_carts) - 1
                )
                old_r_l = self.__latest_r_up_carts[selected_electron_index]
                old_qF_l = qF_up[selected_electron_index]
                new_r_l = old_r_l + self.__Dt * old_qF_l + g_vector

                proposed_r_up_carts = self.__latest_r_up_carts.copy()
                proposed_r_dn_carts = self.__latest_r_dn_carts.copy()
                proposed_r_up_carts[selected_electron_index] = new_r_l

                # compute quantum forces
                new_qF_l, _ = compute_quantum_force(
                    wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                    r_up_carts=proposed_r_up_carts,
                    r_dn_carts=proposed_r_dn_carts,
                )
                new_qF_l = new_qF_l[selected_electron_index]

            else:
                selected_electron_spin = "dn"
                # Randomly select an electron from r_carts_dn
                selected_electron_index = random.randint(
                    0, len(self.__latest_r_dn_carts) - 1
                )
                old_r_l = self.__latest_r_dn_carts[selected_electron_index]
                old_qF_l = qF_dn[selected_electron_index]
                new_r_l = old_r_l + self.__Dt * old_qF_l + g_vector

                proposed_r_up_carts = self.__latest_r_up_carts.copy()
                proposed_r_dn_carts = self.__latest_r_dn_carts.copy()
                proposed_r_dn_carts[selected_electron_index] = new_r_l

                # compute quantum forces
                _, new_qF_l = compute_quantum_force(
                    wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                    r_up_carts=proposed_r_up_carts,
                    r_dn_carts=proposed_r_dn_carts,
                )
                new_qF_l = new_qF_l[selected_electron_index]

            logger.info(
                f"The selected electron is {selected_electron_index+1}-th {selected_electron_spin} electron."
            )
            logger.info(f"The selected electron position is {old_r_l}.")
            logger.info(f"The proposed electron position is {new_r_l}.")

            logger.info(f"The old_qF_l is {old_qF_l}.")
            logger.info(f"The new_qF_l is {new_qF_l}.")

            T_forward = np.exp(
                -1.0
                * (np.linalg.norm(new_r_l - old_r_l - self.__Dt * old_qF_l) ** 2)
                / (4.0 * self.__Dt)
            )
            T_backward = np.exp(
                -1.0
                * (np.linalg.norm(old_r_l - new_r_l - self.__Dt * new_qF_l) ** 2)
                / (4.0 * self.__Dt)
            )
            R_ratio = (
                evaluate_wavefunction(
                    wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                    r_up_carts=proposed_r_up_carts,
                    r_dn_carts=proposed_r_dn_carts,
                )
                / evaluate_wavefunction(
                    wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                    r_up_carts=self.__latest_r_up_carts,
                    r_dn_carts=self.__latest_r_dn_carts,
                )
            ) ** 2.0

            logger.info(
                f"R_ratio, T_forward, T_backward = {R_ratio}, {T_forward}, {T_backward}"
            )
            acceptance_ratio = np.min([1.0, R_ratio * T_backward / T_forward])
            logger.info(f"acceptance_ratio = {acceptance_ratio}")

            b = np.random.uniform(0, 1)

            if b < acceptance_ratio:
                logger.info("The proposed move is accepted!")
                accepted_moves += 1
                self.__latest_r_up_carts = proposed_r_up_carts
                self.__latest_r_dn_carts = proposed_r_dn_carts
                e_L = compute_local_energy(
                    hamiltonian_data=self.__hamiltonian_data,
                    r_up_carts=self.__latest_r_up_carts,
                    r_dn_carts=self.__latest_r_dn_carts,
                )
            else:
                logger.info("The proposed move is rejected!")
                e_L = compute_local_energy(
                    hamiltonian_data=self.__hamiltonian_data,
                    r_up_carts=self.__latest_r_up_carts,
                    r_dn_carts=self.__latest_r_dn_carts,
                )

            logger.info(f"e_L = {e_L}")
            self.__stored_local_energy.append(e_L)

        logger.info(f"acceptance ratio is {accepted_moves/num_mcmc_steps*100} %")
        logger.info(
            f"e_L is {np.average(self.__stored_local_energy)} +- {np.sqrt(np.var(self.__stored_local_energy))} Ha."
        )


if __name__ == "__main__":
    log = getLogger("myqmc")
    log.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    # water
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file="water_trexio.hdf5")

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    # """
    mcmc = MCMC(hamiltonian_data=hamiltonian_data)
    mcmc.run(num_mcmc_steps=100)
    # """

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
