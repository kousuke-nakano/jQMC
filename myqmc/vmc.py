"""VMC module"""

# python modules
import numpy as np
import numpy.typing as npt

# set logger
from logging import getLogger, StreamHandler, Formatter

from hamiltonians import Hamiltonian_data

logger = getLogger("myqmc").getChild(__name__)


class MCMC:
    def __init__(
        self, hamiltonian_data: Hamiltonian_data = None, mcmc_seed: int = 34467
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

    def __init__attributes(self):
        self.__latest_r_up_carts = None
        self.__latest_r_dn_carts = None

        # reset mcmc counter
        self.__mcmc_counter = 0

        # reset stored electron positions
        self.__stored_r_up_carts = []
        self.__stored_r_dn_carts = []

        # stored local energy
        self.__stored_local_energy = []

    def run(num_mcmc_steps: int = 0, continuation: int = 0) -> None:
        """
        Args:
            num_mcmc_steps (int): the number of total mcmc steps
            continuation (int): 1 = VMC run from sctach, 0 = VMC run continuataion
        Returns:
            None
        """

        pass


if __name__ == "__main__":
    log = getLogger("myqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)
