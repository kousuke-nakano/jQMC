from pyscf import gto, scf
from pyscf.tools import trexio

filename = "N_ccecp_ccpvqz_cart.h5"

mol = gto.Mole()
mol.verbose = 5
mol.atom = """
               N    0.00000000   0.00000000   0.00000000
               """
mol.basis = "ccecp-ccpvqz"
mol.unit = "A"
mol.ecp = "ccecp"
mol.charge = 0
mol.spin = 3
mol.symmetry = False
mol.cart = True
mol.output = "N.out"
mol.build()

mf = scf.UHF(mol)
mf.max_cycle = 200
mf_scf = mf.kernel()

trexio.to_trexio(mf, filename)
