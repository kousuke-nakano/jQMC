from pyscf import gto, scf
from pyscf.tools import trexio

filename = "O2_ccecp_ccpvqz_sphe.h5"

mol = gto.Mole()
mol.verbose = 5
mol.atom = """
               O    0.00000000   0.00000000  -0.60000000
               O    0.00000000   0.00000000   0.60000000
               """
mol.basis = "ccecp-ccpvqz"
mol.unit = "A"
mol.ecp = "ccecp"
mol.charge = 0
mol.spin = 2
mol.symmetry = False
mol.cart = False
mol.output = "O2.out"
mol.build()

mf = scf.UHF(mol)
mf.max_cycle = 200
mf_scf = mf.kernel()

trexio.to_trexio(mf, filename)
