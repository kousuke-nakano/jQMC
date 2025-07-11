from pyscf import gto, scf
from pyscf.tools import trexio

filename = "Ar_ccecp_ccpvqz_cart.h5"

mol = gto.Mole()
mol.verbose = 5
mol.atom = """
               Ar    0.00000000   0.00000000   0.00000000
               """
mol.basis = "ccecp-ccpv5z"
mol.unit = "A"
mol.ecp = "ccecp"
mol.charge = 0
mol.spin = 0
mol.symmetry = False
mol.cart = True
mol.output = "Ar.out"
mol.build()

mf = scf.ROHF(mol)
mf.max_cycle = 200
mf_scf = mf.kernel()

trexio.to_trexio(mf, filename)
