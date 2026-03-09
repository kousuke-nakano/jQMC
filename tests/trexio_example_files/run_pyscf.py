from pyscf import gto, scf
from pyscf.tools import trexio

#'''
# H_ae_ccpvdz_cart
filename = "H_ae_ccpvdz_cart.h5"
atoms = """
H    0.00000000   0.00000000  0.00000000
"""
basis = "ccpvdz"
ecp = None
cart = True
charge = 0
spin = 1
#'''

'''
# H2_ecp_ccpvdz_cart
filename = "H2_ecp_ccpvdz_cart.h5"
atoms = """
H    0.00000000   0.00000000  -0.35000000
H    0.00000000   0.00000000  +0.35000000
"""
basis = "ccecp-ccpvdz"
ecp = "ccecp"
cart = True
charge = 0
spin = 0
'''

'''
# H_ecp_ccpvdz_cart
filename = "H_ecp_ccpvdz_cart.h5"
atoms = """
H    0.00000000   0.00000000  0.00000000
"""
basis = "ccecp-ccpvdz"
ecp = "ccecp"
cart = True
charge = 0
spin = 1
'''

'''
# H2_ae_ccpvdz_cart
filename = "H2_ae_ccpvdz_cart.h5"
atoms = """
H    0.00000000   0.00000000  -0.35000000
H    0.00000000   0.00000000  +0.35000000
"""
basis = "ccpvdz"
ecp = None
cart = True
charge = 0
spin = 0
'''

'''
# Li_ae_ccpvdz_cart
filename = "Li_ae_ccpvdz_cart.h5"
atoms = """
Li    0.00000000   0.00000000  0.0000000
"""
basis = "ccpvdz"
ecp = None
cart = True
charge = 0
spin = 1
'''

# N_ae_ccpvdz_cart
filename = "N_ae_ccpvdz_cart.h5"
atoms = """
N    0.00000000   0.00000000  0.0000000
"""
basis = "ccpvdz"
ecp = None
cart = True
charge = 0
spin = 3

# pyscf script below
mol = gto.Mole()
mol.verbose = 5
mol.atom = atoms
mol.basis = basis
mol.unit = "A"
mol.ecp = ecp
mol.charge = charge
mol.spin = spin
mol.symmetry = False
mol.cart = cart
mol.output = "Li.out"
mol.build()

mf = scf.HF(mol)
mf.max_cycle = 200
mf_scf = mf.kernel()

trexio.to_trexio(mf, filename)
