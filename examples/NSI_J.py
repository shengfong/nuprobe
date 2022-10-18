#! /usr/bin/env python3

""" 
NSI scenario in a three-flavor system. 
This code will produce a plot of Naumov-Harrison-Scott combinations as a function of neutrino energy.
"""

import matplotlib.pyplot as plt
import numpy as np
import time

from nuprobe.inputs import NuSystem, create_U_PMNS
from nuprobe.probability import hamiltonian, V_matter, Jt, NHS

t0 = time.time()

# Specify the dimension of the system
nu_sys = NuSystem(3)

# Set the standard three-flavor parameters to normal (True) or inverse (False) mass ordering
nu_sys.set_standard_normal()

# Set the NSI parameters with a d x d Hermitian matrix defined with respect to the matter potential of charge current (arXiv:2106.07755)
# If set_NSI(i, j, value), it automatically fills in the complex conjugation entry (j, i)
# The default values are zeros
nu_sys.set_NSI(1, 1, -0.02)
nu_sys.set_NSI(2, 2, 0.02)
nu_sys.set_NSI(1, 2, -0.02*np.exp(-np.pi*1j/3))
NSIa = nu_sys.NSI.copy()
nu_sys.set_NSI(1, 2, -0.02*np.exp(np.pi*1j/3))
NSIb = nu_sys.NSI.copy()
nu_sys.set_NSI(1, 2, 0)
NSIc = nu_sys.NSI.copy()

# Calculate the matter potential matrix
# The arguments of V_matter(dimension, matter density[g/cm^3], NSI potential parameters)
Va = V_matter(3, 3, V_NSI = NSIa)
Vb = V_matter(3, 3, V_NSI = NSIb)
Vc = V_matter(3, 3, V_NSI = NSIc)

# Construct the full rotation matrix
U = create_U_PMNS(nu_sys.theta, nu_sys.delta) 

NSI1 = []
NSI2 = []
NSI3 = []
NSI10 = []
num = 1000
EE = np.logspace(0, 2, num)

for i in range(num):
    H1 = hamiltonian(EE[i], nu_sys.mass, U, Va, False)
    NSI1.append(np.abs(NHS(2, 1, 2, 1, H1, U))**(1/3))
    H2 = hamiltonian(EE[i], nu_sys.mass, U, Vb, False)
    NSI2.append(np.abs(NHS(2, 1, 2, 1, H2, U))**(1/3))

    H3 = hamiltonian(EE[i], nu_sys.mass, U, np.zeros((3, 3)), False)
    #H3 = hamiltonian(EE[i], nu_sys.mass, U, Vc, False)
    NSI10.append(np.abs(NHS(2, 1, 2, 1, H3, U))**(1/3))

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath,slashed}"
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams["figure.figsize"] = (8, 7)
plt.rc('font', **{'family' : 'serif', 'size' : 17})

x = EE
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r'$E_\nu$  [GeV]')
plt.ylabel(r'$10^9 \left|\lambda_{21}\lambda_{31}\lambda_{32} \tilde J_{e\mu}^{jk}\right|^{1/3}$ [eV]')
plt.title(r'NO, $-\epsilon_{ee} = \epsilon_{\mu\mu} = 0.02$')
plt.xlim([min(x),max(x)])
plt.ylim([1e-6,1e-4])
plt.plot(x, NSI1, '--', color = 'cornflowerblue', label=r'$\epsilon_{e\mu} = -0.02e^{-i\pi/3}, \rho = 3$ g/cm$^3$')
plt.plot(x, NSI2, 'm:', linewidth='2', label=r'$\epsilon_{e\mu} = -0.02e^{i\pi/3}, \rho = 3$ g/cm$^3$')
plt.plot(x, NSI10, 'k-', label=r'$\epsilon_{e\mu} = 0$ or $\rho = 0$')
plt.legend()

dt = time.time() - t0
print('runtime = ', round(dt/60,5), ' min')

plt.show()

