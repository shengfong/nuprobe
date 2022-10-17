#! /usr/bin/env python3

""" 
NSI scenario in a three-flavor system. 
This code will produce a plot of oscillation probability as a function of neutrino energy.
"""

import matplotlib.pyplot as plt
import numpy as np
import time

from nuprobe.inputs import NuSystem, create_U_PMNS
from nuprobe.probability import nuprobe

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

# Construct the full rotation matrix
U = create_U_PMNS(nu_sys.theta, nu_sys.delta) 


PE = []
PEa = []
PEb = []
num = 1000
EE = np.logspace(0, 1.4771, num)
L = 6371*2

for i in range(num):
    PE.append(nuprobe(2, 1, L, EE[i], nu_sys.mass, U, antinu=False, V_NSI=None))
    PEa.append(nuprobe(2, 1, L, EE[i], nu_sys.mass, U, antinu=False, V_NSI=NSIa))
    PEb.append(nuprobe(2, 1, L, EE[i], nu_sys.mass, U, antinu=False, V_NSI=NSIb))


plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath,slashed}"
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams["figure.figsize"] = (8, 7)
plt.rc('font', **{'family' : 'serif', 'size' : 17})

x = EE
plt.xscale("log")
plt.xlabel(r'$E_\nu$  [GeV]')
plt.ylabel(r'$P_{\mu e}$')
plt.title(r'Earth-crossing neutrinos, NO, $-\epsilon_{ee} = \epsilon_{\mu\mu} = 0.02 $')
plt.xlim([min(x),max(x)])
plt.ylim([0,0.6])
plt.plot(x, PE, 'k-', label='Standard')
plt.plot(x, PEa, '--', color='cornflowerblue', label=r'$\epsilon_{e\mu} = -0.02e^{-i\pi/3}$')
plt.plot(x, PEb, 'm:', linewidth='2', label=r'$\epsilon_{e\mu} = -0.02e^{i\pi/3}$')
plt.legend()

#plt.savefig('Figure.png', bbox_inches="tight")

dt = time.time() - t0
print('runtime = ', round(dt/60,5), ' min')

plt.show()
