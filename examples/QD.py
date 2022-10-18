#! /usr/bin/env python3

""" 
Implementation of quasi-Dirac neutrino scenario in a six-flavor system. 
This code will produce a plot of oscillation probability as a function of neutrino energy.
""" 

import matplotlib.pyplot as plt
import numpy as np
import time

from nuprobe.inputs import NuSystem, create_U_PMNS, create_U_NEW
from nuprobe.params import m1n, m2n, m3n
from nuprobe.params import m1i, m2i, m3i
from nuprobe.probability import nuprobe

# Note: It took 20 mins to run this example on my laptop

t0 = time.time()

# Specify the dimension of the system
nu_sys = NuSystem(6)

# Set the standard three-flavor parameters to normal (True) or inverse (False) mass ordering
nu_sys.set_standard_normal()

# Specify the quasi-Dirac masses of neutrinos [eV]
nu_sys.set_mass(4, m1n)
nu_sys.set_mass(5, m2n)
nu_sys.set_mass(6, m3n)
mass0 = nu_sys.mass.copy()

delta = 1.e-3/2.0136
ep3 = 2*m3n*delta
nu_sys.set_mass(3, m3n - ep3)
nu_sys.set_mass(6, m3n + ep3)
massQD = nu_sys.mass.copy()
print('ep3 [eV^2]=', 4*m3n*ep3)

delta = 1.e-2/2.0136
ep3 = 2*m3n*delta
nu_sys.set_mass(3, m3n - ep3)
nu_sys.set_mass(6, m3n + ep3)
massQD2 = nu_sys.mass.copy()
print('ep3 [eV^2]=', 4*m3n*ep3)

# Specify mixing angles and phases of new flavor states
nu_sys.set_theta(3, 4, 0.03)
nu_sys.set_theta(2, 5, 0.03)
nu_sys.set_theta(1, 6, 0.03)

# Construct the full rotation matrix
UPMNS = create_U_PMNS(nu_sys.theta, nu_sys.delta) 
I3 = np.identity(3)
Y = np.block([[I3, 1j*I3],[I3, -1j*I3]])/np.sqrt(2)
U = create_U_NEW(nu_sys.theta, nu_sys.delta) @ UPMNS @ Y

PE = []
PEa = []
PEb = []
num = 1000
EE = np.logspace(0, 1.4771, num)
L = 6371*2


for i in range(num):
    PE.append(nuprobe(2, 1, L, EE[i], mass0, UPMNS, antinu=False, const_matter=False, V_NSI=None))
    PEa.append(nuprobe(2, 1, L, EE[i], massQD, U, antinu=False, const_matter=False, V_NSI=None))
    PEb.append(nuprobe(2, 1, L, EE[i], massQD2, U, antinu=False, const_matter=False, V_NSI=None))


plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath,slashed}"
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams["figure.figsize"] = (8, 7)
plt.rc('font', **{'family' : 'serif', 'size' : 17})

x = EE
plt.xscale("log")
plt.xlabel(r'$E_\nu$  [GeV]')
plt.ylabel(r'$P_{\mu e}$')
plt.title(r'Earth-crossing neutrinos, NO, $\theta_{34} = \theta_{25} = \theta_{16} = 0.03$')
plt.xlim([min(x),max(x)])
plt.ylim([0,0.6])
plt.plot(x, PE, 'k-', label='Standard')
plt.plot(x, PEa, 'm:', linewidth='2', label=r'$\Delta m_{63}^2 = 10^{-5}$ eV$^2$')
plt.plot(x, PEb, '--', color='cornflowerblue', label=r'$\Delta m_{63}^2 = 10^{-4}$ eV$^2$')
plt.legend()

#plt.savefig('Figure.png', bbox_inches="tight")

dt = time.time() - t0
print('runtime = ', round(dt/60,5), ' min')

plt.show()

