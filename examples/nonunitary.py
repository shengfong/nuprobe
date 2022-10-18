#! /usr/bin/env python3

""" 
Nonunitariy in a three-flavor system. 
This code will produce a plot of oscillation probability as a function of neutrino energy.
""" 


import matplotlib.pyplot as plt
import numpy as np
import time

from nuprobe.inputs import NuSystem, create_U_PMNS, create_alpha
from nuprobe.probability import nuprobe

t0 = time.time()

# Specify the dimension of the system
nu_sys = NuSystem(3)

# Set the standard three-flavor parameters to normal (True) or inverse (False) mass ordering
nu_sys.set_standard_normal()

# Set the nonunitary parameters with a 3 x 3 Hermitian matrix of U U^\dagger
# If set_nonunitary(i, j, value), it automatically fills in the complex conjugation entry (j, i)
# The default is an identity matrix
nu_sys.set_nonunitary(1, 1, 0.98)
nu_sys.set_nonunitary(2, 2, 0.98)
nu_sys.set_nonunitary(1, 2, 0.02*np.exp(np.pi*1j/3))
alpha1 = create_alpha(nu_sys.nonunitary.copy())
nu_sys.set_nonunitary(1, 2, 0.02*np.exp(-np.pi*1j/3))
alpha2 = create_alpha(nu_sys.nonunitary.copy())

# Construct the full rotation matrix
UPMNS = create_U_PMNS(nu_sys.theta, nu_sys.delta) 
U1 = alpha1 @ UPMNS
U2 = alpha2 @ UPMNS 

#Calculating oscillation probability as a function of neutrino energy
PE = []
PE1 = []
PE2 = []
num = 1000
EE = np.logspace(0, 1.4771, num)
L = 6371*2

for i in range(num):
    PE.append(nuprobe(2, 1, L, EE[i], nu_sys.mass, UPMNS, antinu=False, const_matter=False, V_NSI=None))
    PE1.append(nuprobe(2, 1, L, EE[i], nu_sys.mass, U1, antinu=False, const_matter=False, V_NSI=None))
    PE2.append(nuprobe(2, 1, L, EE[i], nu_sys.mass, U2, antinu=False, const_matter=False, V_NSI=None))

# Plotting the results
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath,slashed}"
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams["figure.figsize"] = (8, 7)
plt.rc('font', **{'family' : 'serif', 'size' : 17})

x = EE
plt.xscale("log")
plt.xlabel(r'$E_\nu$  [GeV]')
plt.ylabel(r'$P_{\mu e}$')
plt.title(r'Earth-crossing neutrinos, NO, $(UU^\dagger)_{ee} = (UU^\dagger)_{\mu\mu}  = 0.98$')
plt.xlim([min(x),max(x)])
plt.ylim([0,0.6])
plt.plot(x, PE, 'k-', label='Standard')
plt.plot(x, PE1, '--', color = 'cornflowerblue', label=r'$(UU^\dagger)_{e\mu} = 0.02e^{i\pi/3}$')
plt.plot(x, PE2, 'm:', linewidth='2', label=r'$(UU^\dagger)_{e\mu} = 0.02e^{-i\pi/3}$')
plt.legend()

#plt.savefig('Figure.png', bbox_inches="tight")

dt = time.time() - t0
print('runtime = ', round(dt/60,5), ' min')

plt.show()
