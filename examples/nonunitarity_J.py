#! /usr/bin/env python3

""" 
Nonunitary scenario in a three-flavor system. 
This code will produce a plot of Jarlskog combinations as a function of neutrino energy.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import time

from nuprobe.inputs import NuSystem, create_U_PMNS, create_alpha
from nuprobe.probability import hamiltonian, V_matter, Jt


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
alpha = create_alpha(nu_sys.nonunitary.copy())

# Construct the full rotation matrix
UPMNS = create_U_PMNS(nu_sys.theta, nu_sys.delta) 
U = alpha @ UPMNS

#Calculating Jarlskog combination
J1 = []
J2 = []
J3 = []
J10 = []
J20 = []
J30 = []
num = 100
EE = np.logspace(0, 2, num)
V = V_matter(3, 3)

for i in range(num):
    H1 = hamiltonian(EE[i], nu_sys.mass, U, V, False)
    Jt1 = (Jt(1, 2, 1, 2, H1, U) + Jt(1, 2, 1, 3, H1, U))
    Jt2 = -(Jt(1, 2, 2, 1, H1, U) + Jt(1, 2, 2, 3, H1, U))
    Jt3 = (Jt(1, 2, 3, 1, H1, U) + Jt(1, 2, 3, 2, H1, U))
    J1.append(Jt1)
    J2.append(Jt2)
    J3.append(Jt3)

    H0 = hamiltonian(EE[i], nu_sys.mass, U, np.zeros((3, 3)), False)
    Jt10 = (Jt(1, 2, 1, 2, H0, U) + Jt(1, 2, 1, 3, H0, U))
    Jt20 = -(Jt(1, 2, 2, 1, H0, U) + Jt(1, 2, 2, 3, H0, U))
    Jt30 = (Jt(1, 2, 3, 1, H0, U) + Jt(1, 2, 3, 2, H0, U))
    J10.append(Jt10)
    J20.append(Jt20)
    J30.append(Jt30)

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath,slashed}"
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams["figure.figsize"] = (8, 7)
plt.rc('font', **{'family' : 'serif', 'size' : 17})

x = EE
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r'$E_\nu$  [GeV]')
plt.ylabel(r'$\displaystyle{\sum_{k \neq j} \widetilde J_{e \mu}^{j k}}$')
plt.title(r'NO, $(UU^\dagger)_{ee} = (UU^\dagger)_{\mu\mu} = 0.98,\; (UU^\dagger)_{e\mu} = 0.02e^{i\pi/3}$')
plt.xlim([min(x),max(x)])
plt.ylim([1e-5,1e-2])
plt.plot(x, J1, 'r-')
plt.plot(x[::5], J1[::5], linestyle = 'None', marker='o', markersize=5, markerfacecolor='r', markeredgecolor='r')
plt.plot(x, J2, 'g-')
plt.plot(x[::5], J2[::5], linestyle = 'None', marker='x', markersize=5, markerfacecolor='g', markeredgecolor='g')
plt.plot(x, J3, 'b-')
plt.plot(x[::5], J3[::5], linestyle = 'None', marker='^', markersize=5, markerfacecolor='b', markeredgecolor='b')
plt.plot(x, J10, 'r--')
plt.plot(x[::5], J10[::5], linestyle = 'None', marker='o', markersize=5, markerfacecolor='r', markeredgecolor='r')
plt.plot(x, J20, 'g--')
plt.plot(x[::5], J20[::5], linestyle = 'None', marker='x', markersize=5, markerfacecolor='g', markeredgecolor='g')
plt.plot(x, J30, 'b--')
plt.plot(x[::5], J30[::5], linestyle = 'None', marker='^', markersize=5, markerfacecolor='b', markeredgecolor='b')

legend_elements = [
    Line2D([0], [0], linestyle='-', color='r', marker='o', markersize=5, label=r'$j = 1, \rho = 3$ g/cm$^3$'),
    Line2D([0], [0], linestyle='-', color='g', marker='x', markersize=5, label=r'$j = 2, \rho = 3$ g/cm$^3$'),
    Line2D([0], [0], linestyle='-', color='b', marker='^', markersize=5, label=r'$j = 3, \rho = 3$ g/cm$^3$'),
    Line2D([0], [0], linestyle='--', color='r', marker='o', markersize=5, label=r'$j = 1, \rho = 0$'),
    Line2D([0], [0], linestyle='--', color='g', marker='x', markersize=5, label=r'$j = 2, \rho = 0$'),
    Line2D([0], [0], linestyle='--', color='b', marker='^', markersize=5, label=r'$j = 3, \rho = 0$')
    ]

plt.legend(handles=legend_elements)

# plt.savefig('Figure.png', bbox_inches="tight")

dt = time.time() - t0
print('runtime = ', round(dt/60,5), ' min')

plt.show()

