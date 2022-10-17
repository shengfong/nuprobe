#! /usr/bin/env python3

""" 
NSI scenario in a three-flavor system. 
This code will produce a plot of Naumov-Harrison-Scott combinations as a function of neutrino energy.
"""

import numpy as np

from nuprobe.inputs import NuSystem, create_U_PMNS, create_U_NEW
from nuprobe.probability import nuprobe


# Specify the dimension of the system
nu_sys = NuSystem(4)

# Set the standard three-flavor parameters to normal (True) or inverse (False) mass ordering
nu_sys.set_standard_normal(False)

# Specify mixing angles, phases and masses of new flavor states
nu_sys.set_mass(4, 2)
nu_sys.set_theta(1, 4, 0.2)
nu_sys.set_theta(2, 4, 0.1)
nu_sys.set_theta(3, 4, 0.3)
nu_sys.set_delta(1, 4, 1.8)


# Construct the full rotation matrix
U0 = create_U_PMNS(nu_sys.theta, nu_sys.delta) 
UNP = create_U_NEW(nu_sys.theta, nu_sys.delta) 
U = UNP @ U0

L = 1000
E = 2
P = nuprobe(2, 1, L, E, nu_sys.mass, U, antinu=True, V_NSI=None)
print('P =', P)

