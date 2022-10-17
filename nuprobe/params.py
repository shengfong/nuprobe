import numpy as np

# Relative difference for two eigenvalues to be considered equal 
rtol = 1e-6

# Absolute value [eV^2/GeV] where an eigenvalue is considered zero 
# Any eigenvalues below atol will be set to atol
atol = 1e-20


####### Physical constants from CODATA 2018 #######

# Reduced planck constant [GeV s]
hbar = 6.582119569e-25

# Speed of light [m/s]
c = 299792458

# electric charge [C]
e = 1.602176634e-19

# Avogadro constant [1/mol]
NA = 6.02214076e23

# Fermi constant [1/GeV^2]
GF = 1.1663788e-5

# Conversion from [g] to [eV]
CONV_g_eV = 1/(1e3*e/c**2)

# Conversion from [cm] to [1/eV]
CONV_cm_ieV = 1e-2*1e-9/(hbar*c)

# Conversion from [km] to [GeV/eV^2] 
CONV_L = 1e5*CONV_cm_ieV*1e-9

# Conversion factor for the matter potential [eV^2/GeV] when the input is matter density [g/cm^3]
CONV_matter = np.sqrt(2)*GF*1e-18*NA/(CONV_cm_ieV)**3*1e9


####### Global fit from NuFIT 5.1 (www.nu-fit.org) #######
# Angles and phases are in [radian]
# Squared masses are in [eV^2] 
# Masses are in [eV]

# Normal mass ordering 
theta12n = 0.5836
theta23n = 0.8587
theta13n = 0.1496
delta13n = 3.3859
m21sqn = 7.42e-5 
m31sqn = 2.517e-3 
m1n = 0
m2n = np.sqrt(m21sqn)
m3n = np.sqrt(m31sqn)

# Inverse mass ordering 
theta12i = 0.5838
theta23i = 0.8639
theta13i = 0.1501
delta13i = 5.0091
m21sqi = 7.42e-5 
m32sqi = -2.498e-3
m1i = np.sqrt(-m32sqi - m21sqi)
m2i = np.sqrt(m21sqi + m1i**2)
m3i = 0






