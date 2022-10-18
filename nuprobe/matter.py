import numpy as np

# Constant matter density [g/cm^3]
rho_const = 3

# Average number of electron per nucleon [dimensionless]
Y_e = 0.5

# Average number of neutron per electron [dimensionless]
Y_n = 1.06

####### Matter density profile #######
# Earth radius [km]
R0 = 6371 

# Layers in [km]
LL = np.array([0, 0.1, 0.45, 0.81, 1.19, 1.55, 1.9, 2])*R0 

# Matter densities in [g/cm^3]
rho_LL = [3.6, 5, 10, 13, 10, 5, 3.6] 



