import numpy as np
import sys
from nuprobe.params import theta12n, theta23n, theta13n, delta13n, m1n, m2n, m3n
from nuprobe.params import theta12i, theta23i, theta13i, delta13i, m1i, m2i, m3i

class NuSystem:
    def __init__(self, d, index_start_with_one=True):
        """ Initializes NuSystem
        params:
        - d (int): number of neutrino flavors
        - index_start_with_one (bool): whether index starts with one (default is true)

        Sets parameters for the system based on the number of neutrino flavors
        - theta [radian]: d x d matrix where the elements in the upper triangle are the angles 
        - delta [radian]: d x d matrix where the elements in the upper triangle are the phases 
        - mass [eV]: d vector of masses of neutrinos
        - NSI [dimensionless]: d x d Hermitian matrix defined with respect to the matter potential of charge current (arXiv:2106.07755)
          If set_NSI(i, j, value), it automatically fills in the complex conjugation entry (j, i)
        - nonunitary [dimensionless]: 3 x 3 Hermitian matrix of U U^\dagger
          If set_nonunitary(i, j, value), it automatically fills in the complex conjugation entry (j, i)

        """

        if d <= 1:
            print("It takes two to oscillate.")
            sys.exit()

        self.d = d
        self.index_one = index_start_with_one
        self.theta = np.zeros((self.d, self.d), dtype = np.float64)
        self.delta = np.zeros((self.d, self.d), dtype = np.float64)
        self.mass = np.zeros(self.d, dtype = np.float64)
        self.NSI = np.zeros((self.d, self.d), dtype = complex)
        self.nonunitary = np.identity(3, dtype = complex)
        

    def set_theta(self, i, j, value):
        if self.index_one:
            i = i - 1
            j = j - 1
     
        self.theta[i][j] = value


    def set_delta(self, i, j, value):
        if self.index_one:
            i = i - 1
            j = j - 1
     
        self.delta[i][j] = value


    def set_mass(self, i, value):
        if self.index_one:
            i = i - 1
     
        self.mass[i] = value


    def set_NSI(self, i, j, value):
        if self.index_one:
            i = i - 1
            j = j - 1
     
        self.NSI[i][j] = value
        self.NSI[j][i] = np.conjugate(value)
    

    def set_nonunitary(self, i, j, value):
        if self.index_one:
            i = i - 1
            j = j - 1
        
        self.nonunitary[i][j] = value
        self.nonunitary[j][i] = np.conjugate(value)
    
    def set_standard_normal(self, normal=True):
        if normal:
            # Specify the Euler angles and phases [radian]
            self.set_theta(1, 2, theta12n)
            self.set_theta(2, 3, theta23n)
            self.set_theta(1, 3, theta13n)
            self.set_delta(1, 3, delta13n)

            # Specify the masses of neutrinos [eV]
            self.set_mass(1, m1n)
            self.set_mass(2, m2n)
            self.set_mass(3, m3n)
        else:
            # Specify the Euler angles and phases [radian]
            self.set_theta(1, 2, theta12i)
            self.set_theta(2, 3, theta23i)
            self.set_theta(1, 3, theta13i)
            self.set_delta(1, 3, delta13i)

            # Specify the masses of neutrinos [eV]
            self.set_mass(1, m1i)
            self.set_mass(2, m2i)
            self.set_mass(3, m3i)


def R(d, ii, jj, theta_value, delta_value):
    """ Construct rotation matrix in d dimension 
    params:
    - d (int): number of dimension
    - ii, jj (int): rotation around axis perpendicular to the ii-jj plane
    - theta_value [radian]: rotation angle
    - delta_value [radian]: phase
    """ 

    RR = np.identity(d, dtype = complex)
    RR[ii-1, ii-1] = np.cos(theta_value)
    RR[jj-1, jj-1] = np.cos(theta_value)
    RR[ii-1, jj-1] = np.exp(-1j * delta_value) * np.sin(theta_value)
    RR[jj-1, ii-1] = -np.exp(1j * delta_value) * np.sin(theta_value)
    return RR

def create_U_PMNS(theta, delta):
    """ Construct 3 x 3 PMNS matrix as part of d x d matrix 
    params:
    - theta [radian]: d x d upper triangle matrix where [ii, jj] element corresponds to rotation angle in the ii-jj plane
    - delta [radian]: d x d upper triangle matrix where [ii, jj] element corresponds to phase in the ii-jj plane
    """ 

    d = len(theta)
    UU = R(d, 2, 3, theta[1][2], delta[1][2]) @ R(d, 1, 3, theta[0][2], delta[0][2]) @ R(d, 1, 2, theta[0][1], delta[0][1])
    return UU

def create_U_NEW(theta, delta):
    """ Construct d x d matrix involving new angles and phases when the dimension d > 3
    params:
    - theta [radian]: d x d upper triangle matrix where [ii, jj] element corresponds to rotation angle in the ii-jj plane
    - delta [radian]: d x d upper triangle matrix where [ii, jj] element corresponds to phase in the ii-jj plane
    """ 

    d = len(theta)
    UU = np.identity(d)
    if d > 3:
        #m = 4
        for j in range(4, d+1):
            for i in range(1, j):
                UU = R(d, i, j, theta[i-1][j-1], delta[i-1][j-1]) @ UU
    return UU

def create_alpha(nonunitary):
    """ Construct 3 x 3 matrix of alpha parametrization (arXiv:1503.08879)
    param:
    - nonunitary [dimensionless]: 3 x 3 Hermitian matrix U U^\dagger
    """ 

    d = len(nonunitary)
    U = nonunitary
    alpha = np.identity(3, dtype = complex)
    alpha[0, 0] = np.sqrt(U[0, 0])
    alpha[1, 0] = np.conjugate(U[0, 1])/np.sqrt(U[0, 0])
    alpha[1, 1] = np.sqrt(U[1, 1] - np.abs(U[0, 1])**2/U[0, 0])
    alpha[2, 0] = np.conjugate(U[0, 2])/np.sqrt(U[0, 0])
    alpha[2, 1] = (np.conjugate(U[1, 2]) - U[0, 1]*np.conjugate(U[0,2])/U[0, 0])/alpha[1, 1]
    alpha[2, 2] = np.sqrt(U[2, 2] - np.abs(U[0, 2])**2/U[0, 0] - np.abs(alpha[2, 1])**2)
    return alpha    
