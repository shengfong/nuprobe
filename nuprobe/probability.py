import numpy as np
from numpy import linalg as LA
import sys

from nuprobe.params import CONV_L, CONV_matter, rtol
from nuprobe.degeneracy import calc_unique_values
from nuprobe.matter import const_matter, rho_const, rho_LL, LL

def eigenvalues(H):
    """ Calculate distinct eigenvalues from the Hamiltonian H 
    param:
    - H [eV^2/GeV]: d x d matrix
    """ 

    d = len(H) # Dimension of the system
    ll = [] # Array to store the unique eigenvalues

    if d == 1:
        ll = H

    elif d == 2:
        T = np.trace(H)
        D = LA.det(H)
        if np.abs(np.sqrt(T**2-D)/T) < rtol:
            ll = T
        else:
            l1 = (T-np.sqrt(T**2-D))/2
            l2 = (T+np.sqrt(T**2-D))/2
            ll.append(l1)
            ll.append(l2)

    elif d == 3:
        T = np.trace(H)
        D = LA.det(H)
        T2 = np.trace(H @ H)
        A = (T**2  - T2)/2
        F = np.sqrt(T**2-3*A)
        G = np.arccos((2*T**3-9*A*T+27*D)/(2*F**3))/3

        if np.abs(F/T) < rtol:
            ll = T
        elif np.abs((F/T)*np.sin(G)) < rtol:
            l1 = (T-F)/3 
            l2 = (T+2*F)/3
            ll.append(l1)
            ll.append(l2)
        else:
            l1 = (T-F*np.cos(G))/3 - F*np.sin(G)/np.sqrt(3)
            l2 = (T-F*np.cos(G))/3 + F*np.sin(G)/np.sqrt(3)
            l3 = (T+2*F*np.cos(G))/3
            ll.append(l1)
            ll.append(l2)
            ll.append(l3)

    elif d == 4:
        T = np.trace(H)
        D = LA.det(H)
        
        T2 = np.trace(H @ H)
        A = (T**2 - T2)/2

        T3 = np.trace(H @ H @ H)
        A2 = (T**3 - 3*T*T2 + 2*T3)/6

        P = (3/8)*T**2 - A
        Q = -T**3/8 + T*A/2 - A2
        D1 = 2*A**3 - 9*T*A*A2 + 27*T**2*D + 27*A2**2 - 72*A*D
        F = np.sqrt(A**2 - 3*T*A2 + 12*D)
        G = np.arccos(D1/(2*F**3))/3
        S = np.sqrt(2*P/3 + 2*F*np.cos(G)/3)/2
        l1 = T/4 - S + np.sqrt(2*P -4*S**2 + Q/S)/2
        l2 = T/4 - S - np.sqrt(2*P -4*S**2 + Q/S)/2
        l3 = T/4 + S + np.sqrt(2*P -4*S**2 - Q/S)/2
        l4 = T/4 + S - np.sqrt(2*P -4*S**2 - Q/S)/2
        ll.append(l1)
        ll.append(l2)
        ll.append(l3)
        ll.append(l4)

        # Consider only unique eigenvalues
        ll = calc_unique_values(ll)

    # For d > 4, calculate the eigenvalues numerically
    else:
        ll = LA.eigvals(H) # Calculating the eigenvalues numerically

        # Consider only unique eigenvalues
        ll = calc_unique_values(ll)

    return np.real(ll)


def mixing(H):
    """ Calculate mixing elements analytically from the Hamiltonian H 
    This code works for Hamiltonian in any basis 
    but we have chosen to work in the vacuum mass basis in calculating the S matrix

    param:
    - H [eV^2/GeV]: d x d matrix
    """ 

    d = len(H)
    ll = eigenvalues(H) # Determine distinct eigenvalues
    d_eigen = len(ll)

    # Calculate diagolization matrices based on the number of distinct eigenvalues d_eigen
    Iden = np.identity(d, dtype = complex)
    HH=[]
    XX=[]

    Hi = H
    for i in range(d_eigen - 1):
        HH.append(Hi)
        Hi = Hi.dot(H)

    if d_eigen == 1:
        XX = Iden
        ll = [0]

    elif d_eigen == 2:
        l1 = ll[0]
        l2 = ll[1]

        XX.append((Iden*l2 - HH[0])/(l2-l1))
        XX.append((Iden*l1 - HH[0])/(l1-l2))

    elif d_eigen == 3:
        l1 = ll[0]
        l2 = ll[1]
        l3 = ll[2]

        XX.append((Iden*l2*l3 - HH[0]*(l2+l3) + HH[1])/(l2-l1)/(l3-l1))
        XX.append((Iden*l1*l3 - HH[0]*(l1+l3) + HH[1])/(l1-l2)/(l3-l2))
        XX.append((Iden*l1*l2 - HH[0]*(l1+l2) + HH[1])/(l1-l3)/(l2-l3))

    elif d_eigen == 4:
        l1 = ll[0]
        l2 = ll[1]
        l3 = ll[2]
        l4 = ll[3]

        XX.append((Iden*l2*l3*l4 - HH[0]*(l2*l3+l2*l4+l3*l4) + HH[1]*(l2+l3+l4) - HH[2])/(l2-l1)/(l3-l1)/(l4-l1))
        XX.append((Iden*l1*l3*l4 - HH[0]*(l1*l3+l1*l4+l3*l4) + HH[1]*(l1+l3+l4) - HH[2])/(l1-l2)/(l3-l2)/(l4-l2))
        XX.append((Iden*l1*l2*l4 - HH[0]*(l1*l2+l1*l4+l2*l4) + HH[1]*(l1+l2+l4) - HH[2])/(l1-l3)/(l2-l3)/(l4-l3))
        XX.append((Iden*l1*l2*l3 - HH[0]*(l1*l2+l1*l3+l2*l3) + HH[1]*(l1+l2+l3) - HH[2])/(l1-l4)/(l2-l4)/(l3-l4))

    elif d_eigen == 5:
        l1 = ll[0]
        l2 = ll[1]
        l3 = ll[2]
        l4 = ll[3]
        l5 = ll[4]

        XX.append((Iden*l2*l3*l4*l5 - HH[0]*(l2*l3*l4+l2*l3*l5+l2*l4*l5+l3*l4*l5) 
        + HH[1]*(l2*l3+l2*l4+l2*l5+l3*l4+l3*l5+l4*l5) - HH[2]*(l2+l3+l4+l5) + HH[3])
        /(l2-l1)/(l3-l1)/(l4-l1)/(l5-l1))
        XX.append((Iden*l1*l3*l4*l5 - HH[0]*(l1*l3*l4+l1*l3*l5+l1*l4*l5+l3*l4*l5) 
        + HH[1]*(l1*l3+l1*l4+l1*l5+l3*l4+l3*l5+l4*l5) - HH[2]*(l1+l3+l4+l5) + HH[3])
        /(l1-l2)/(l3-l2)/(l4-l2)/(l5-l2))
        XX.append((Iden*l1*l2*l4*l5 - HH[0]*(l1*l2*l4+l1*l2*l5+l1*l4*l5+l2*l4*l5) 
        + HH[1]*(l1*l2+l1*l4+l1*l5+l2*l4+l2*l5+l4*l5) - HH[2]*(l1+l2+l4+l5) + HH[3])
        /(l1-l3)/(l2-l3)/(l4-l3)/(l5-l3))
        XX.append((Iden*l1*l2*l3*l5 - HH[0]*(l1*l2*l3+l1*l2*l5+l1*l3*l5+l2*l3*l5) 
        + HH[1]*(l1*l2+l1*l3+l1*l5+l2*l3+l2*l5+l3*l5) - HH[2]*(l1+l2+l3+l5) + HH[3])
        /(l1-l4)/(l2-l4)/(l3-l4)/(l5-l4))
        XX.append((Iden*l1*l2*l3*l4 - HH[0]*(l1*l2*l3+l1*l2*l4+l1*l3*l4+l2*l3*l4) 
        + HH[1]*(l1*l2+l1*l3+l1*l4+l2*l3+l2*l4+l3*l4) - HH[2]*(l1+l2+l3+l4) + HH[3])
        /(l1-l5)/(l2-l5)/(l3-l5)/(l4-l5))

    elif d_eigen == 6:
        l1 = ll[0]
        l2 = ll[1]
        l3 = ll[2]
        l4 = ll[3]
        l5 = ll[4]
        l6 = ll[5]

        XX.append((Iden*l2*l3*l4*l5*l6 - HH[0]*(l2*l3*l4*l5+l2*l3*l4*l6+l2*l3*l5*l6+l2*l4*l5*l6+l3*l4*l5*l6) 
        + HH[1]*(l2*l3*l4+l2*l3*l5+l2*l3*l6+l2*l4*l5+l2*l4*l6+l2*l5*l6+l3*l4*l5+l3*l4*l6+l3*l5*l6+l4*l5*l6) 
        - HH[2]*(l2*l3+l2*l4+l2*l5+l2*l6+l3*l4+l3*l5+l3*l6+l4*l5+l4*l6+l5*l6) + HH[3]*(l2+l3+l4+l5+l6) - HH[4])
        /(l2-l1)/(l3-l1)/(l4-l1)/(l5-l1)/(l6-l1))
        XX.append((Iden*l1*l3*l4*l5*l6 - HH[0]*(l1*l3*l4*l5+l1*l3*l4*l6+l1*l3*l5*l6+l1*l4*l5*l6+l3*l4*l5*l6) 
        + HH[1]*(l1*l3*l4+l1*l3*l5+l1*l3*l6+l1*l4*l5+l1*l4*l6+l1*l5*l6+l3*l4*l5+l3*l4*l6+l3*l5*l6+l4*l5*l6) 
        - HH[2]*(l1*l3+l1*l4+l1*l5+l1*l6+l3*l4+l3*l5+l3*l6+l4*l5+l4*l6+l5*l6) + HH[3]*(l1+l3+l4+l5+l6) - HH[4])
        /(l1-l2)/(l3-l2)/(l4-l2)/(l5-l2)/(l6-l2))
        XX.append((Iden*l1*l2*l4*l5*l6 - HH[0]*(l1*l2*l4*l5+l1*l2*l4*l6+l1*l2*l5*l6+l1*l4*l5*l6+l2*l4*l5*l6) 
        + HH[1]*(l1*l2*l4+l1*l2*l5+l1*l2*l6+l1*l4*l5+l1*l4*l6+l1*l5*l6+l2*l4*l5+l2*l4*l6+l2*l5*l6+l4*l5*l6) 
        - HH[2]*(l1*l2+l1*l4+l1*l5+l1*l6+l2*l4+l2*l5+l2*l6+l4*l5+l4*l6+l5*l6) + HH[3]*(l1+l2+l4+l5+l6) - HH[4])
        /(l1-l3)/(l2-l3)/(l4-l3)/(l5-l3)/(l6-l3))
        XX.append((Iden*l1*l2*l3*l5*l6 - HH[0]*(l1*l2*l3*l5+l1*l2*l3*l6+l1*l2*l5*l6+l1*l3*l5*l6+l2*l3*l5*l6) 
        + HH[1]*(l1*l2*l3+l1*l2*l5+l1*l2*l6+l1*l3*l5+l1*l3*l6+l1*l5*l6+l2*l3*l5+l2*l3*l6+l2*l5*l6+l3*l5*l6) 
        - HH[2]*(l1*l2+l1*l3+l1*l5+l1*l6+l2*l3+l2*l5+l2*l6+l3*l5+l3*l6+l5*l6) + HH[3]*(l1+l2+l3+l5+l6) - HH[4])
        /(l1-l4)/(l2-l4)/(l3-l4)/(l5-l4)/(l6-l4))
        XX.append((Iden*l1*l2*l3*l4*l6 - HH[0]*(l1*l2*l3*l4+l1*l2*l3*l6+l1*l2*l4*l6+l1*l3*l4*l6+l2*l3*l4*l6) 
        + HH[1]*(l1*l2*l3+l1*l2*l4+l1*l2*l6+l1*l3*l4+l1*l3*l6+l1*l4*l6+l2*l3*l4+l2*l3*l6+l2*l4*l6+l3*l4*l6) 
        - HH[2]*(l1*l2+l1*l3+l1*l4+l1*l6+l2*l3+l2*l4+l2*l6+l3*l4+l3*l6+l4*l6) + HH[3]*(l1+l2+l3+l4+l6) - HH[4])
        /(l1-l5)/(l2-l5)/(l3-l5)/(l4-l5)/(l6-l5))
        XX.append((Iden*l1*l2*l3*l4*l5 - HH[0]*(l1*l2*l3*l4+l1*l2*l3*l5+l1*l2*l4*l5+l1*l3*l4*l5+l2*l3*l4*l5) 
        + HH[1]*(l1*l2*l3+l1*l2*l4+l1*l2*l5+l1*l3*l4+l1*l3*l5+l1*l4*l5+l2*l3*l4+l2*l3*l5+l2*l4*l5+l3*l4*l5) 
        - HH[2]*(l1*l2+l1*l3+l1*l4+l1*l5+l2*l3+l2*l4+l2*l5+l3*l4+l3*l5+l4*l5) + HH[3]*(l1+l2+l3+l4+l5) - HH[4])
        /(l1-l6)/(l2-l6)/(l3-l6)/(l4-l6)/(l5-l6))

    elif d_eigen == 7:
        l1 = ll[0]
        l2 = ll[1]
        l3 = ll[2]
        l4 = ll[3]
        l5 = ll[4]
        l6 = ll[5]
        l7 = ll[6]

        XX.append(
        (Iden*l2*l3*l4*l5*l6*l7 
        - HH[0]*(l2*l3*l4*l5*l6+l2*l3*l4*l5*l7+l2*l3*l4*l6*l7+l2*l3*l5*l6*l7+l2*l4*l5*l6*l7+l3*l4*l5*l6*l7) 
        + HH[1]*(l2*l3*l4*l5+l2*l3*l4*l6+l2*l3*l4*l7+l2*l3*l5*l6+l2*l3*l5*l7+l2*l3*l6*l7+l2*l4*l5*l6+l2*l4*l5*l7+l2*l4*l6*l7
                +l2*l5*l6*l7+l3*l4*l5*l6+l3*l4*l5*l7+l3*l4*l6*l7+l3*l5*l6*l7+l4*l5*l6*l7)
        - HH[2]*(l2*l3*l4+l2*l3*l5+l2*l3*l6+l2*l3*l7+l2*l4*l5+l2*l4*l6+l2*l4*l7+l2*l5*l6+l2*l5*l7+l2*l6*l7
                 +l3*l4*l5+l3*l4*l6+l3*l4*l7+l3*l5*l6+l3*l5*l7+l3*l6*l7+l4*l5*l6+l4*l5*l7+l4*l6*l7+l5*l6*l7)
        + HH[3]*(l2*l3+l2*l4+l2*l5+l2*l6+l2*l7+l3*l4+l3*l5+l3*l6+l3*l7+l4*l5+l4*l6+l4*l7+l5*l6+l5*l7+l6*l7) 
        - HH[4]*(l2+l3+l4+l5+l6+l7)
        + HH[5])
        /(l2-l1)/(l3-l1)/(l4-l1)/(l5-l1)/(l6-l1)/(l7-l1)
        )
        XX.append(
        (Iden*l1*l3*l4*l5*l6*l7 
        - HH[0]*(l1*l3*l4*l5*l6+l1*l3*l4*l5*l7+l1*l3*l4*l6*l7+l1*l3*l5*l6*l7+l1*l4*l5*l6*l7+l3*l4*l5*l6*l7) 
        + HH[1]*(l1*l3*l4*l5+l1*l3*l4*l6+l1*l3*l4*l7+l1*l3*l5*l6+l1*l3*l5*l7+l1*l3*l6*l7+l1*l4*l5*l6+l1*l4*l5*l7+l1*l4*l6*l7
                +l1*l5*l6*l7+l3*l4*l5*l6+l3*l4*l5*l7+l3*l4*l6*l7+l3*l5*l6*l7+l4*l5*l6*l7)
        - HH[2]*(l1*l3*l4+l1*l3*l5+l1*l3*l6+l1*l3*l7+l1*l4*l5+l1*l4*l6+l1*l4*l7+l1*l5*l6+l1*l5*l7+l1*l6*l7
                 +l3*l4*l5+l3*l4*l6+l3*l4*l7+l3*l5*l6+l3*l5*l7+l3*l6*l7+l4*l5*l6+l4*l5*l7+l4*l6*l7+l5*l6*l7)
        + HH[3]*(l1*l3+l1*l4+l1*l5+l1*l6+l1*l7+l3*l4+l3*l5+l3*l6+l3*l7+l4*l5+l4*l6+l4*l7+l5*l6+l5*l7+l6*l7) 
        - HH[4]*(l1+l3+l4+l5+l6+l7)
        + HH[5])
        /(l1-l2)/(l3-l2)/(l4-l2)/(l5-l2)/(l6-l2)/(l7-l2)
        )
        XX.append(
        (Iden*l1*l2*l4*l5*l6*l7 
        - HH[0]*(l1*l2*l4*l5*l6+l1*l2*l4*l5*l7+l1*l2*l4*l6*l7+l1*l2*l5*l6*l7+l1*l4*l5*l6*l7+l2*l4*l5*l6*l7) 
        + HH[1]*(l1*l2*l4*l5+l1*l2*l4*l6+l1*l2*l4*l7+l1*l2*l5*l6+l1*l2*l5*l7+l1*l2*l6*l7+l1*l4*l5*l6+l1*l4*l5*l7+l1*l4*l6*l7
                +l1*l5*l6*l7+l2*l4*l5*l6+l2*l4*l5*l7+l2*l4*l6*l7+l2*l5*l6*l7+l4*l5*l6*l7)
        - HH[2]*(l1*l2*l4+l1*l2*l5+l1*l2*l6+l1*l2*l7+l1*l4*l5+l1*l4*l6+l1*l4*l7+l1*l5*l6+l1*l5*l7+l1*l6*l7
                 +l2*l4*l5+l2*l4*l6+l2*l4*l7+l2*l5*l6+l2*l5*l7+l2*l6*l7+l4*l5*l6+l4*l5*l7+l4*l6*l7+l5*l6*l7)
        + HH[3]*(l1*l2+l1*l4+l1*l5+l1*l6+l1*l7+l2*l4+l2*l5+l2*l6+l2*l7+l4*l5+l4*l6+l4*l7+l5*l6+l5*l7+l6*l7) 
        - HH[4]*(l1+l2+l4+l5+l6+l7)
        + HH[5])
        /(l1-l3)/(l2-l3)/(l4-l3)/(l5-l3)/(l6-l3)/(l7-l3)
        )
        XX.append(
        (Iden*l1*l2*l3*l5*l6*l7 
        - HH[0]*(l1*l2*l3*l5*l6+l1*l2*l3*l5*l7+l1*l2*l3*l6*l7+l1*l2*l5*l6*l7+l1*l3*l5*l6*l7+l2*l3*l5*l6*l7) 
        + HH[1]*(l1*l2*l3*l5+l1*l2*l3*l6+l1*l2*l3*l7+l1*l2*l5*l6+l1*l2*l5*l7+l1*l2*l6*l7+l1*l3*l5*l6+l1*l3*l5*l7+l1*l3*l6*l7
                +l1*l5*l6*l7+l2*l3*l5*l6+l2*l3*l5*l7+l2*l3*l6*l7+l2*l5*l6*l7+l3*l5*l6*l7)
        - HH[2]*(l1*l2*l3+l1*l2*l5+l1*l2*l6+l1*l2*l7+l1*l3*l5+l1*l3*l6+l1*l3*l7+l1*l5*l6+l1*l5*l7+l1*l6*l7
                 +l2*l3*l5+l2*l3*l6+l2*l3*l7+l2*l5*l6+l2*l5*l7+l2*l6*l7+l3*l5*l6+l3*l5*l7+l3*l6*l7+l5*l6*l7)
        + HH[3]*(l1*l2+l1*l3+l1*l5+l1*l6+l1*l7+l2*l3+l2*l5+l2*l6+l2*l7+l3*l5+l3*l6+l3*l7+l5*l6+l5*l7+l6*l7) 
        - HH[4]*(l1+l2+l3+l5+l6+l7)
        + HH[5])
        /(l1-l4)/(l2-l4)/(l3-l4)/(l5-l4)/(l6-l4)/(l7-l4)
        )
        XX.append(
        (Iden*l1*l2*l3*l4*l6*l7 
        - HH[0]*(l1*l2*l3*l4*l6+l1*l2*l3*l4*l7+l1*l2*l3*l6*l7+l1*l2*l4*l6*l7+l1*l3*l4*l6*l7+l2*l3*l4*l6*l7) 
        + HH[1]*(l1*l2*l3*l4+l1*l2*l3*l6+l1*l2*l3*l7+l1*l2*l4*l6+l1*l2*l4*l7+l1*l2*l6*l7+l1*l3*l4*l6+l1*l3*l4*l7+l1*l3*l6*l7
                +l1*l4*l6*l7+l2*l3*l4*l6+l2*l3*l4*l7+l2*l3*l6*l7+l2*l4*l6*l7+l3*l4*l6*l7)
        - HH[2]*(l1*l2*l3+l1*l2*l4+l1*l2*l6+l1*l2*l7+l1*l3*l4+l1*l3*l6+l1*l3*l7+l1*l4*l6+l1*l4*l7+l1*l6*l7
                 +l2*l3*l4+l2*l3*l6+l2*l3*l7+l2*l4*l6+l2*l4*l7+l2*l6*l7+l3*l4*l6+l3*l4*l7+l3*l6*l7+l4*l6*l7)
        + HH[3]*(l1*l2+l1*l3+l1*l4+l1*l6+l1*l7+l2*l3+l2*l4+l2*l6+l2*l7+l3*l4+l3*l6+l3*l7+l4*l6+l4*l7+l6*l7) 
        - HH[4]*(l1+l2+l3+l4+l6+l7)
        + HH[5])
        /(l1-l5)/(l2-l5)/(l3-l5)/(l4-l5)/(l6-l5)/(l7-l5)
        )
        XX.append(
        (Iden*l1*l2*l3*l4*l5*l7 
        - HH[0]*(l1*l2*l3*l4*l5+l1*l2*l3*l4*l7+l1*l2*l3*l5*l7+l1*l2*l4*l5*l7+l1*l3*l4*l5*l7+l2*l3*l4*l5*l7) 
        + HH[1]*(l1*l2*l3*l4+l1*l2*l3*l5+l1*l2*l3*l7+l1*l2*l4*l5+l1*l2*l4*l7+l1*l2*l5*l7+l1*l3*l4*l5+l1*l3*l4*l7+l1*l3*l5*l7
                +l1*l4*l5*l7+l2*l3*l4*l5+l2*l3*l4*l7+l2*l3*l5*l7+l2*l4*l5*l7+l3*l4*l5*l7)
        - HH[2]*(l1*l2*l3+l1*l2*l4+l1*l2*l5+l1*l2*l7+l1*l3*l4+l1*l3*l5+l1*l3*l7+l1*l4*l5+l1*l4*l7+l1*l5*l7
                 +l2*l3*l4+l2*l3*l5+l2*l3*l7+l2*l4*l5+l2*l4*l7+l2*l5*l7+l3*l4*l5+l3*l4*l7+l3*l5*l7+l4*l5*l7)
        + HH[3]*(l1*l2+l1*l3+l1*l4+l1*l5+l1*l7+l2*l3+l2*l4+l2*l5+l2*l7+l3*l4+l3*l5+l3*l7+l4*l5+l4*l7+l5*l7) 
        - HH[4]*(l1+l2+l3+l4+l5+l7)
        + HH[5])
        /(l1-l6)/(l2-l6)/(l3-l6)/(l4-l6)/(l5-l6)/(l7-l6)
        )
        XX.append(
        (Iden*l1*l2*l3*l4*l5*l6 
        - HH[0]*(l1*l2*l3*l4*l5+l1*l2*l3*l4*l6+l1*l2*l3*l5*l6+l1*l2*l4*l5*l6+l1*l3*l4*l5*l6+l2*l3*l4*l5*l6) 
        + HH[1]*(l1*l2*l3*l4+l1*l2*l3*l5+l1*l2*l3*l6+l1*l2*l4*l5+l1*l2*l4*l6+l1*l2*l5*l6+l1*l3*l4*l5+l1*l3*l4*l6+l1*l3*l5*l6
                +l1*l4*l5*l6+l2*l3*l4*l5+l2*l3*l4*l6+l2*l3*l5*l6+l2*l4*l5*l6+l3*l4*l5*l6)
        - HH[2]*(l1*l2*l3+l1*l2*l4+l1*l2*l5+l1*l2*l6+l1*l3*l4+l1*l3*l5+l1*l3*l6+l1*l4*l5+l1*l4*l6+l1*l5*l6
                 +l2*l3*l4+l2*l3*l5+l2*l3*l6+l2*l4*l5+l2*l4*l6+l2*l5*l6+l3*l4*l5+l3*l4*l6+l3*l5*l6+l4*l5*l6)
        + HH[3]*(l1*l2+l1*l3+l1*l4+l1*l5+l1*l6+l2*l3+l2*l4+l2*l5+l2*l6+l3*l4+l3*l5+l3*l6+l4*l5+l4*l6+l5*l6) 
        - HH[4]*(l1+l2+l3+l4+l5+l6)
        + HH[5])
        /(l1-l7)/(l2-l7)/(l3-l7)/(l4-l7)/(l5-l7)/(l6-l7)
        )

    else:
        print("The dimension of the system is greater than 7. Analytic expression should be constructed by the user.")
        sys.exit()

    return XX


def SS(b, a, L, E, H, U):
    """ Calculate the d x d S matrix where the input Hamiltonian is in the vacuum mass basis
    params:
    - b, a (int): transition of a neutrino from flavor a to b
      a = (0, 1, 2, 3, 4, ..., d-1) = (e, \mu, \tau, s_1, s_2, ..., s_{d-3})
    - L [km]: the distance the neutrino has traveled
    - E [GeV]: energy of the neutrino
    - H [eV^2/GeV]: d x d matrix in the vacuum mass basis
    - U [dimensionless]: d x d mixing matrix
    """ 
    
    d = len(H)
    ll = eigenvalues(H) # Determine distinct eigenvalues
    d_eigen = len(ll)
    XX = mixing(H)
    
    # Sum over the distinct eigenvalues
    Xtot = np.zeros(shape=(d, d), dtype=complex)
    for i in range(d_eigen):
        Xtot = Xtot + XX[i]*np.exp(-1j*CONV_L*ll[i]*L)

    S = 0
    for m in range(d):
        for n in range(d):
            S = S + U[b, m]*Xtot[m, n]*np.conjugate(U[a, n])
        
    return S


def nuprobe(a, b, L, E, mass, UU, antinu=False, V_NSI=None):
    """ Calculate the oscillation probability of neutrino of flavor a to b
    params:
    - a, b (int): transition of a neutrino from flavor a to b
      a = (1, 2, 3, ..., d) = (e, \mu, \tau, s_1, s_2, ..., s_{d-3})
    - L [km]: the distance the neutrino has traveled
    - E [GeV]: energy of the neutrino
    - mass [eV]: d vector of neutrino masses
    - U [dimensionless]: d x d mixing matrix
    - antinu (bool): if true, anti neutrino will be considered (default if false)
    - V_NSI [dimensionless]: d x d Hermitian matrix defined with respect to the matter potential of charge current (arXiv:2106.07755)
    """ 

    # Shift the neutrino flavor indices for convenience 
    a = a - 1
    b = b - 1

    SSi = []
    d = len(mass)

    if V_NSI is None:
        V_NSI = np.zeros((d, d))

    if const_matter:
        V = V_matter(d, rho_const, 0.5, 1.06, V_NSI)

        if antinu: 
            UU = np.conjugate(UU)
            V = -V
        
        H = hamiltonian(E, mass, UU, V) 
        Prob = np.abs(SS(b, a, L, E, H, UU))**2
        
    else:
        dl = len(rho_LL)
        rhol = rho_LL

        for i in range(dl):
            V = V_matter(d, rhol[i], 0.5, 1.06, V_NSI)

            if antinu: 
                UU = np.conjugate(UU)
                V = -V

            H = hamiltonian(E, mass, UU, V) 
            
            Li = ((L - LL[i])*np.heaviside(LL[i+1]-L, 1)+(LL[i+1]-LL[i])*np.heaviside(L-LL[i+1], 1))*np.heaviside(L-LL[i], 1)
            
            Si = np.identity(d, dtype = complex)
            for m in range(d):
                for n in range(d):
                    Si[m, n] = SS(m, n, Li, E, H, UU)
            
            SSi.append(Si)

        Stot = SSi[0]
        
        for i in range(dl-1):
            Stot = SSi[i+1] @ Stot

        Prob = np.abs(Stot[b, a])**2 
    
    if Prob - 1 > rtol: 
        print("WARNING: Probability greater than 1. Check the relative tolerance rtol.")
        #sys.exit()

    return Prob


def V_matter(d, rho, Ye=0.5, Yn=1.06, V_NSI=None):
    """ Calculate the d x d matter potential [eV^2/GeV]
    params:
    - d (int): number of neutrino flavors
    - rho [g/cm^3]: matter density
    - Ye [dimensionless]: average number of electron per nucleon
    - Yn [dimensionless]: average number of neutron per electron
    - V_NSI [dimensionless]: d x d Hermitian matrix defined with respect to the matter potential of charge current (arXiv:2106.07755)
    """ 

    if V_NSI is None:
        V_NSI = np.zeros((d, d))

    ne = rho*Ye
    nn = Yn*ne
    Vd = np.zeros(d)
    Vd[0] = CONV_matter*(ne-nn/2)
    Vd[1] = CONV_matter*(-nn/2)
    Vd[2] = CONV_matter*(-nn/2)

    V = np.diag(Vd) + CONV_matter*ne*V_NSI

    return V


def hamiltonian(E, mass, U, V, flavor_basis=False):
    """ Calculate the d x d Hamiltonian is in the vacuum mass or flavor basis [eV^2/GeV]
    params:
    - E [GeV]: energy of the neutrino
    - mass [eV]: d vector of neutrino masses
    - U [dimensionless]: d x d mixing matrix
    - V [eV^2/GeV]: d x d matrix of matter potential
    - flavor_basis (bool): if true, the Hamiltonian will be in the flavor basis, otherwise, it will be in the vacuum mass basis
    """ 

    Uh = np.transpose(np.conjugate(U))
    Ui = LA.inv(U)

    if flavor_basis:
        H = U @ np.diag(mass**2)/(2*E) @ Uh + V 
    else:
        H = np.diag(mass**2)/(2*E) + Uh @ V @ U

    return H


def Jt(b, a, j, k, H, U):
    """ Calculate the element of Jarlskog invariant J_{ba}^{jk}
    params:
    - b, a, j, k (int): the indices of J_{ba}^{jk}
    - H [eV^2/GeV]: d x d Hamiltonian in the flavor basis
    - U [dimensionless]: d x d mixing matrix
    """ 

    # Shift the neutrino flavor indices for convenience 
    a = a - 1
    b = b - 1
    j = j - 1
    k = k - 1

    d = len(H)
    XX = mixing(H)

    Uj = 0
    Uk = 0
    for m in range(d):
        for n in range(d):
            Uj = Uj + U[b, m]*XX[j][m, n]*np.conjugate(U[a, n])
            Uk = Uk + U[a, m]*XX[k][m, n]*np.conjugate(U[b, n])

    J = np.imag(Uj*Uk)
    
    return J


def NHS(b, a, j, k, H, U):
    """ Calculate the Naumov-Harrison-Scott combination \lambda_{21}\lambda_{31}\lambda_{32}J_{ba}^{jk}
    params:
    - b, a, j, k (int): the indices of J_{ba}^{jk}
    - H [eV^2/GeV]: d x d Hamiltonian in the flavor basis
    - U [dimensionless]: d x d mixing matrix
    """ 

    # Shift the neutrino flavor indices for convenience 
    a = a - 1
    b = b - 1
    j = j - 1
    k = k - 1

    d = len(H)
    if d != 3:
        print("For the moment, this applies only for the case with 3-flavors")
        sys.exit()

    XX = mixing(H)

    Uj = 0
    Uk = 0
    for m in range(d):
        for n in range(d):
            Uj = Uj + U[b, m]*XX[j][m, n]*np.conjugate(U[a, n])
            Uk = Uk + U[a, m]*XX[k][m, n]*np.conjugate(U[b, n])

    J = np.imag(Uj*Uk)
    ll = eigenvalues(H)
    lcom = (ll[1] - ll[0])*(ll[2] - ll[0])*(ll[2] - ll[1])
    
    return J*lcom
    


