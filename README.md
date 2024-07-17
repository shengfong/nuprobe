# NuProbe: Neutrino oscillation as a Probe of New Physics

If you use (and extend) this code, please cite our work

- **Analytic Neutrino Oscillation Probabilities**\
  Chee Sheng Fong\
  https://arxiv.org/abs/2210.09436 \
  [SciPost Phys. **15** (2023) 1, 013](https://scipost.org/10.21468/SciPostPhys.15.1.013)
  
  
## Abstract

We have implemented analytic expression for neutrino oscillation probabilities up to system with 3+4 neutrino flavors in an arbitrary matter potential based on the [article](https://arxiv.org/abs/2210.09436). It is built such that the user can easily specify the masses, mixing angles and phases related to neutrinos, nonunitary parameters, extend the Standard Model (SM) matter potential with NonStandard neutrino Interaction (NSI) parameters and specify matter density profile in layers of constant matter densities. Example applications to nonunitary, NSI, quasi-Dirac neutrino scenarios are given. For the Earth-crossing neutrinos, a simplified 4-layer PREM model is used.


## Changelog

**October 16, 2022** \
In this initial release, we have implemented up to system with 3+4 neutrino flavors. For the Earth matter density profile, we have implemented a simplified 4-layer PREM model. 

## Installation 
To install from the github repository:
```
git clone https://github.com/shengfong/nuprobe.git
cd nuprobe
pip install -e .
```

## Usage
The two basic packages are $\texttt{inputs.py}$ and $\texttt{probability.py}$:
```
from nuprobe.nuinputs import NuSystem, create_U_PMNS, create_U_NEW
from nuprobe.probability import nuprobe
```
In the above, we have imported $\texttt{NuSystem}$ that allows us to specify the number of neutrinos, the mixing angles and phases, nonunitary and NSI parameters. The function `create_U_PMNS` will create the PMNS matrix with the standard three neutrinos angle and phases while the function `create_U_NEW` will create matrix involving only the new angles and phases. The function $\texttt{nuprobe}$ is to calculate the neutrino oscillation probability. We will explain these in more details shortly.

First of all, let us specify the numbers of neutrino flavor, say 4:
```
nu_sys = NuSystem(4)
```
Then, we set the standard three-flavor parameters to, say inverse mass ordering (by default, it is normal mass ordering):
```
nu_sys.set_standard_normal(False)
```
The default values are specified in $\texttt{params.py}$.

Now let us specify the properties of the fourth neutrino. Say we would like a fourth neutrino with a mass of 2 eV with mixing angles $\theta_{14} = 0.2$, $\theta_{24} = 0.1$ and $\theta_{34} = 0.3$ and phase $\delta_{14} = 1.8$ (mixing angles and phases are in radian). All other parameters not specified will be set to zero.  
```
nu_sys.set_mass(4, 2)
nu_sys.set_theta(1, 4, 0.2)
nu_sys.set_theta(2, 4, 0.1)
nu_sys.set_theta(3, 4, 0.3)
nu_sys.set_delta(1, 4, 1.8)
```
Next we will construct the rotation matrices by calling:
```
U0 = create_U_PMNS(nu_sys.theta, nu_sys.delta)
UNP = create_U_NEW(nu_sys.theta, nu_sys.delta)
U = UNP @ U0
```
For a system with $d$ neutrino flavors, the function `create_U_PMNS` creates a $d \times d$ matrix with $3 \times 3$ submatrix as the standard three-flavor mixing (PMNS) matrix. The function `create_U_NEW` will create $d \times d$ matrix involving only new angles and phases. The full rotation matrix is obtained by a matrix multiplication between the two. 

To calculate the neutrino oscillation probability, we will call the function `nuprobe(a, b, L, E, mass, U, antinu, const_matter, V_NSI)` with the following input parameters:

- $\texttt{a, b}$ (int): transition of a neutrino from flavor a to b with $(1, 2, 3, 4, ..., d) = (e, \mu, \tau, s_1, ..., s_{d-3})$
- $\texttt{L}$ [km]: the distance the neutrino has traveled
- $\texttt{E}$ [GeV]: energy of the neutrino
- $\texttt{mass}$ [eV]: $d$ vector of neutrino masses
- $\texttt{U}$ [dimensionless]: $d \times d$ mixing matrix
- $\texttt{antinu}$ (bool): if true, anti neutrino will be considered (default if false)
- `const_matter` (bool): if true, constant matter potential will be considered (default is true)
- `V_NSI` [dimensionless]: $d \times d$ Hermitian matrix defined with respect to the matter potential of charge current as in [arXiv:2106.07755](https://arxiv.org/abs/2106.07755).
 
For instance, to calculate the probability of $\bar\nu_\mu \to \bar\nu_e$ for neutrino of energy $E = 2$ GeV at a distance $L = 1000$ km, we write
```
L = 1000
E = 2
P = nuprobe(2, 1, L, E, nu_sys.mass, U, antinu=True, const_matter=True, V_NSI=None)
print('P =', P)
```
Running this code, we obtain
```
P = 0.04770409710452118
```

By default, the SM density matter potential with constant density is used. The relevant parameters are specified in $\texttt{matter.py}$:
```
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
```
If `const_matter =  True`, a constant matter density (g/cm<sup>3</sup>) specified in `rho_const = 3` will be used. If `const_matter =  False`, the matter density profile can be specified by a $D+1$ vector of layers in km and a $D$ vector of matter densities in g/cm<sup>3</sup>. The default is a four-layer simplified PREM model for neutrino passing through the Earth core.

Examples of nonunitary, NSI and quasi-Dirac neutrino scenarios that are used to produced the figures in the [article](https://arxiv.org/abs/2210.09436) are contained in the $\texttt{examples}$ folder.


## Contact
To report bugs, to give suggestions for improvements, please write to sheng [dot] fong [at] ufabc [dot] edu [dot] br.


