# Kitaev Honeycomb in Python

Python package to simulate the Kitaev honeycomb model

## Contents
* [Setup](#setup)
* [Usage](#usage)
* [Examples](#examples)

## Setup

Clone the repository onto your machine. 

The package is called `kithcmb` and is in the `packages` folder. To make the package accessible from anywhere add the package to your python path environment variable. This can be done to putting the following in your `~/.bash_profile` file: `export PYTHONPATH="$PYTHONPATH:/path/to/repository/packages"`

Import the package to python using `import kithcmb`.

## Usage

There are three modules in the package:

`kitaevhoneycomb`  
`parityprojectedfermions`  
`montecarlo`

### `kitaevhoneycomb`

This module contains a class definition, also called `kitaevhoneycomb`. Objects of this class represent a realisation of the kitaev honeycomb model for a __fixed vortex sector__. The class provides methods to diagonalise the sector and obtain the spectrum and eigenvectors.  

The arguments to the class are all passed as keyword arguments. They are the coupling strengths of the model `J` and `K`, as well as the link gauge variables `ux`, `uy`, `uz`, example:

```python
# system size
Lrows,Lcols = 10,10

# middle of the phase diagram with a small gap at the fermi points
J = [1,1,1]
K = 0.1

# no-vortex sector
ux = np.ones((Lrows,Lcols),dtype=np.int8)
uy = np.ones((Lrows,Lcols),dtype=np.int8)
uz = np.ones((Lrows,Lcols),dtype=np.int8)

# initialse object
ky_sys = kitaevhoneycomb.kitaevhoneycomb(J=J,K=K,ux=ux,uy=uy,uz=uz) 

# draw the system
kh_sys.draw_system()
```

The link variables `u` encode the vortex (and topological) sector, but they can also be used to implement __different boundary conditions__. For example a cylinder can be created by setting a row of `uz` to zero

```python
uz[-1,:] = 0
```

Typically the `u` arrays will only have values +1, -1 and 0, hence why they are constructed with the data type `np.int8` in the example. However, the class should still work without them being integers. 

### `parityprojectedfermions`

This module contains functions that take the spectrum (and possibly eigenvectors) output by `kitaevhoneycomb` objects and compute finite temperature observables of the fermions such as the thermal energy or the correlation matrix. Fermionic states of the kitaev honeycomb are subject to a global quasiparticle parity constraint, each function in the module has a version that obeys the parity constraint and another version that ignores it.

The module has a special function `get_desired_parity` that computes the physical parity for a given sector, to do this it needs the `u` arrays and either the `Q` or `U` eigenvector matrices. These are passed as keyword args (see example "Obtaining the correlation matrix for a fixed vortex sector" below).

All other functions have two versions `get_unprojected_*` and `get_projected_*`, they have the same arguments but the projected version takes an additional argument which is the physical parity. All of these functions have the same return signatures, to give an example:

```python
thermal_energy,extras = parityprojectedfermions.get_projected_thermal_energy(spectrum,T,+1)
```

where `thermal_energy` is the result of the computation and `extras` is a dictionary of other physical quantities that were calculated in the process of computing the thermal energy. For example in the case of thermal energy these are the quasiparticle occupancies and the fermionic partition function. 

### `montecarlo`

This module generates a sequence of random vortex sectors, where each vortex sector occurs with the probability it would have in the thermal distribution over all vortex sectors. It does this using a Markov chain Monte Carlo approach. This can be used to estimate finite temperature observables for the complete model.

The module contains a class, also called `montecarlo`. The constructor for the class takes keyword args that save physical information about it needs about the system and fix the type of Markov steps the chain makes. 

At each step of the chain a new instance of `kitaevhoneycomb` is generated internally so the class needs to know the coupling constants `J` and `K`, additionally we set whether or not the fermions should be parity projected with the flag `project_fermions`. 

The keyword argument `MC_step_type` sets the type of Markov step proposals. It can take values `"link"` where a single `u` link is flipped at random, or `"vortex"` where a pair of vortices is inserted at random by flipping a chain of `u` links (depending on the previous configuration this could also have the effect of transporting a single vortex or annihilating two vortices). Additionally, the flag `change_topo_sector` fixes whether or not to superimpose a randomly selected non-trivial loop of `u` flips on top of the proposal. Due to the way the chains are constructed `"vortex"` proposals will never cross a boundary, so `change_topo_sector=True` is needed to explore the different topological sectors.

## Examples

Here are some simple examples of using each of the modules. More examples can be found in the `examples` directory.

#### Computing the spectrum for a fixed vortex sector (`kitaevhoneycomb`)

```python
import numpy as np
from kithcmb import kitaevhoneycomb
from matplotlib import pyplot as plt

# fix the parameter values and system size
J = [1,1,1]
K = 0.1
Lrows,Lcols = 10,10

# fix the values of the gauge fields. start from the no-vortex sector
ux = np.ones((Lrows,Lcols),dtype=np.int8)
uy = np.ones((Lrows,Lcols),dtype=np.int8)
uz = np.ones((Lrows,Lcols),dtype=np.int8)

# insert a pair of vortices by flipping a line of uz's
uz[Lrows//2,Lcols//4:3*Lcols//4] = -1

# initialise kitaevhoneycomb object
kh_sys = kitaevhoneycomb.kitaevhoneycomb(J=J,K=K,ux=ux,uy=uy,uz=uz) 

# (optionally) draw the system to see location of vortices, note this is slow
kh_sys.draw_system()

# obtain spectrum
spectrum = kh_sys.get_spectrum()

# plot the spectrum
fig,ax = plt.subplots(figsize=(8,6))
plt.title('spectrum',fontsize=24)
ax.plot(spectrum,'.')
ax.set_ylabel(r'$\varepsilon$',fontsize=24)
plt.show()
```

#### Obtaining the correlation matrix for a fixed vortex sector (`parityprojectedfermions`)

```python
import numpy as np
from kithcmb import kitaevhoneycomb
from kithcmb import parityprojectedfermions
from matplotlib import pyplot as plt

# temperature to compute correlation matrix at, set to zero
T = 0.

# fix the parameter values and system size
J = [1,1,1]
K = 0.1
Lrows,Lcols = 10,10

# fix the values of the gauge fields. start from the no-vortex sector
ux = np.ones((Lrows,Lcols),dtype=np.int8)
uy = np.ones((Lrows,Lcols),dtype=np.int8)
uz = np.ones((Lrows,Lcols),dtype=np.int8)

# go to the full vortex sector by flipping every second uz along rows
uz[:,::2] = -1

# initialise kitaevhoneycomb object
kh_sys = kitaevhoneycomb.kitaevhoneycomb(J=J,K=K,ux=ux,uy=uy,uz=uz) 

# (optionally) draw the system to see location of vortices, note this is slow
kh_sys.draw_system()

# obtain eigenvectors and eigenvalues of A matrix
D,U = kh_sys.get_diagonal_form()

# plot spectrum
fig,ax = plt.subplots(figsize=(8,6))
ax.plot(np.diag(np.real(1j*D)),'.')
plt.title('spectrum',fontsize=24)
ax.set_ylabel(r'$\varepsilon$',fontsize=24)
plt.show()

# -------------
# version 1: obtain the correlation matrix WITHOUT parity constraint
# -------------
print('\n'+'-'*30+'\n'+'UNPROJECTED'+'\n'+'-'*30+'\n')
corr_mat,extras = parityprojectedfermions.get_unprojected_correlation_matrix(D,U,T)

# plot the quasipaticle occupations in extras
fig,ax = plt.subplots(figsize=(8,6))
ax.plot(extras['quasi_occs'],'.')
plt.title('quasiparticle occupations',fontsize=24)
plt.show()

# plot real and imaginary part of correlation matrix
fig,(axl,axr) = plt.subplots(1,2,figsize=(16,6))
plt.suptitle('correlation matrix',fontsize=24)
im = axl.imshow(np.real(corr_mat))
plt.colorbar(im,ax=axl)
axl.set_title('real',fontsize=22)
im = axr.imshow(np.imag(corr_mat))
plt.colorbar(im,ax=axr)
axr.set_title('imag',fontsize=22)
plt.show()

# -------------
# version 2: obtain the correlation matrix WITH parity constraint
# -------------
print('\n'+'-'*30+'\n'+'PROJECTED'+'\n'+'-'*30+'\n')
desired_parity = parityprojectedfermions.get_desired_parity(ux=ux,uy=uy,uz=uz,U=U)
corr_mat,extras = parityprojectedfermions.get_projected_correlation_matrix(D,U,T,desired_parity)

# plot the quasipaticle occupations in extras
fig,ax = plt.subplots(figsize=(8,6))
ax.plot(extras['quasi_occs'],'.')
plt.title('quasiparticle occupations',fontsize=24)
plt.show()

# plot real and imaginary part of correlation matrix
fig,(axl,axr) = plt.subplots(1,2,figsize=(16,6))
plt.suptitle('correlation matrix',fontsize=24)
im = axl.imshow(np.real(corr_mat))
plt.colorbar(im,ax=axl)
axl.set_title('real',fontsize=22)
im = axr.imshow(np.imag(corr_mat))
plt.colorbar(im,ax=axr)
axr.set_title('imag',fontsize=22)
plt.show()
```

#### Estimating thermal energy and vortex density for the spin model (`montecarlo`)

```python
import numpy as np
from kithcmb import kitaevhoneycomb
from kithcmb import parityprojectedfermions
from kithcmb import montecarlo
from matplotlib import pyplot as plt
%matplotlib inline

# set of temperatures to compute thermal energy and vortex density at
T_vals = np.logspace(-3,3,7)
thermal_energies = np.zeros(T_vals.size)
vortex_densities = np.zeros(T_vals.size)

# number of rounds of Monte Carlo steps to carry out at each temperature 
mcmc_rounds = 1000

# fix the parameter values and system size
J = [1,1,1]
K = 0.1
Lrows,Lcols = 4,4

for Tidx,T in enumerate(T_vals):
    print('\n'+'-'*30+'\n'+'T='+"{0:.3g}".format(T)+'\n'+'-'*30+'\n')

    # initialise montecarlo object, this uses default options:
    # MC_step_type='vortex',change_topo_sector=True,project_fermions=True
    mcmc_obj = montecarlo.montecarlo(J=J,K=K,project_fermions=False)
    
    # store Monte Carlo timeseries of computed values
    thermal_energy_timeseries = np.zeros(mcmc_rounds)
    vortex_density_timeseries = np.zeros(mcmc_rounds)
    
    # start from a ground state, no-vortex sector with anti-periodic BC (fully periodic BC 
    # no-vortex sector projects to odd fermion parity and so is not a ground state)
    ux = np.ones((Lrows,Lcols),dtype=np.int8)
    uy = np.ones((Lrows,Lcols),dtype=np.int8)
    uz = np.ones((Lrows,Lcols),dtype=np.int8)
    uz[-1,:] = -1

    # initialise kitaevhoneycomb object
    kh_sys = kitaevhoneycomb.kitaevhoneycomb(J=J,K=K,ux=ux,uy=uy,uz=uz) 
    # (optionally) draw the initial state of the system to see the BC, note this is slow
    kh_sys.draw_system()
    
    # carry out Monte Carlo updates, each round is a 'sweep' of (Lrows*Lcols) Metropolis steps
    for i in range(mcmc_rounds):
        ux,uy,uz = mcmc_obj.metropolis_sweep(ux,uy,uz,T)
        
        # initialise a kitaevhoneycomb object with the current configuration and diagonalise
        kh_sys = kitaevhoneycomb.kitaevhoneycomb(J=J,K=K,ux=ux,uy=uy,uz=uz) 
        Sigma,Q = kh_sys.get_normal_form()
        spectrum = np.sort(np.sum(Sigma,axis=0))
        
        # get current value of vortex density and thermal energy
        desired_parity = parityprojectedfermions.get_desired_parity(ux=ux,uy=uy,uz=uz,Q=Q)
        thermal_energy_timeseries[i],extras = parityprojectedfermions.get_projected_thermal_energy(spectrum,T,desired_parity)
        vortex_density_timeseries[i] = np.sum(0.5*(1.-kh_sys.vortices))/ux.size
        
    # (optionally) draw very final state of the system
    kh_sys.draw_system()
        
    # plot timeseries
    fig,ax = plt.subplots(figsize=(8,6))
    plt.title('thermal energy timeseries')
    ax.plot(thermal_energy_timeseries,'-')
    plt.show()
    fig,ax = plt.subplots(figsize=(8,6))
    plt.title('vortex density timeseries')
    ax.plot(vortex_density_timeseries,'-')
    plt.show()
    
    # allow for burn-in and compute averages with last 90% of timeseries
    thermal_energies[Tidx] = np.mean(thermal_energy_timeseries[mcmc_rounds//10:])
    vortex_densities[Tidx] = np.mean(vortex_density_timeseries[mcmc_rounds//10:])
    
fig,ax = plt.subplots(figsize=(8,6))
ax.plot(T_vals,thermal_energies,'o-',markerfacecolor='w',markersize=12)
ax.set_xscale('log')
ax.set_xlabel(r'$T$',fontsize=22)
ax.set_ylabel(r'$\langle E \rangle$',fontsize=22)
plt.show()

fig,ax = plt.subplots(figsize=(8,6))
ax.plot(T_vals,vortex_densities,'o-',markerfacecolor='w',markersize=12,color='C1')
ax.set_xscale('log')
ax.set_xlabel(r'$T$',fontsize=22)
ax.set_ylabel('vortex density',fontsize=22)
plt.show()
```
