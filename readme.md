# Kitaev Honeycomb in Python

Python package to simulate the Kitaev honeycomb model

## Contents
* [Setup](##setup)
* [Usage](##usage)
* [Examples](##examples)

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

The link variables `u` encode the vortex (and topological) sector, but they can also be used to implement __different boundary conditions__. For example a cylinder can be created by setting a row of `uz` to zero

	uz[-1,:] = 0

Typically the `u` arrays will only have values +1, -1 and 0, hence why they are constructed with the data type `np.int8` in the example. However, the class should still work without them being integers. 

### `parityprojectedfermions`

This module contains functions that take the spectrum (and possibly eigenvectors) output by `kitaevhoneycomb` objects and compute finite temperature observables of the fermions such as the thermal energy or the correlation matrix. Fermionic states of the kitaev honeycomb are subject to a global quasiparticle parity constraint, each function in the module has a version that obeys the parity constraint and another version that ignores it.

The module has a special function `get_desired_parity` that computes the physical parity for a given sector, to do this it needs the `u` arrays and either the `Q` or `U` eigenvector matrices. These are passed as keyword args (see example "Obtaining the correlation matrix for a fixed vortex sector" below).

All other functions have two versions `get_unprojected_*` and `get_projected_*`, they have the same arguments but the projected version takes an additional argument which is the physical parity. All of these functions have the same return signatures, to give an example:

	thermal_energy,extras = parityprojectedfermions.get_projected_thermal_energy(spectrum,beta,+1)

where `thermal_energy` is the result of the computation and `extras` is a dictionary of other physical quantities that were calculated in the process of computing the thermal energy. For example in the case of thermal energy these are the quasiparticle occupancies and the fermionic partition function. 

### `montecarlo`

This module generates a sequence of random vortex sectors, where each vortex sector occurs with the probability it would have in the thermal distribution over all vortex sectors. It does this using a Markov chain Monte Carlo approach. This can be used to estimate finite temperature observables for the complete model.

The module contains a class, also called `montecarlo`. The constructor for the class takes keyword args that save physical information about it needs about the system and fix the type of Markov steps the chain makes. 

At each step of the chain a new instance of `kitaevhoneycomb` is generated internally so the class needs to know the coupling constants `J` and `K`, additionally we set whether or not the fermions should be parity projected with the flag `project_fermions`. 

The keyword argument `MC_step_type` sets the type of Markov step proposals. It can take values `"link"` where a single `u` link is flipped at random, or `"vortex"` where a pair of vortices is inserted at random by flipping a chain of `u` links (depending on the previous configuration this could also have the effect of transporting a single vortex or annihilating two vortices). Additionally, the flag `change_topo_sector` fixes whether or not to superimpose a randomly selected non-trivial loop of `u` flips on top of the proposal. Due to the way the chains are constructed `"vortex"` proposals will never cross a boundary, so `change_topo_sector=True` is needed to explore the different topological sectors.

## Examples

Here are some simple examples of using each of the modules. More examples can be found in the `examples` directory.

#### Computing the spectrum for a fixed vortex sector

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

    # (optionally) draw the system to see location of vortices
    kh_sys.draw_system()

    # diagonalise
    spectrum = kh_sys.get_spectrum()

    # plot the spectrum
    fig,ax = plt.subplots()
    ax.plot(spectrum,'.')
    plt.show()

#### Obtaining the correlation matrix for a fixed vortex sector

    import numpy as np
    from kithcmb import kitaevhoneycomb
    from kithcmb import parityprojectedfermions
    from matplotlib import pyplot as plt

    # temperature to compute correlation matrix at, set to zero
    T = 1E-6

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

    # (optionally) draw the system to see location of vortices
    kh_sys.draw_system()

    # obtain eigenvectors and eigenvalues of A matrix
    D,U = kh_sys.get_diagonal_form()

	# -------------
    # version 1: obtain the correlation matrix without parity constraint
    # -------------
    corr_mat = parityprojectedfermions.get_unprojected_correlation_matrix()


#### Estimating the thermal energy averaging over vortex sectors