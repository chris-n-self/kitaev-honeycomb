# Kitaev Honeycomb in Python

## Setup

Clone the repository onto your machine. 

The package is called `kithcmb` and is in the `packages` folder. To make the package accessible from anywhere add the package to your python path environment variable. This can be done to putting the following in your `~/.bash_profile` file:  
`export PYTHONPATH="$PYTHONPATH:/path/to/repository/packages"`

Import the package to python using `import kithcmb`.

## Usage

There are three modules in the package:

`kitaevhoneycomb`  
`parityprojectedfermions`  
`montecarlo`

The first module `kitaevhoneycomb` contains a class definition, also called `kitaevhoneycomb`. Objects of this class represent a realisation of the kitaev honeycomb model for a fixed vortex sector. The class provides methods to diagonalise the sector and obtain the spectrum and eigenvectors.  

Module `parityprojectedfermions` contains functions that take the spectrum (and possibly eigenvectors) output by `kitaevhoneycomb` objects and compute finite temperature observables such as the thermal energy or the fermionic correlation matrix. Fermionic states of the kitaev honeycomb are subject to a global quasiparticle parity constraint, each function in the module has a version that obeys the parity constraint and another version that ignores it.

The final module `montecarlo` generates a sequence of random vortex sectors, where each vortex sector occurs with the probability it would have in the thermal distribution over all vortex sectors. It does this using a Markov chain Monte Carlo approach. This can be used to estimate finite temperature observables for the complete model.

### Examples

Here are some simple examples of using each of the modules. More examples can be found in the `examples` directory.

#### Computing the spectrum for a fixed vortex sector

    import numpy as np
    from kithcmb import kitaevhoneycomb
    from matplotlib import pyplot as plt

    # fix the parameter values and system size
    J = [1,1,1]
    K = 0.1
    Lrows,Lcols = 10,10

    # start from the no-vortex sector
    Ux = np.ones((Lrows,Lcols),dtype=np.int8)
    Uy = np.ones((Lrows,Lcols),dtype=np.int8)
    Uz = np.ones((Lrows,Lcols),dtype=np.int8)

    # insert a pair of vortices by flipping a line of Uz's
    Uz[Lrows//2,Lcols//4:3*Lcols//4] = -1

    # initialise kitaevhoneycomb object
    kh_sys = kitaevhoneycomb.kitaevhoneycomb(J=J,K=K,Ux=Ux,Uy=Uy,Uz=Uz) 

    # (optionally) draw the system to see location of vortices
    kh_sys.draw_system()

    # diagonalise
    D,U = kh_sys.get_diagonal_form()

    # plot the spectrum
    fig,ax = plt.subplots()
    ax.plot(np.diag(D),'.')
    plt.show()

#### Obtaining the correlation matrix for a fixed vortex sector

#### Estimating the thermal energy averaging over vortex sectors