import numpy as np
from kithcmb import kitaevhoneycomb
from kithcmb import parityprojectedfermions
from kithcmb import montecarlo
from matplotlib import pyplot as plt

# set of temperatures to compute thermal energy and vortex density at
T_vals = np.logspace(-3,3,7)
thermal_energies = np.zeros(T_vals.size)
vortex_densities = np.zeros(T_vals.size)

# number of rounds of Monte Carlo to carry out at each temperature 
mcmc_rounds = 1000

# fix the parameter values and system size
J = [1,1,1]
K = 0.1
Lrows,Lcols = 4,4

for Tidx,T in enumerate(T_vals):
    print('\n'+'-'*30+'\n'+'T='+"{0:.3g}".format(T)+'\n'+'-'*30+'\n')

    # initialise montecarlo object, this uses default options:
    # MC_step_type='vortex',change_topo_sector=True,project_fermions=True
    mcmc_obj = montecarlo.montecarlo(J=J,K=K)
    
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
    plt.show(block=False)
    fig,ax = plt.subplots(figsize=(8,6))
    plt.title('vortex density timeseries')
    ax.plot(vortex_density_timeseries,'-')
    plt.show(block=False)
    
    # allow for burn-in and compute averages with last 90% of timeseries
    thermal_energies[Tidx] = np.mean(thermal_energy_timeseries[mcmc_rounds//10:])
    vortex_densities[Tidx] = np.mean(vortex_density_timeseries[mcmc_rounds//10:])
 
#
# plot average thermal energies and vortex densities as functions of temperature
#
fig,ax = plt.subplots(figsize=(8,6))
ax.plot(T_vals,thermal_energies,'o-',markerfacecolor='w',markersize=12)
ax.set_xscale('log')
ax.set_xlabel(r'$T$',fontsize=22)
ax.set_ylabel(r'$\langle E \rangle$',fontsize=22)
plt.show(block=False)

fig,ax = plt.subplots(figsize=(8,6))
ax.plot(T_vals,vortex_densities,'o-',markerfacecolor='w',markersize=12,color='C1')
ax.set_xscale('log')
ax.set_xlabel(r'$T$',fontsize=22)
ax.set_ylabel('vortex density',fontsize=22)
plt.show(block=False)