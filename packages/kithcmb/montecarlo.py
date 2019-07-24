# 09/07/19
# Chris Self

import numpy as np
from . import parityprojectedfermions as projferms
from . import kitaevhoneycomb as kithcmb

def single_link_flip(ux,uy,uz):
    """ 
    Generate a new set of u arrays where one link has been flipped from
    the arg arrays ux,uy,uz
    """
    
    # choose a random u-link
    row = np.random.randint(ux.shape[0])
    col = np.random.randint(ux.shape[1])
    l = np.random.randint(3)

    # flip the link
    if l==0:
        ux[row,col] = -1*ux[row,col]
    elif l==1:
        uy[row,col] = -1*uy[row,col]
    elif l==2:
        uz[row,col] = -1*uz[row,col]
        
    return ux,uy,uz

def add_vortex_pair(ux,uy,uz):
    """ 
    Generate a new set of u arrays where a vortex pair has been added
    relative to the arg arrays ux,uy,uz. (This doesn't necessarily actually
    add a pair of vortices, depending on the previous state of the system.
    It could instead move vortices or annihilate a pair.)
    """

    # random positions for vortices
    v1_row = np.random.randint(ux.shape[0])
    v1_col = np.random.randint(ux.shape[1])
    v2_row = np.random.randint(ux.shape[0])
    v2_col = np.random.randint(ux.shape[1])

    # detect trivial case
    if (v1_row==v2_row) and (v1_col==v2_col):
        return ux,uy,uz

    # sort corners of box bounding v1 and v2
    r1,r2 = min(v1_row,v2_row),max(v1_row,v2_row)
    c1,c2 = min(v1_col,v2_col),max(v1_col,v2_col)

    # implement flips. these copies need to be here or we will modify the u arrays
    # outside of this function and lose the current configuration
    ux = np.array(ux,copy=True)
    uy = np.array(uy,copy=True)
    uz = np.array(uz,copy=True)
    uy[r1+1:r2+1,c1] = -1*uy[r1+1:r2+1,c1]
    if (v1_row==r1 and v1_col==c1) or (v2_row==r1 and v2_col==c1):
        uz[r2,c1+1:c2+1] = -1*uz[r2,c1+1:c2+1]
    else:
        uz[r1,c1+1:c2+1] = -1*uz[r1,c1+1:c2+1]

    return ux,uy,uz

def change_topological_sector(ux,uy,uz):
    """ 
    With equal probability add one of the four non-trivial loops to the
    arg ux,uy,uz arrays: column loop, row loop, both, neither.
    """
    
    # choose type of loop
    loop = np.random.randint(4)

    # add winding loop
    if loop==0:
        uy[:,-1] = -1*uy[:,-1]
    elif loop==1:
        uz[-1,:] = -1*uz[-1,:]
    elif loop==2:
        uy[:,-1] = -1*uy[:,-1]
        uz[-1,:] = -1*uz[-1,:]
    # final case is do nothing

    return ux,uy,uz

class montecarlo(object):
    """ """

    def __init__( self,*args,**kwargs ):
        """ 
        Constructor assigns both the generate_proposal and compute_log_partition_function methods,
        which have different forms depending on some of the kwargs
        """

        # this class will need to construct new kithcmb objs, so we store the coupling
        # parameters J and K in it
        self.J = kwargs['J']
        self.K = kwargs['K']

        #
        # unpack rest of kwargs and set defaults
        #
        MC_step_type='vortex'
        if 'MC_step_type' in kwargs:
            MC_step_type = kwargs['MC_step_type']

        change_topo_sector=True
        if 'change_topo_sector' in kwargs:
            change_topo_sector = kwargs['change_topo_sector']

        project_fermions=True
        if 'project_fermions' in kwargs:
            project_fermions = kwargs['project_fermions']

        #
        # assign MCMC step function
        #
        if MC_step_type=='link':
            
            if change_topological_sector:

                # pack together with change_topological_sector
                def mcmc_step(ux,uy,uz):
                    ux,uy,uz = single_link_flip(ux,uy,uz)
                    ux,uy,uz = change_topological_sector(ux,uy,uz)
                    return ux,uy,uz
                self.generate_proposal = mcmc_step

            else:
                self.generate_proposal = single_link_flip

        elif MC_step_type=='vortex':
            
            if change_topological_sector:

            # pack together with change_topological_sector
                def mcmc_step(ux,uy,uz):
                    ux,uy,uz = add_vortex_pair(ux,uy,uz)
                    ux,uy,uz = change_topological_sector(ux,uy,uz)
                    return ux,uy,uz
                self.generate_proposal = mcmc_step

            else:
                self.generate_proposal = add_vortex_pair

        else:
            print('ERROR, montecarlo class cannot recognise input step type value.')

        # 
        # assign how to compute the log partition function, this is different in the
        # projected and unprojected cases
        #
        if project_fermions:

            def compute_log_partition_function(T,ux,uy,uz,khobj):
                """ 
                Version of compute_log_partition_function for the projected case.
                """
                # solve fully for Q matrix and spectrum 
                Sigma,Q = khobj.get_normal_form()
                spectrum =np.sort(np.sum(Sigma,axis=0))
                # get desired parity 
                desired_parity = projferms.get_desired_parity(ux=ux,uy=uy,uz=uz,Q=Q)
                # compute projected partition function
                logZ,extras = projferms.get_projected_log_Z(spectrum,T,desired_parity)
                return logZ

        else:
            
            def compute_log_partition_function(T,ux,uy,uz,khobj):
                """ 
                Version of compute_log_partition_function for the unprojected case.

                here ux,uy,uz are not needed, but they are included so this function has the
                same signature as the projected version above.
                """
                # get spectrum 
                spectrum = khobj.get_spectrum()
                # compute projected partition function
                logZ,extras = projferms.get_unprojected_log_Z(spectrum,T)           
                return logZ

        self.compute_log_partition_function = compute_log_partition_function

    def metropolis_step( self,ux,uy,uz,T,curr_log_partition_func ):
        """
        Implement a move, compute the partition function of the new sector and from there 
        decide to accept or reject the change. At temp T
        """
        new_ux,new_uy,new_uz = self.generate_proposal(ux,uy,uz)
        
        # create a new kithcmb object with the new U arrays
        new_khobj = kithcmb.kitaevhoneycomb(J=self.J,K=self.K,ux=new_ux,uy=new_uy,uz=new_uz)
        
        # get the new log partition function value
        prop_log_partition_func = self.compute_log_partition_function(T,new_ux,new_uy,new_uz,new_khobj)
        
        # calculate probability that change is accepted
        p = min( np.exp(prop_log_partition_func-curr_log_partition_func), 1. )
        
        # accept or reject the change
        if ( np.random.random()>p ):
            # reject
            return ux,uy,uz,curr_log_partition_func
        else:
            # accept
            return new_ux,new_uy,new_uz,prop_log_partition_func

    def metropolis_sweep( self,ux,uy,uz,T ):
        """
        Carry out a sweep of O(L^2) metropolis steps to generate a new vortex sector at temperature T
        """
        
        # compute the current log partition function value, need temporary kithcmb obj to compute this
        khobj = kithcmb.kitaevhoneycomb(J=self.J,K=self.K,ux=ux,uy=uy,uz=uz)
        curr_log_partition_func = self.compute_log_partition_function(T,ux,uy,uz,khobj)
        
        # do O(L^2) steps
        for m in range(ux.size):
            ux,uy,uz,curr_log_partition_func = self.metropolis_step(ux,uy,uz,T,curr_log_partition_func)

        return ux,uy,uz

