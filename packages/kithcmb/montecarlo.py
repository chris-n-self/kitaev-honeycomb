# 09/07/19
# Chris Self

import numpy as np
import kitaevhoneycomb as kithcmb
import parityprojectedfermions as projferms

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
    row1 = np.random.randint(ux.shape[0])
    col1 = np.random.randint(ux.shape[1])
    row2 = np.random.randint(ux.shape[0])
    col2 = np.random.randint(ux.shape[1])

    # implement flips. flip lines meet at point [row1,col2]
    # (this is robust to cases where e.g. row1==row2)
    uy[min(row1,row2):max(row1,row2),col2] = -1*uy[min(row1,row2):max(row1,row2),col2]
    uz[row1,min(col1,col2):max(col1,col2)] = -1*uz[row1,min(col1,col2):max(col1,col2)]

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

class montacarlo(object):
    """ """

    def __init__( self,J=J,K=K,MC_step_type='vortex',change_topo_sector=True,project_fermions=True ):
        """ 
        Constructor assigns both the generate_proposal and compute_log_partition_function methods,
        which have different forms depending on some of the kwargs
        """

        # this class will need to construct new kithcmb objs, so we store the coupling
        # parameters J and K in it
        self.J = J
        self.K = K

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
            print 'ERROR, montecarlo class cannot recognise input step type value.'

        # 
        # assign how to compute the log partition function, this is different in the
        # projected and unprojected cases
        #
        if project_fermions:

            def compute_log_partition_function(beta,ux,uy,uz,khobj):
                """ 
                Version of compute_log_partition_function for the projected case.
                """
                # solve fully for Q matrix and spectrum 
                Sigma,Q = khobj.get_normal_form()
                spectrum =np.sort(np.sum(Sigma,axis=0))
                # get desired parity 
                desired_parity = projferms.get_desired_parity(ux,uy,uz,Q)
                # compute projected partition function
                return projferms.get_projected_log_Z(spectrum,beta,desired_parity)

        else:
            
            def compute_log_partition_function(beta,ux,uy,uz,khobj):
                """ 
                Version of compute_log_partition_function for the unprojected case.

                here ux,uy,uz are not needed, but they are included so this function has the
                same signature as the projected version above.
                """
                # get spectrum 
                spectrum = khobj.get_spectrum()
                # compute projected partition function
                return projferms.get_unprojected_log_Z(spectrum,beta)           

        self.compute_log_partition_function = compute_log_partition_function

    def metropolis_step( self,ux,uy,uz,beta,curr_log_partition_func ):
        """
        Implement a move, compute the partition function of the new sector and from there 
        decide to accept or reject the change. At temp T=1/beta
        """
        new_ux,new_uy,new_uz = self.generate_proposal(ux,uy,uz)
        
        # create a new kithcmb object with the new U arrays
        new_khobj = kithcmb.kitaevhoneycomb(J=self.J,K=self.K,ux=new_ux,uy=new_uy,uz=new_uz)
        
        # get the new log partition function value
        prop_log_partition_function = self.compute_log_partition_function(beta,new_ux,new_uy,new_uz,new_khobj)
        
        # calculate probability that change is accepted
        p = min( np.exp(prop_log_partition_function-curr_log_partition_function), 1. )
        
        # accept or reject the change
        if ( np.random.random()>p ):
            # reject
            return ux,uy,uz,curr_log_partition_func
        else:
            # accept
            return new_ux,new_uy,new_uz,prop_log_partition_function

    def metropolis_sweep( self,ux,uy,uz,beta ):
        """
        Carry out a sweep of O(L^2) metropolis steps to generate a new vortex sector at temperature T=1/beta
        """
        
        # compute the current log partition function value, need temporary kithcmb obj to compute this
        khobj = kithcmb.kitaevhoneycomb(J=self.J,K=self.K,ux=ux,uy=uy,uz=uz)
        curr_log_partition_func = self.compute_log_partition_function(beta,ux,uy,uz,khobj)
        
        # do O(L^2) steps
        for m in range(ux.size):
            ux,uy,uz,curr_log_partition_func = self.metropolis_step(ux,uy,uz,beta,curr_log_partition_func)

        return ux,uy,uz

