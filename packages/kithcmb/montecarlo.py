# 09/07/19
# Chris Self

import numpy as np
import kitaevhoneycomb as kithcmb
import parityprojectedfermions as projferms

def single_link_flip(Ux,Uy,Uz):
    """ 
    Generate a new set of U arrays where one link has been flipped from
    the arg arrays Ux,Uy,Uz
    """
    
    # choose a random u-link
    row = np.random.randint(Ux.shape[0])
    col = np.random.randint(Ux.shape[1])
    l = np.random.randint(3)

    # flip the link
    if l==0:
        Ux[row,col] = -1*Ux[row,col]
    elif l==1:
        Uy[row,col] = -1*Uy[row,col]
    elif l==2:
        Uz[row,col] = -1*Uz[row,col]
        
    return Ux,Uy,Uz

def add_vortex_pair(Ux,Uy,Uz):
    """ 
    Generate a new set of U arrays where a vortex pair has been added
    relative to the arg arrays Ux,Uy,Uz. (This doesn't necessarily actually
    add a pair of vortices, depending on the previous state of the system.
    It could instead move vortices or annihilate a pair.)
    """

    # random positions for vortices
    row1 = np.random.randint(Ux.shape[0])
    col1 = np.random.randint(Ux.shape[1])
    row2 = np.random.randint(Ux.shape[0])
    col2 = np.random.randint(Ux.shape[1])

    # implement flips. flip lines meet at point [row1,col2]
    # (this is robust to cases where e.g. row1==row2)
    Uy[min(row1,row2):max(row1,row2),col2] = -1*Uy[min(row1,row2):max(row1,row2),col2]
    Uz[row1,min(col1,col2):max(col1,col2)] = -1*Uz[row1,min(col1,col2):max(col1,col2)]

    return Ux,Uy,Uz

def change_topological_sector(Ux,Uy,Uz):
    """ 
    With equal probability add one of the four non-trivial loops to the
    arg Ux,Uy,Uz arrays: column loop, row loop, both, neither.
    """
    
    # choose type of loop
    loop = np.random.randint(4)

    # add winding loop
    if loop==0:
        Uy[:,-1] = -1*Uy[:,-1]
    elif loop==1:
        Uz[-1,:] = -1*Uz[-1,:]
    elif loop==2:
        Uy[:,-1] = -1*Uy[:,-1]
        Uz[-1,:] = -1*Uz[-1,:]
    # final case is do nothing

    return Ux,Uy,Uz

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
                def mcmc_step(Ux,Uy,Uz):
                    Ux,Uy,Uz = single_link_flip(Ux,Uy,Uz)
                    Ux,Uy,Uz = change_topological_sector(Ux,Uy,Uz)
                    return Ux,Uy,Uz
                self.generate_proposal = mcmc_step

            else:
                self.generate_proposal = single_link_flip

        elif MC_step_type=='vortex':
            
            if change_topological_sector:

            # pack together with change_topological_sector
                def mcmc_step(Ux,Uy,Uz):
                    Ux,Uy,Uz = add_vortex_pair(Ux,Uy,Uz)
                    Ux,Uy,Uz = change_topological_sector(Ux,Uy,Uz)
                    return Ux,Uy,Uz
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

            def compute_log_partition_function(beta,Ux,Uy,Uz,khobj):
                """ 
                Version of compute_log_partition_function for the projected case.
                """
                # solve fully for Q matrix and spectrum 
                Sigma,Q = khobj.get_normal_form()
                spectrum =np.sort(np.sum(Sigma,axis=0))
                # get desired parity 
                desired_parity = projferms.get_desired_parity(Ux,Uy,Uz,Q)
                # compute projected partition function
                return projferms.get_projected_log_Z(spectrum,beta,desired_parity)

        else:
            
            def compute_log_partition_function(beta,Ux,Uy,Uz,khobj):
                """ 
                Version of compute_log_partition_function for the unprojected case.

                here Ux,Uy,Uz are not needed, but they are included so this function has the
                same signature as the projected version above.
                """
                # get spectrum 
                spectrum = khobj.get_spectrum()
                # compute projected partition function
                return projferms.get_unprojected_log_Z(spectrum,beta)           

        self.compute_log_partition_function = compute_log_partition_function

    def metropolis_step( self,Ux,Uy,Uz,beta,curr_log_partition_func ):
        """
        Implement a move, compute the partition function of the new sector and from there 
        decide to accept or reject the change. At temp T=1/beta
        """
        new_Ux,new_Uy,new_Uz = self.generate_proposal(Ux,Uy,Uz)
        
        # create a new kithcmb object with the new U arrays
        new_khobj = kithcmb.kitaevhoneycomb(J=self.J,K=self.K,Ux=new_Ux,Uy=new_Uy,Uz=new_Uz)
        
        # get the new log partition function value
        prop_log_partition_function = self.compute_log_partition_function(beta,new_Ux,new_Uy,new_Uz,new_khobj)
        
        # calculate probability that change is accepted
        p = min( np.exp(prop_log_partition_function-curr_log_partition_function), 1. )
        
        # accept or reject the change
        if ( np.random.random()>p ):
            # reject
            return Ux,Uy,Uz,curr_log_partition_func
        else:
            # accept
            return new_Ux,new_Uy,new_Uz,prop_log_partition_function

    def metropolis_sweep( self,Ux,Uy,Uz,beta ):
        """
        Carry out a sweep of O(L^2) metropolis steps to generate a new vortex sector at temperature T=1/beta
        """
        
        # compute the current log partition function value, need temporary kithcmb obj to compute this
        khobj = kithcmb.kitaevhoneycomb(J=self.J,K=self.K,Ux=Ux,Uy=Uy,Uz=Uz)
        curr_log_partition_func = self.compute_log_partition_function(beta,Ux,Uy,Uz,khobj)
        
        # do O(L^2) steps
        for m in range(Ux.size):
            Ux,Uy,Uz,curr_log_partition_func = self.metropolis_step(Ux,Uy,Uz,beta,curr_log_partition_func)

        return Ux,Uy,Uz

