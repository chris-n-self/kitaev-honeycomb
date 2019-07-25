# 09/05/19
# Chris Self
import numpy as np
from scipy import linalg as spla
from matplotlib import pyplot as plt

def make_A(J,K,ux,uy,uz,resolved_idx_to_one_d):
    """
    """
    A = np.zeros((2*ux.size,2*ux.size))

    # nn-links
    A[resolved_idx_to_one_d[:,:,0],resolved_idx_to_one_d[:,:,1]] = J[0]*ux
    A[resolved_idx_to_one_d[:,(np.arange(ux.shape[1])+1)%ux.shape[1],0],resolved_idx_to_one_d[:,:,1]] = J[1]*uy
    A[resolved_idx_to_one_d[(np.arange(ux.shape[0])+1)%ux.shape[0],:,0],resolved_idx_to_one_d[:,:,1]] = J[2]*uz

    # nnn-links
    # bb
    A[resolved_idx_to_one_d[(np.arange(ux.shape[0])+1)%ux.shape[0],:,0],resolved_idx_to_one_d[:,:,0]] = K*ux*uz
    A[resolved_idx_to_one_d[:,:,0],resolved_idx_to_one_d[:,(np.arange(ux.shape[1])+1)%ux.shape[1],0]] = K*ux*uy
    A[resolved_idx_to_one_d[:,(np.arange(ux.shape[1])+1)%ux.shape[1],0],\
      resolved_idx_to_one_d[(np.arange(ux.shape[0])+1)%ux.shape[0],:,0]] = K*uy*uz
    # ww
    A[resolved_idx_to_one_d[:,:,1],resolved_idx_to_one_d[(np.arange(ux.shape[0])+1)%ux.shape[0],:,1]] = K*ux[(np.arange(ux.shape[0])+1)%ux.shape[0],:]*uz
    A[resolved_idx_to_one_d[:,(np.arange(ux.shape[1])+1)%ux.shape[1],1],resolved_idx_to_one_d[:,:,1]] = K*ux[:,(np.arange(ux.shape[1])+1)%ux.shape[1]]*uy
    A[resolved_idx_to_one_d[(np.arange(ux.shape[0])+1)%ux.shape[0],:,1],\
      resolved_idx_to_one_d[:,(np.arange(ux.shape[1])+1)%ux.shape[1],1]] = K*uy[(np.arange(ux.shape[0])+1)%ux.shape[0],:]*uz[:,(np.arange(ux.shape[1])+1)%ux.shape[1]]

    # symmetrise
    A = A - np.transpose(A)
    
    return 2.*A

class kitaevhoneycomb(object):

    def __init__(self,*args,**kwargs):
        """
        """

        # unpack kwargs
        self.J = tuple(kwargs['J'])
        assert len(self.J)==3
        self.K = kwargs['K']

        ux,uy,uz = np.array(kwargs['ux']),np.array(kwargs['uy']),np.array(kwargs['uz'])
        assert ux.shape==uy.shape
        assert ux.shape==uz.shape

        # tensors that convert between different indexing of the sites
        self.resolved_index_to_one_d = np.zeros((ux.shape[0],ux.shape[1],2),dtype=np.int32)
        self.resolved_index_to_one_d[:,:,0] = np.arange(ux.size).reshape((ux.shape[0],ux.shape[1])) # b sites
        self.resolved_index_to_one_d[:,:,1] = ux.size + np.arange(ux.size).reshape((ux.shape[0],ux.shape[1])) # w sites
        self.one_d_index_to_resolved = np.zeros((2*ux.size,3),dtype=np.int32)
        self.one_d_index_to_resolved[:,0] = (np.arange(2*ux.size)%ux.size)//ux.shape[1] # row index
        self.one_d_index_to_resolved[:,1] = (np.arange(2*ux.size)%ux.size)%ux.shape[1] # col index
        self.one_d_index_to_resolved[:,2] = np.arange(2*ux.size)//ux.size # bw value

        # tensor to give the real-space position of sites
        a1 = np.array([np.sqrt(3),0.])
        a2 = np.array([np.sqrt(3)/2,-3./2])
        bw_vector = np.array([np.sqrt(3)/2,-1./2])
        self.real_space_positions = np.zeros((2*ux.size,2))
        self.real_space_positions[:,0] = self.one_d_index_to_resolved[:,0]*a2[0] +\
                                    self.one_d_index_to_resolved[:,1]*a1[0] +\
                                    self.one_d_index_to_resolved[:,2]*bw_vector[0]
        self.real_space_positions[:,1] = self.one_d_index_to_resolved[:,0]*a2[1] +\
                                    self.one_d_index_to_resolved[:,1]*a1[1] +\
                                    self.one_d_index_to_resolved[:,2]*bw_vector[1]

        # compute A matrix
        self.A = make_A(self.J,self.K,ux,uy,uz,self.resolved_index_to_one_d)

        # identify vortices
        self.vortices = uz[:,:]*uy[:,:]*ux[:,(np.arange(ux.shape[1])+1)%ux.shape[1]]\
                        *uz[:,(np.arange(ux.shape[1])+1)%ux.shape[1]]\
                        *ux[(np.arange(ux.shape[0])+1)%ux.shape[0],:]\
                        *uy[(np.arange(ux.shape[0])+1)%ux.shape[0],:]

    def draw_resolved_labelling(self,display_fig=True,filename=None,figsizeboost=1.,**savefigkwargs):
        """
        draw the lattice with the resolved labelling annotated
        """
        fscale = figsizeboost*self.vortices.shape[0]/10.
        fig,ax = plt.subplots(figsize=(fscale*12,fscale*9))
        ax.set_aspect('equal')
        plt.axis('off')

        bw_vector = np.array([np.sqrt(3)/2,-1./2])
        for row,col,bw in self.one_d_index_to_resolved[self.one_d_index_to_resolved[:,2]==1]:
            ax.annotate('1',xy=tuple(self.real_space_positions[self.resolved_index_to_one_d[row,col,bw]]-np.array([0.,0.25*bw_vector[1]])),fontsize=7*figsizeboost)
            ax.annotate('0',xy=tuple(self.real_space_positions[self.resolved_index_to_one_d[row,col,bw]]-bw_vector-np.array([0.,0.25*bw_vector[1]])),fontsize=7*figsizeboost)

            pos = self.real_space_positions[self.resolved_index_to_one_d[row,col,bw]]-np.array([0.5*bw_vector[0],1.5*bw_vector[1]])
            ax.annotate("("+"{0:.3g}".format(row)+","+"{0:.3g}".format(col)+")",xy=tuple(pos),fontsize=8*figsizeboost,rotation=30)

            pos = self.real_space_positions[self.resolved_index_to_one_d[row,col,bw]]-bw_vector/2.
            ax.plot([(pos-bw_vector/2.)[0],(pos+bw_vector/2.)[0]],[(pos-bw_vector/2.)[1],(pos+bw_vector/2.)[1]],'k')

        for bw in [0,1]:
            sites = (self.one_d_index_to_resolved[:,2]==bw)
            ax.scatter(self.real_space_positions[sites,0],self.real_space_positions[sites,1],c='C'+str(bw),s=100)
            
        if not filename is None:
            plt.savefig(filename,**savefigkwargs)

        if display_fig:
            plt.show()
        else:
            plt.close()

    def draw_one_d_labelling(self,display_fig=True,filename=None,figsizeboost=1.,**savefigkwargs):
        """
        draw the lattice with the one-d site labelling annotated
        """
        fscale = figsizeboost*self.vortices.shape[0]/10.
        fig,ax = plt.subplots(figsize=(fscale*12,fscale*9))
        ax.set_aspect('equal')
        plt.axis('off')

        for bw in [0,1]:
            sites = (self.one_d_index_to_resolved[:,2]==bw)
            ax.scatter(self.real_space_positions[sites,0],self.real_space_positions[sites,1],c='C'+str(bw),s=100)
            
        for site in np.arange(2*self.vortices.size):
            pos = self.real_space_positions[site]
            ax.annotate(str(site),xy=pos,fontsize=8*figsizeboost)
            
        if not filename is None:
            plt.savefig(filename,**savefigkwargs)

        if display_fig:
            plt.show()
        else:
            plt.close()

    def draw_system(self,draw_sites=True,display_fig=True,defer_display=False,filename=None,figsizeboost=1.,**savefigkwargs):
        """
        draw the complete system with all sites, vortices and all A-links indicated.
        (this will be slow for large systems.)

        the A-link couplings have their direction drawn and are coloured based on whether
        they are flipped relative to the 'regular' definition of the no-vortex sector.

        this function is slow because of the loop over all lower-triangular elements of A,
        I have written it this way so that this function can be used as a visual tests of
        whether the A matrix is being computed correctly or not. however, a quicker version
        could be written that only visits the A-links we expect to have non-zero values.
        """
        fscale = figsizeboost*self.vortices.shape[0]/10.
        fig,ax = plt.subplots(figsize=(fscale*12,fscale*9))
        ax.set_aspect('equal')
        plt.axis('off')

        # draw sites
        if draw_sites:
            for bw in [0,1]:
                sites = (self.one_d_index_to_resolved[:,2]==bw)
                ax.scatter(self.real_space_positions[sites,0],self.real_space_positions[sites,1],c='C'+str(bw),s=50)
         
        #   
        # add couplings
        #

        # identify the elements of A that need to be drawn
        links_to_draw = np.argwhere(np.logical_and(self.A>0,np.logical_not(np.isclose(self.A,0.))))

        a1 = np.array([np.sqrt(3),0.])
        a2 = np.array([np.sqrt(3)/2,-3./2])
        bw_vector = np.array([np.sqrt(3)/2,-1./2])

        # canonical version of the no-vortex sector to base colouring against
        regular_A = make_A(self.J,self.K,np.ones(self.vortices.shape),np.ones(self.vortices.shape),np.ones(self.vortices.shape),self.resolved_index_to_one_d)

        for site_a,site_b in links_to_draw:
            pos_a = self.real_space_positions[site_a]
            pos_b = self.real_space_positions[site_b]

            # if the links are flipped relative to no-vortex sector colour them red
            colour = 'k'
            if not np.isclose(self.A[site_a,site_b],regular_A[site_a,site_b]):
                colour = 'r'
                
            # handle boundary links
            # ---
            # also draws invisible point at edge to ensure arrows hanging off the sides
            # are not culled
            invisible_point_size = 20
            if np.abs(self.one_d_index_to_resolved[site_a][0]-self.one_d_index_to_resolved[site_b][0])==(self.vortices.shape[0]-1) and\
               np.abs(self.one_d_index_to_resolved[site_a][1]-self.one_d_index_to_resolved[site_b][1])==(self.vortices.shape[1]-1):
                # corner wrap around draw correction
                #colour = 'b'
                if self.one_d_index_to_resolved[site_a][1]>self.one_d_index_to_resolved[site_b][1]:
                    pos_a = pos_a+self.vortices.shape[0]*a2
                    pos_a = pos_a-self.vortices.shape[1]*a1
                    ax.scatter(pos_a[0],pos_a[1],c='w',s=invisible_point_size)
                else:
                    pos_b = pos_b+self.vortices.shape[0]*a2
                    pos_b = pos_b-self.vortices.shape[1]*a1
                    ax.scatter(pos_b[0],pos_b[1],c='w',s=invisible_point_size)
                    
            else:
                # left-right (col) wrap around draw correction
                if np.abs(self.one_d_index_to_resolved[site_a][1]-self.one_d_index_to_resolved[site_b][1])==(self.vortices.shape[1]-1):
                    #colour = 'r'
                    if self.one_d_index_to_resolved[site_a][1]>self.one_d_index_to_resolved[site_b][1]:
                        pos_b = pos_b+self.vortices.shape[1]*a1
                        ax.scatter(pos_b[0],pos_b[1],c='w',s=invisible_point_size)
                    else:
                        pos_a = pos_a+self.vortices.shape[1]*a1
                        ax.scatter(pos_a[0],pos_a[1],c='w',s=invisible_point_size)
                # up-down (row) wrap around draw correction
                if np.abs(self.one_d_index_to_resolved[site_a][0]-self.one_d_index_to_resolved[site_b][0])==(self.vortices.shape[0]-1):
                    #colour = 'g'
                    if self.one_d_index_to_resolved[site_a][0]>self.one_d_index_to_resolved[site_b][0]:
                        pos_b = pos_b+self.vortices.shape[0]*a2
                        ax.scatter(pos_b[0],pos_b[1],c='w',s=invisible_point_size)
                    else:
                        pos_a = pos_a+self.vortices.shape[0]*a2
                        ax.scatter(pos_a[0],pos_a[1],c='w',s=invisible_point_size)
                        
            # draw A-links as arrows
            alpha = np.abs(self.A[site_a,site_b])/np.max(self.A)
            midpoint = pos_a+0.6*(pos_b-pos_a)
            ax.annotate("",xytext=tuple(pos_a),xy=tuple(midpoint),\
                        arrowprops={'arrowstyle':'->, head_width=0.5, head_length=1',\
                                    'linewidth':0.9,\
                                    'alpha':alpha,\
                                    'color':colour})
            midpoint = pos_a+0.5*(pos_b-pos_a)
            plt.plot([midpoint[0],pos_b[0]],[midpoint[1],pos_b[1]],color=colour,alpha=alpha,linewidth=0.9)
                        
        # draw vortices
        ax.scatter(self.real_space_positions[self.resolved_index_to_one_d[self.vortices==-1,1]][:,0]+bw_vector[0],\
                   self.real_space_positions[self.resolved_index_to_one_d[self.vortices==-1,1]][:,1]+bw_vector[1],\
                   c='k',s=100)
              
        if not filename is None:
            plt.savefig(filename,**savefigkwargs)

        if defer_display:
            return fig,ax
        else:
            if display_fig:
                plt.show()
            else:
                plt.close()

    def get_spectrum(self):
        """ """
        return spla.eigvalsh(1j*self.A)

    def get_normal_form(self):
        """
        """

        # add small random noise to lift degeneracies enough to be able
        # to properly order the columns of Q
        epsilon = 1E-8
        R = 2.*(np.random.rand(*self.A.shape) - 0.5)
        R = epsilon*(R + np.transpose(R))

        # get the block-diagonal fully-real Schur normal form
        diagonalised_A = self.A
        diagonalised_A = diagonalised_A + np.sign(diagonalised_A)*R
        Sigma,Q=spla.schur(diagonalised_A, output='real', lwork=None, overwrite_a=True, sort=None, check_finite=True)

        # Sigma and Q returned above are not in the form we want them: which is:
        #
        # \Sigma = [[   0,   \l_1,    0,     0,   . . . ]
        #           [ -\l_1,   0,     0,     0,   . . . ]
        #           [   0,     0,     0,   \l_2,  . . . ]
        #           [   0,     0,   -\l_2,   0,   . . . ]
        #           [   .      .      .      .    .     ]
        #           [   .      .      .      .      .   ]
        #           [   .      .      .      .        . ]]
        #
        # where \l_1 > \l_2 > ... > 0 and the related ordering of Q.
        # instead, the matrices returned sometimes have the signs the other way around, and
        # the signs are not consistent between blocks. additionally, zero modes are not paired
        # into a block and appear as randomly placed single columns. 
        # the following fixes both of these problems first by finding the permutation that puts
        # all of the values contained in Sigma in decreasing order [ -\l_1, -\l_2, ... , \l_2, \l_1 ], 
        # then by 'folding' this permutation back on itself about the middle so that the column of Q
        # associated with -\l_1 becomes the first column, with \l_1 becomes the second column, ...
        # 
        # flatten evals of A
        evals = np.sum(Sigma,axis=0)
        # get the -ve to +ve ordering
        permutation = np.argsort(evals)
        # fold permutation back onto itself to get correct 
        # ordering for columns of Q matrix
        q_permutation = np.zeros(self.A.shape[0],dtype='i')
        q_permutation[::2] = permutation[:self.A.shape[0]//2]
        q_permutation[1::2] = permutation[-1:-self.A.shape[0]//2-1:-1]
        Q = Q[:,q_permutation]
        Sigma = Sigma[:,:]
        Sigma = Sigma[:,q_permutation]
        Sigma = Sigma[q_permutation,:]

        # tests
        assert np.isclose(np.abs(spla.det(Q)),1.) # |detQ|=1
        assert np.all(np.isclose(np.dot(Q,np.transpose(Q)),np.diag(np.ones(Q.shape[0])))) # Q orthogonal
        assert np.all(np.isclose(np.dot(Q,np.dot(Sigma,np.transpose(Q))),self.A)) # Q \Sigma Q^T = A
        assert np.isclose(np.sum(Sigma),0.) # sum_ij \Sigma_ij = 0

        return Sigma,Q

    def get_diagonal_form(self,Sigma=None,Q=None):
        """
        returns: diagonal matrix D with the eigenvalues ordered from most negative to most 
        positive (the spectrum is given by the diagonal elements of iD), eigenvector matrix
        U with the columns ordered in the corresponding order to D
        """

        if (Sigma is None) or (Q is None):
            Sigma,Q = self.get_normal_form()

        # construct W
        sy = 1./np.sqrt(2) * np.array([[1,1],[1j,-1j]])
        W = np.kron( np.identity(self.A.shape[0]//2),sy )

        # rotate Sigma to get D, and Q to get U
        D = np.dot(np.dot(np.matrix(W).getH(),Sigma),W)
        U = np.dot(Q,W)

        # use argsort to correctly order the spectrum, and use that to permute
        # the columns of U
        permutation = np.argsort(1j*np.diag(D))
        U = U[:,permutation]
        # sort D
        D = D[permutation,:][:,permutation]

        # tests
        assert np.isclose(np.abs(spla.det(U)),1.) # |detU|=1
        assert np.all(np.isclose(np.dot(U,np.conj(np.transpose(U))),np.diag(np.ones(U.shape[0])))) # U unitary
        assert np.all(np.isclose(np.dot(U,np.dot(D,np.conj(np.transpose(U)))),self.A)) # U D U^\dagger = A
        assert np.isclose(np.sum(D),0.) # \sum_ij D_ij = 0
        assert np.isclose(np.trace(D),0.) # tr(D) = 0

        return D,U
