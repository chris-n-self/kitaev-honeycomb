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