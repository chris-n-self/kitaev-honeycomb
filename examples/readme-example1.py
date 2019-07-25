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