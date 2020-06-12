#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
import scipy.optimize as optim

#Simulation parameters:
L = 15.        #Simulation box size in reduced LJ units
N = 100        #Number of particles in box
T = 1.0        #Temperature in reduced LJ units
nSteps = 5000  #Total number of time steps in simulation

#Data collection parameters:
collectInterval = 10  #Output velocities and print energies every so-many steps
outputIntervalV = 1000 #Output velocity files every so-many steps (of data collected at collectInterval)
outputIntervalD = 1000 #Output mean-squared displacement vs time based on blocks of so-many steps


class MD:
    """
    Class that performs simple MD with LJ potentials.
    Note that the class itself supports much more general functionality,
    including simulation in 3 or higher dimensions, than what we use for this course.
    """
    
    def __init__(self, L):
        """
        Initialize simulation for d-dimensional box of length L (LJ units).
        Note d is set by the pos passed in various functions below.
        """
        self.L = L
    
    def pairPotential(self, pos):
        """
        Return potential energy and force on all atoms (Nxd array) given positions, pos (Nxd array).
        Potential is LJ with epsilon = sigma = 1 with minimum-image convention (effective cutoff ~ L/2).
        """
        x = (1./self.L)*(pos[None,...] - pos[:,None,:]) #NxNxd array of fractional relative positions
        x = self.L*(x-np.floor(0.5+x)) #apply minimum-image convention and convert back to Cartesian
        rInv2 = 1./squareform(np.sum(x**2, axis=-1)) #pairwise inverse squared distance
        rInv6 = rInv2**3 #squareform above makes these N*(N-1)/2 vector instead of NxN
        rInv12 = rInv6**2
        pe = 4.*np.sum(rInv12 - rInv6) #total potential energy (scalar)
        forces = 24.*np.sum(squareform((rInv6 -2.*rInv12)*rInv2)[...,None] * x, axis=1) #forces on each particle (Nxd)
        return pe, forces  #second squareform above restores to NxN
    
    def minimize(self, pos):
        """Minimize potential energy starting from pos."""
        def residual(x, md, d):
            pe, forces = md.pairPotential(x.reshape(-1,d))
            return pe, -forces.flatten() #Note gradient(pe) = -forces
        res = optim.minimize(residual, pos.flatten(), args=(self,pos.shape[1]), jac=True, options={'disp':True})
        peOpt = res.fun #optimized potential energy
        posOpt = res.x.reshape(pos.shape)
        return peOpt, posOpt
    
    """Create initial velocity distribution at specified temperature (LJ units)."""
    def initialVelocities(self, pos, T):
        v = np.random.standard_normal(pos.shape) * np.sqrt(T) #kB = m = 1
        v -= np.mean(v, axis=0)[None,:] #remove c.o.m velocity
        return v
    
    def run(self, pos, vel, dt, nSteps, collectInterval, T0=None, tDampT=1.):
        """
        Run molecular dynamics simulation:
        NVE simulation by default.
        Set T0 and tDampT for NVT with Berendsen thermostat.
        Return positions and velocities collected every collectInterval steps.
        """
        posAll = []
        velAll = []
        #Compute initial energy and forces:
        pe, forces = self.pairPotential(pos)
        ke = 0.5*np.sum(vel**2)
        print(' {:5s} {:8s} {:8s} {:8s} {:5s}'.format('Step', 'PotEng', 'KinEng', 'TotEng', 'Temp'))
        for iStep in range(nSteps+1):
            #Data collection and reporting:
            if(iStep % collectInterval == 0):
                posAll.append(pos.copy())
                velAll.append(vel.copy())
                print('{:5d} {:8.3f} {:8.3f} {:8.3f} {:5.3f}'.format(iStep, pe, ke, pe+ke, 2.*ke/pos.size))
            #Velocity verlet step:
            vel += (0.5*dt)*forces #velocity update first half
            pos += dt*vel #position update (note m=1)
            pe, forces = self.pairPotential(pos) #force update
            vel += (0.5*dt)*forces #velocity update second half
            ke = 0.5*np.sum(vel**2)
            #Thermostat:
            if(T0):
                keScale = 1. + (0.5*T0*pos.size/ke - 1.) * dt/tDampT
                vel *= np.sqrt(keScale)
                ke *= keScale
        return np.array(posAll), np.array(velAll)


#Test optimization of some particles in a 2D box
md = MD(L)
np.random.seed(123) #to make it reproducible below
pos0 = np.random.rand(N, 2)*L
print('Initial PE:', md.pairPotential(pos0)[0])

peOpt, posOpt = md.minimize(pos0)
print('Optimized PE:', peOpt)

#Run MD simulation starting from above:

#Make sure nSteps, output(V), output(D) and collect intervals are sequential multiples:
outputIntervalV = (outputIntervalV//collectInterval)*collectInterval
outputIntervalD = (outputIntervalD//outputIntervalV)*outputIntervalV
nSteps = (nSteps//outputIntervalD)*outputIntervalD

vel = md.initialVelocities(posOpt, T)
posAll, velAll = md.run(posOpt.copy(), vel, 0.01, nSteps, collectInterval, T0=T, tDampT=0.1) #, T0=None

#Save velocities (grouped by outputInterval) to text file:
velGrouped = velAll[1:].reshape(nSteps//outputIntervalV, (outputIntervalV//collectInterval)*posOpt.shape[0], posOpt.shape[1])
for iOut,vel in enumerate(velGrouped):
    fname = 'v-T{:.1f}-{:04d}.dat'.format(T, (iOut+1)*outputIntervalV)
    np.savetxt(fname, vel)

#Compute and save mean-squared displacements:
outDstride = outputIntervalD//collectInterval #number of data points in each block
tOutD = np.arange(outDstride+1)*collectInterval #time steps for MSD data
msdArr = []
for iOut in range(nSteps//outputIntervalD):
	dpos = posAll[outDstride*iOut:outDstride*(iOut+1)+1] - posAll[outDstride*iOut,None]
	msd = np.mean(np.sum(dpos**2, axis=-1), axis=-1)
	msdArr.append(msd)
	fnameD = 'msd-T{:.1f}-{:04d}.dat'.format(T, (iOut+1)*outputIntervalD)
	np.savetxt(fnameD, np.array([tOutD, msd]).T)
msdArr = np.array(msdArr).T

#Plot configurations:
fig, ax = plt.subplots(2,4, figsize=(15,8))
plt.subplots_adjust(hspace=0.3)
labels = ['Initial positions', 'Optimized positions', 'Final MD positions']
alphaArr = [ 1., 1., 1./len(posAll) ]
for iData, data in enumerate([pos0, posOpt, posAll.reshape(-1,posAll.shape[-1])]):
    plt.sca(ax[0,iData])
    plt.scatter(data[:,0]%L, data[:,1]%L, color='b', alpha=alphaArr[iData])
    plt.xlim(0, L)
    plt.ylim(0, L)
    plt.title(labels[iData])

#Plot velocity distributions:
#fig, ax = plt.subplots(1,3, figsize=(10,3))
labels = [
	'Velocity distribution\nof final configuration',
	'Velocity distribution\nof group of steps in '+fname,
	'Velocity distribution\nof entire trajectory' ]
for iData, data in enumerate([velAll[-1], velGrouped[-1], velAll]):
    plt.sca(ax[1,iData])
    plt.hist(data.flatten(), bins=50, color='r')
    plt.title(labels[iData])

#Plot MSD:
plt.sca(ax[0,3])
plt.plot(tOutD, msdArr, 'g')
plt.title('Mean-squared displacement (MSD)')
plt.xlabel(r'$\Delta t$ [steps]')
plt.xlim(0, tOutD[-1])

#Plot MSD distribution:
plt.sca(ax[1,3])
msdMean = np.mean(msdArr, axis=1)
msdStd = np.std(msdArr, axis=1)
for nSigma in np.arange(3.,0.1,-0.1):
	gaussVal = np.exp(-0.5*nSigma**2)
	color = np.ones(3)*(1-gaussVal) + np.array([0,0.5,0])*gaussVal
	plt.fill_between(tOutD, msdMean-nSigma*msdStd, msdMean+nSigma*msdStd, color=color)
plt.title('MSD distribution')
plt.xlabel(r'$\Delta t$ [steps]')
plt.xlim(0, tOutD[-1])

plt.savefig('MD-{:.1f}.png'.format(T), bbox_inches='tight')
plt.show()
