"""

SGS_RayleighBenard_EVP

Usage:
    SGS_RayleighBenard.py

Options:
"""

from docopt import docopt
import time
from configparser import ConfigParser
from pathlib import Path
import numpy as np
import sys
import h5py
import eigentools as eigentools
import dedalus.public as de
from mpi4py import MPI
from scipy import optimize
import matplotlib.pyplot as plt
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)

args = docopt(__doc__)

def RayleighBernardSolve(kx,ky,Nz,Ra,Pr,Pm,guess,mpi_comm,mode):
    sparse = True

    z_basis = de.Chebyshev('z', Nz, interval=(0,1), dealias=3/2*Nz)
    domain = de.Domain([z_basis], grid_dtype=np.complex128, comm=mpi_comm)

    # 3D MRI
    #problem_variables = ['p','vx','vy','vz','b','vxz','vyz','vzz','vxzz','vyzz','vzzz','vxzzz','vyzzz','vzzzz','vxzzzz','vyzzzz','vzzzzz','bz']
    problem_variables = ['p','vx','vy','vz','b','vxz','vyz','vzz','bz']
    problem = de.EVP(domain, variables=problem_variables, eigenvalue='om',ncc_cutoff=1e-12)

    # Local parameters
    problem.parameters['kx'] = kx
    problem.parameters['ky'] = ky
    problem.parameters['R'] = 1/np.sqrt(Ra/Pr)
    problem.parameters['RR'] = (1/np.sqrt(Ra/Pr))**12
    print((1/np.sqrt(Ra/Pr))**12)
    problem.parameters['P'] = 1/np.sqrt(Ra*Pr)
        
    # Operator substitutions for derivatives
    problem.substitutions['dx(A)'] = "1j*kx*A"
    problem.substitutions['dy(A)'] = "1j*ky*A"
    problem.substitutions['dt(A)'] = "om*A"
        
    # non ideal
    problem.substitutions['L(A)'] = "dx(dx(A)) + dy(dy(A))"    
    
    #First order formulation
    problem.add_equation("vxz - dz(vx) = 0")
    problem.add_equation("vyz - dz(vy) = 0")
    problem.add_equation("vzz - dz(vz) = 0")

    #problem.add_equation("vxzz - dz(vxz) = 0")
    #problem.add_equation("vyzz - dz(vyz) = 0")
    #problem.add_equation("vzzz - dz(vzz) = 0")

    #problem.add_equation("vxzzz - dz(vxzz) = 0")
    #problem.add_equation("vyzzz - dz(vyzz) = 0")
    #problem.add_equation("vzzzz - dz(vzzz) = 0")

    #problem.add_equation("vxzzzz - dz(vxzzz) = 0")
    #problem.add_equation("vyzzzz - dz(vyzzz) = 0")
    #problem.add_equation("vzzzzz - dz(vzzzz) = 0")

    problem.add_equation("bz - dz(b) = 0")
    
    #Continuity
    problem.add_equation("dx(vx) + dy(vy) + vzz = 0")
    
    #Momentum
    #problem.add_equation("dt(vx) + dx(p) - R*L(vx) - R*dz(vxz) + RR*L(L(vx)) = 0") #+ RR*vxzzzz = 0")
    #problem.add_equation("dt(vy) + dy(p) - R*L(vy) - R*dz(vyz) + RR*L(L(vy)) = 0") #+ RR*vyzzzz = 0")
    #problem.add_equation("dt(vz) + dz(p) - b - R*L(vz) - R*dz(vzz) + RR*L(L(vz)) = 0") #+ RR*vzzzzz = 0")

    problem.add_equation("dt(vx) + dx(p) - R*dz(vxz) + RR*L(L(L(L(L(L(vx)))))) = 0") #+ RR*vxzzzz = 0")
    problem.add_equation("dt(vy) + dy(p) - R*dz(vyz) + RR*L(L(L(L(L(L(vy)))))) = 0") #+ RR*vyzzzz = 0")
    problem.add_equation("dt(vz) + dz(p) - b - R*dz(vzz) + RR*L(L(L(L(L(L(vz)))))) = 0") #+ RR*vzzzzz = 0")

    #Bouyancy
    problem.add_equation("dt(b) - P*L(b) - P*dz(bz) - vz = 0")

    problem.add_bc("left(vx) = 0")
    problem.add_bc("right(vx) = 0")

    #problem.add_bc("right(vxzzz) = 0")
    #problem.add_bc("left(vxzzzz) = 0")
    #problem.add_bc("right(vxzzzz) = 0")

    problem.add_bc("left(vy) = 0")
    problem.add_bc("right(vy) = 0")

    #problem.add_bc("right(vyzzz) = 0")
    #problem.add_bc("left(vyzzzz) = 0")
    #problem.add_bc("right(vyzzzz) = 0")

    problem.add_bc("left(vz) = 0")
    problem.add_bc("right(vz) = 0")

    #problem.add_bc("right(vzzzz) = 0")
    #problem.add_bc("left(vzzzzz) = 0")
    #problem.add_bc("right(vzzzzz) = 0")

    problem.add_bc("left(b) = 0")
    problem.add_bc("right(b) = 0")

    # GO
    EP = eigentools.Eigenproblem(problem)
    gr, idx, freq = EP.growth_rate(sparse=sparse,N=32,target=guess)

    vals = np.zeros(3,dtype=np.float64)
    vals[0] = gr
    if (freq is None):
        freq = 0e0
    vals[1] = freq
    vals[2] = float(idx)
    if (mode):
        scale = 1 #int(Nxmode/Nx)
        mode = EP.eigenmode(idx,scales=scale)
    else:
        mode = 0e0
    return vals, mode

nkx = 100
nRa = 20
Nz = 96

kxpts = np.linspace(0.0,10.0,nkx)+1e-2
Rapts = 10e0**(np.linspace(3e0,3.5e0,nRa))

grate = np.zeros((nkx,nRa),dtype=np.float64)
gfreq = np.zeros((nkx,nRa),dtype=np.float64)
modeset = np.zeros((nkx,nRa,Nz,5),dtype=np.complex128)

nsolves = nkx*nRa
t1 = time.time()
qq=0
Pr = 1e0
Pm = 1e0
ky = 0e0
guess = 0e0
kk = CW.rank
for jj in range(nkx):
    if (np.real(guess)>=0e0):
        try:
            vmid, modemid = RayleighBernardSolve(kxpts[jj],ky,Nz,Rapts[kk],Pr,Pm,guess,MPI.COMM_SELF,True)
        except RuntimeError:
            guess = 0e0
            vmid, modemid = RayleighBernardSolve(kxpts[jj],ky,Nz,Rapts[kk],Pr,Pm,guess,MPI.COMM_SELF,True)
    else:
        guess = 0e0
        vmid, modemid = RayleighBernardSolve(kxpts[jj],ky,Nz,Rapts[kk],Pr,Pm,guess,MPI.COMM_SELF,True)
                    
    guess = vmid[0]+1j*vmid[1]
    grate[jj,kk] = vmid[0]
    gfreq[jj,kk] = vmid[1]

    modes = np.zeros((Nz,5),dtype=np.complex128)
    qq = 0
    for field in ['p','vx','vy','vz','b']:
        modemid[field].require_grid_space()
        modes[:,qq] = modemid[field].data
        qq = qq+1

    modeset[jj,kk,:,:] = modes

CW.barrier()

snd_buf = np.zeros((2,nkx,nRa),dtype=np.float64)
snd_buf[0,:,:] = grate
snd_buf[1,:,:] = gfreq
rcv_buf = None

if (CW.rank == 0):
    rcv_buf = np.zeros((2,nkx,nRa),dtype=np.float64)

CW.Reduce([snd_buf,MPI.DOUBLE],[rcv_buf, MPI.DOUBLE],op=MPI.SUM,root=0)

CW.barrier()

if (CW.rank ==0):
    #tmp = np.sum(rcv_buf,axis=0)
    tmp = rcv_buf
    grate = tmp[0,:,:]
    gfreq = tmp[1,:,:]

    np.save('grate',grate)
    np.save('gfreq',gfreq)

CW.barrier()

snd_buf = modeset
rcv_buf = None

if (CW.rank == 0):
    rcv_buf = np.zeros((nkx,nRa,Nz,5),dtype=np.complex128)

CW.Reduce([snd_buf,MPI.COMPLEX32],[rcv_buf, MPI.COMPLEX32],op=MPI.SUM,root=0)

CW.barrier()

if (CW.rank ==0):
    modeset = rcv_buf
    np.save('modeset',modeset)

print("modeset done.")
CW.barrier()
