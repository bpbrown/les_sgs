"""
Dedalus script for incompressible thermally forced convection,
to test SGS methods for LES.

Usage:
    rayleigh_benard.py [options]

Options:
    --nx=<nx>         Fourier horizontal resolution of domain, default is aspect*nz
    --nz=<nz>         Chebyshev vertical resolution of domain [default: 256]
    --aspect=<ap>     Aspect ratio of domain [default: 4]

    --Rayleigh=<RA>   Rayleigh number of the convection [default: 1e6]
    --Prandtl=<Pr>    Prandtl number [default: 1]

    --mesh=<mesh>     Processor mesh, takes format n1,n2 for a 2-d mesh decomposition of n1xn2 cores

    --GQL              Use Generalized Quasi Linear approximation
    --Lambda=<Lambda>  Cutoff for low mode projection

    --SBDF4           Use SBDF4

    --label=<label>   Case name label
"""

import numpy as np
import sys
import os
from mpi4py import MPI

import time

from fractions import Fraction

from docopt import docopt
args = docopt(__doc__)

# Find MPI rank
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

aspect = float(args['--aspect'])
nz = int(args['--nz'])
if args['--nx']:
    nx = ny = int(args['--nx'])
else:
    nx = ny = int(aspect*nz)

Rayleigh = float(args['--Rayleigh'])
Prandtl = float(Fraction(args['--Prandtl']))

# save data in directory named after script
data_dir = sys.argv[0].split('.py')[0]
data_dir += '_Ra{:}_Pr{:}_nx{:d}_nz{:d}'.format(args['--Rayleigh'], args['--Prandtl'], nx, nz)
if args['--GQL']:
    data_dir += '_GQL_Lambda{:}'.format(args['--Lambda'])
    GQL = True
    Lambda = int(float(args['--Lambda']))
else:
    GQL = False
if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])

from dedalus.tools.config import config

config['logging']['filename'] = os.path.join(data_dir,'logs/dedalus_log')
config['logging']['file_level'] = 'DEBUG'

if rank == 0:
    if not os.path.exists('{:s}/'.format(data_dir)):
        os.makedirs('{:s}/'.format(data_dir))
    logdir = os.path.join(data_dir,'logs')
    if not os.path.exists(logdir):
        os.mkdir(logdir)

import logging
logger = logging.getLogger(__name__)
logger.info("saving run in: {}".format(data_dir))

import dedalus.public as de
from dedalus.extras import flow_tools
from GQLProjection import GQLProjection
de.operators.parseables["Project"] = GQLProjection

mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]
else:
    log2 = np.log2(size)
    if log2 == int(log2):
        mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
    logger.info("running on processor mesh={}".format(mesh))

logger.info(sys.argv)
logger.info('-'*40)
logger.info("Run parameters")
for key in args:
    logger.info("{} = {}".format(key, args[key]))
logger.info('-'*40)

Lz = 1
Lx = Ly = aspect*Lz

# Bases and domain
x_basis = de.Fourier('x', nx, interval=(-Lx/2, Lx/2))
y_basis = de.Fourier('y', ny, interval=(-Ly/2, Ly/2))
z_basis = de.Chebyshev('z', nz, interval=(0, Lz))
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype='float', mesh=mesh)

problem = de.IVP(domain, variables=['T','T_z','Ox','Oy','p','u','v','w'])
problem.meta['p','T','u','v','w']['z']['dirichlet'] = True

problem.substitutions['Lap(A,A_z)'] = '(dx(dx(A)) + dy(dy(A)) + dz(A_z))'
problem.substitutions['Oz'] = '(dx(v)  - dy(u))'
problem.substitutions['Kx'] = '(dy(Oz) - dz(Oy))'
problem.substitutions['Ky'] = '(dz(Ox) - dx(Oz))'
problem.substitutions['Kz'] = '(dx(Oy) - dy(Ox))'
problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Ly/Lz'
problem.substitutions['plane_avg(A)'] = 'integ(A, "x", "y")/Lx/Ly'
problem.substitutions['enstrophy'] = 'Ox*Ox+Oy*Oy+Oz*Oz'
problem.substitutions['KE'] = '0.5*(u*u+v*v+w*w)'
problem.substitutions['Re'] = 'sqrt(u*u + v*v + w*w)/R'
problem.substitutions['T0'] = '1 - F*z'
problem.substitutions['T0_z'] = '-F'
problem.substitutions['enth_flux'] = 'w*(T+T0)'
problem.substitutions['cond_flux'] = '-P*(T_z+T0_z)'
problem.substitutions['tot_flux'] = 'cond_flux+enth_flux'
problem.substitutions['Nu'] = '(enth_flux + cond_flux)/vol_avg(cond_flux)'

problem.substitutions['u_z'] = "Oy + dx(w)"
problem.substitutions['v_z'] = "dy(w) - Ox"
problem.substitutions['w_z'] = "-dx(u) - dy(v)"

problem.parameters['P'] = (Rayleigh * Prandtl)**(-1/2)
problem.parameters['R'] = (Rayleigh / Prandtl)**(-1/2)
problem.parameters['F'] = F = 1
problem.parameters['Lx'] = Lx
problem.parameters['Ly'] = Ly
problem.parameters['Lz'] = Lz

if GQL:
    # substitutions for projecting onto the low and high modes
    problem.substitutions['P_h(A)'] = "Project(A,[{:d},{:d},{:d}],'h')".format(Lambda, Lambda, Lambda)
    problem.substitutions['P_l(A)'] = "Project(A,[{:d},{:d},{:d}],'l')".format(Lambda, Lambda, Lambda)

    problem.substitutions['u_l'] = "P_l(u)"
    problem.substitutions['u_h'] = "P_h(u)"
    problem.substitutions['v_l'] = "P_l(v)"
    problem.substitutions['v_h'] = "P_h(v)"
    problem.substitutions['w_l'] = "P_l(w)"
    problem.substitutions['w_h'] = "P_h(w)"
    problem.substitutions['T_l'] = "P_l(T)"
    problem.substitutions['T_h'] = "P_h(T)"

    problem.substitutions['UdotGrad_l(A_l,A_l_z,A_h,A_h_z)'] = 'P_l(u_l*dx(A_l) + v_l*dy(A_l) + w_l*(A_l_z) + u_h*dx(A_h) + v_h*dy(A_h) + w_h*(A_h_z))'
    problem.substitutions['UdotGrad_h(A_l,A_l_z,A_h,A_h_z)'] = 'P_h(u_l*dx(A_h) + v_l*dy(A_h) + w_l*(A_h_z) + u_l*dx(A_h) + v_l*dy(A_h) + w_l*(A_h_z))'
    problem.substitutions['UdotGrad(A,A_z)'] = 'UdotGrad_l(P_l(A),P_l(A_z),P_h(A),P_h(A)) + UdotGrad_h(P_l(A),P_l(A_z),P_h(A),P_h(A))'

else:
    # DNS equations
    problem.substitutions['UdotGrad(A,A_z)'] = '(u*dx(A) + v*dy(A) + w*(A_z))'

problem.add_equation("dt(T) - P*Lap(T, T_z)         - F*w = -UdotGrad(T, T_z)")
# O == omega = curl(u);  K = curl(O)
problem.add_equation("dt(u)  + R*Kx  + dx(p)              =  -UdotGrad(u, u_z)") #v*Oz - w*Oy")
problem.add_equation("dt(v)  + R*Ky  + dy(p)              =  -UdotGrad(v, v_z)") #w*Ox - u*Oz")
problem.add_equation("dt(w)  + R*Kz  + dz(p)    -T        =  -UdotGrad(w, w_z)") #u*Oy - v*Ox")

problem.add_equation("dx(u) + dy(v) + dz(w) = 0")
problem.add_equation("Ox + dz(v) - dy(w) = 0")
problem.add_equation("Oy - dz(u) + dx(w) = 0")
problem.add_equation("T_z - dz(T) = 0")

problem.add_bc("left(T) = 0")
problem.add_bc("left(u) = 0")
problem.add_bc("left(v) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(T) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("right(v) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_bc("right(p) = 0", condition="(nx == 0) and (ny == 0)")

# Build solver
if args['--SBDF4']:
    ts = de.timesteppers.SBDF4
    timestepper_history = [0,1,2,3]
    safety = 0.2
else:
    ts = de.timesteppers.SBDF2
    timestepper_history = [0,1]
    safety = 0.4

solver = problem.build_solver(ts)
logger.info('Solver built using {}'.format(ts))


# Initial conditions
def global_noise(domain, seed=42, filter_scale=0.5, scale=None, **kwargs):
    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=domain.dealias)
    slices = domain.dist.grid_layout.slices(scales=domain.dealias)
    rand = np.random.RandomState(seed=seed)
    noise = rand.standard_normal(gshape)[slices]

    # filter in k-space
    noise_field = domain.new_field()
    noise_field.set_scales(domain.dealias, keep_data=False)
    noise_field['g'] = noise
    noise_field.set_scales(filter_scale, keep_data=True)
    noise_field['c']
    noise_field['g']
    if scale is not None:
        noise_field.set_scales(scale, keep_data=True)
    else:
        noise_field.set_scales(domain.dealias, keep_data=True)

    return noise_field

T = solver.state['T']
Tz = solver.state['T_z']
amp = 1e-3
# for noise perturbations that respect div.u=0, populate A with noise and take u=curl(A)
noise = global_noise(domain, scale=1, filter_scale=0.5)
T['g'] = amp*noise['g']
T.differentiate('z', out=Tz)

# Integration parameters
solver.stop_sim_time = np.inf #Re
solver.stop_wall_time = np.inf

dt = 0.1

cfl_cadence = 1
report_cadence = 1
hermitian_cadence = 100
out_dt = 0.1

# Analysis
snapshots = solver.evaluator.add_file_handler('{:}/snapshots'.format(data_dir), sim_dt=out_dt, max_writes=10)
snapshots.add_task("interp(T, z={:})".format(Lz/2), layout='c', name='T midplane c')
snapshots.add_task("interp(u, z={:})".format(Lz/2), layout='c', name='u midplane c')
snapshots.add_task("interp(v, z={:})".format(Lz/2), layout='c', name='v midplane c')
snapshots.add_task("interp(w, z={:})".format(Lz/2), layout='c', name='w midplane c')
snapshots.add_task("interp(w*T, z={:})".format(Lz/2), layout='c', name='h midplane c')
snapshots.add_task("interp(T, z={:})".format(Lz/2), name='T midplane')
snapshots.add_task("interp(u, z={:})".format(Lz/2), name='u midplane')
snapshots.add_task("interp(v, z={:})".format(Lz/2), name='v midplane')
snapshots.add_task("interp(w, z={:})".format(Lz/2), name='w midplane')
snapshots.add_task("interp(w*T, z={:})".format(Lz/2), name='h midplane')
snapshots.add_task("interp(enth_flux, z={:})".format(Lz/2), name='enth midplane')
snapshots.add_task("interp(T, y=0)", name='T xz')
snapshots.add_task("interp(u, y=0)", name='u xz')
snapshots.add_task("interp(v, y=0)", name='v xz')
snapshots.add_task("interp(w, y=0)", name='w xz')
snapshots.add_task("interp(w*T, y=0)", name='h xz')
snapshots.add_task("interp(enth_flux, y=0)", name='enth xz')
snapshots.add_task("plane_avg(enth_flux)", name='avg enth_flux')
snapshots.add_task("plane_avg(cond_flux)", name='avg cond_flux')
snapshots.add_task("plane_avg(Nu)", name='avg Nu')


scalar = solver.evaluator.add_file_handler('{:}/scalar'.format(data_dir), sim_dt=out_dt, max_writes=np.inf)
scalar.add_task('vol_avg(KE)', name='KE')
scalar.add_task('vol_avg(enstrophy)', name='enstrophy')
scalar.add_task('vol_avg(Re)', name='Re')
scalar.add_task('vol_avg(Nu)', name='Nu')

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=cfl_cadence, safety=safety,
                     max_change=1.5, min_change=0.5, max_dt=dt, threshold=0.1)
CFL.add_velocities(('u', 'v', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=report_cadence)
flow.add_property('Re', name='Re')
flow.add_property('Nu', name='Nu')
flow.add_property('enstrophy', name='enstrophy')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    good_solution = True
    while solver.proceed and good_solution:
        if solver.iteration % hermitian_cadence in timestepper_history:
            for field in solver.state.fields:
                field.require_grid_space()
                logger.debug("Hermitian enforcment for {}".format(field))

        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if solver.iteration == 1 or solver.iteration % report_cadence == 0:
            log_string = 'Iteration: {:d}, Time: {:g}, dt: {:5.2g}, Re = {:.3g}, Ens = {:.3g}, Nu = {:.3g}'.format(
                solver.iteration, solver.sim_time, dt, flow.volume_average('Re'), flow.volume_average('enstrophy'), flow.volume_average('Nu'))
            logger.info(log_string)
            good_solution = np.isfinite(flow.volume_average('Re'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    main_loop_time = end_time-start_time
    n_iter_loop = solver.iteration-1
    n_cpu = domain.dist.comm_cart.size
    logger.info('Iterations: {:d}'.format(n_iter_loop))
    logger.info('Sim end time: {:f}'.format(solver.sim_time))
    logger.info('Run time: {:f} sec'.format(main_loop_time))
    logger.info('Run time: {:f} cpu-hr'.format(n_cpu*main_loop_time/(3600)))
    logger.info('mode-iter/cpu-sec: {:f} (main loop only)'.format(n_iter_loop*nx*ny*nz/(main_loop_time*n_cpu)))
