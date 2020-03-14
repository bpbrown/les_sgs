"""
Dedalus script for incompressible forced turbulence,
to test SGS methods for LES.

Usage:
    incompressible_hydro.py [options]

Options:
    --n=<n>           Fourier resolution of domain, nxnxn [default: 128]
    --k=<k>           k-value to stir domain at [default: 10]
    --Reynolds=<Re>   Reynolds number of the turbulence [default: 1e3]

    --dt=<dt>         Initial dt, default is at dt=Re

    --mesh=<mesh>     Processor mesh, takes format n1,n2 for a 2-d mesh decomposition of n1xn2 cores

    --label=<label>   Case name label
"""

import numpy as np
import dedalus.public as de
from dedalus.extras import flow_tools

import sys
import os
from mpi4py import MPI

import logging
import time

from docopt import docopt
args = docopt(__doc__)

# Find MPI rank
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

nx = ny = nz = n = int(args['--n'])
kx = ky = kz = k = float(args['--k'])
Re = float(args['--Reynolds'])

# save data in directory named after script
data_dir = sys.argv[0].split('.py')[0]
data_dir += '_Re{:}_n{:}'.format(args['--Reynolds'], args['--n'])
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
logger = logging.getLogger(__name__)
logger.info("saving run in: {}".format(data_dir))


mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]
else:
    log2 = np.log2(size)
    if log2 == int(log2):
        mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
    logger.info("running on processor mesh={}".format(mesh))

Lx = Ly = Lz = 2*np.pi

# Bases and domain
x_basis = de.Fourier('x', nx, interval=(0, Lx))
y_basis = de.Fourier('y', ny, interval=(0, Ly))
z_basis = de.Fourier('z', nz, interval=(0, Lz))
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype='float')

problem = de.IVP(domain, variables=['u','v','w','p','lubar','lubart'])
problem.parameters['R'] = 1/Re
problem.parameters['k'] = k
problem.parameters['Lx'] = Lx
problem.parameters['Ly'] = Ly
problem.parameters['Lz'] = Lz
problem.substitutions['Lap(A)'] = 'dx(dx(A))+dy(dy(A))+dz(dz(A))'
problem.substitutions['u_grad(A)'] = 'u*dx(A) + v*dy(A) + w*dz(A)'
problem.substitutions['fx'] = 'cos(k*x)*cos(k*z)'
problem.substitutions['fy'] = 'cos(k*y)*cos(k*z)'
problem.substitutions['fz'] = '(sin(k*x)+sin(k*y))*sin(k*z)'
problem.substitutions['O_x'] = 'dy(w) - dz(v)'
problem.substitutions['O_y'] = '-dx(w) + dz(u)'
problem.substitutions['O_z'] = 'dx(v) - dy(u)'
problem.substitutions['KE'] = '0.5*(u*u+v*v+w*w)'
problem.substitutions['Re'] = 'sqrt(u*u + v*v + w*w) / R'
problem.substitutions['enstrophy'] = 'O_x*O_x+O_y*O_y+O_z*O_z'
problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Ly/Lz'
problem.add_equation('lubar=log(integ(sqrt(u*u+v*v+w*w))/Lx/Ly/Lz)')
problem.add_equation('dt(lubar)-lubart = 0')
problem.add_equation('dt(u) + dx(p) - R*Lap(u) = -u*lubart - u_grad(u) + R*(exp(-lubar)-1)*Lap(u) + fx*exp(-2*lubar)')
problem.add_equation('dt(v) + dy(p) - R*Lap(v) = -v*lubart - u_grad(v) + R*(exp(-lubar)-1)*Lap(v) + fy*exp(-2*lubar)')
problem.add_equation('dt(w) + dz(p) - R*Lap(w) = -w*lubart - u_grad(w) + R*(exp(-lubar)-1)*Lap(w) + fz*exp(-2*lubar)')
problem.add_equation('dx(u) + dy(v) + dz(w) = 0', condition="(nx != 0) or (ny != 0) or (nz != 0)")
problem.add_equation('p = 0', condition="(nx == 0) and (ny == 0) and (nz == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF4)
logger.info('Solver built')


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

u = solver.state['u']
v = solver.state['v']
w = solver.state['w']
Ax = domain.new_field()
Ay = domain.new_field()
Az = domain.new_field()
noise_x = global_noise(domain, scale=1, filter_scale=0.5, seed=42)
noise_y = global_noise(domain, scale=1, filter_scale=0.5, seed=43)
noise_z = global_noise(domain, scale=1, filter_scale=0.5, seed=44)
amp = 0.1*1/Re
# for noise perturbations that respect div.u=0, populate A with noise and take u=curl(A)
Ax['g'] = amp*noise_x['g']
Ay['g'] = amp*noise_y['g']
Az['g'] = amp*noise_z['g']
u['g'] = Az.differentiate('y')['g'] - Ay.differentiate('z')['g']
v['g'] = Ax.differentiate('z')['g'] - Az.differentiate('x')['g']
w['g'] = Ay.differentiate('x')['g'] - Ax.differentiate('y')['g']


# Integration parameters
solver.stop_sim_time = np.inf #Re
solver.stop_wall_time = np.inf

if args['--dt']:
    dt = float(args['--dt'])
else:
    dt = Re

cfl_cadence = 1
report_cadence = 1
out_dt = 10*dt

# Analysis
snapshots = solver.evaluator.add_file_handler('{:}/snapshots'.format(data_dir), sim_dt=out_dt, max_writes=10)
snapshots.add_system(solver.state)
snapshots.add_task('u', name='u_hat', layout='c')
snapshots.add_task('v', name='v_hat', layout='c')
snapshots.add_task('w', name='w_hat', layout='c')
scalar = solver.evaluator.add_file_handler('{:}/scalar'.format(data_dir), sim_dt=out_dt, max_writes=np.inf)
scalar.add_task('vol_avg(KE)', name='KE')
scalar.add_task('vol_avg(enstrophy)', name='enstrophy')
scalar.add_task('vol_avg(Re)', name='Re')

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=cfl_cadence, safety=0.4,
                     max_change=1.5, min_change=0.5, max_dt=dt, threshold=0.1)
CFL.add_velocities(('u', 'v', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=report_cadence)
flow.add_property('Re', name='Re')
flow.add_property('enstrophy', name='enstrophy')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    good_solution = True
    while solver.proceed and good_solution:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if solver.iteration == 1 or (solver.iteration) % report_cadence == 0:
            log_string = 'Iteration: {:d}, Time: {:g}, dt: {:5.2g}, Re = {:.3g}, Ens = {:.3g}'.format(
                solver.iteration, solver.sim_time, dt, flow.volume_average('Re'), flow.volume_average('enstrophy'))
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
    logger.info('mode-iter/cpu-sec: {:f} (main loop only)'.format(n_iter_loop*n**3/(main_loop_time*n_cpu)))
