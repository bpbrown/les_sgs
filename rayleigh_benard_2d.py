"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.

Usage:
    rayleigh_benard_2d.py [options]

Options:
    --Nx=<Nx>              Horizontal modes; default is aspect x Nz
    --Nz=<Nz>              Vertical modes [default: 64]
    --aspect=<aspect>      Aspect ratio of domain [default: 4]

    --tau_drag=<tau_drag>       1/Newtonian drag timescale; default is zero drag

    --stress_free               Use stress free boundary conditions
    --flux_temp                 Use mixed flux/temperature boundary conditions

    --Rayleigh=<Rayleigh>       Rayleigh number [default: 1e6]

    --run_time_iter=<iter>      How many iterations to run for
    --run_time_simtime=<run>    How long (simtime) to run for

    --label=<label>             Additional label for run output directory
"""
import numpy as np
import time
import sys
import os

from docopt import docopt
args = docopt(__doc__)


aspect = float(args['--aspect'])
# Parameters
Lx, Lz = aspect, 1
Nz = int(args['--Nz'])
if args['--Nx']:
    Nx = int(args['--Nx'])
else:
    Nx = int(aspect*Nz)

stress_free = args['--stress_free']
flux_temp = args['--flux_temp']

data_dir = './'+sys.argv[0].split('.py')[0]
if stress_free:
    data_dir += '_SF'
if flux_temp:
    data_dir += '_FT'
data_dir += '_Ra{}'.format(args['--Rayleigh'])
if args['--tau_drag']:
    tau_drag = float(args['--tau_drag'])
    data_dir += '_tau{}'.format(args['--tau_drag'])
else:
    tau_drag = 0
data_dir += '_Nz{}_Nx{}'.format(Nz, Nx)
if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])

import dedalus.tools.logging as dedalus_logging
dedalus_logging.add_file_handler(data_dir+'/logs/dedalus_log', 'DEBUG')

import dedalus.public as d3

import logging
logger = logging.getLogger(__name__)
dlog = logging.getLogger('evaluator')
dlog.setLevel(logging.WARNING)
logger.info("saving data in {}".format(data_dir))

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
ncpu = comm.size

Rayleigh = float(args['--Rayleigh'])
Prandtl = 1
dealias = 3/2
if args['--run_time_simtime']:
    stop_sim_time = float(args['--run_time_simtime'])
else:
    stop_sim_time = np.inf
if args['--run_time_iter']:
    stop_iter = int(float(args['--run_time_iter']))
else:
    stop_iter = np.inf
timestepper = d3.SBDF2
max_timestep = 0.1
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
x = xbasis.local_grid(1)
z = zbasis.local_grid(1)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
b = dist.Field(name='b', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
τ_p = dist.Field(name='τ_p')
τ_b1 = dist.Field(name='τ_b1', bases=xbasis)
τ_b2 = dist.Field(name='τ_b2', bases=xbasis)
τ_u1 = dist.VectorField(coords, name='τ_u1', bases=xbasis)
τ_u2 = dist.VectorField(coords, name='τ_u2', bases=xbasis)

grid = lambda A: d3.Grid(A)
div = lambda A: d3.Divergence(A, index=0)
from dedalus.core.operators import Skew
skew = lambda A: Skew(A)
#avg = lambda A: d3.Integrate(A, coords)/(Lx*Lz)
integ = lambda A: d3.Integrate(d3.Integrate(A, 'x'), 'z')
avg = lambda A: integ(A)/(Lx*Lz)
x_avg = lambda A: d3.Integrate(A, coords['x'])/(Lx)
grad = lambda A: d3.Gradient(A, coords)
transpose = lambda A: d3.TransposeComponents(A)

# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)

ex, ez = coords.unit_vector_fields(dist)

lift_basis = zbasis.clone_with(a=zbasis.a+2, b=zbasis.b+2)
lift = lambda A, n: d3.Lift(A, lift_basis, n)

lift_basis1 = zbasis.clone_with(a=zbasis.a+1, b=zbasis.b+1)
lift1 = lambda A, n: d3.Lift(A, lift_basis1, n)

b0 = dist.Field(name='b0', bases=zbasis)
b0['g'] = Lz - z

e_ij = grad(u) + transpose(grad(u))

nu_inv = 1/nu

# Problem
problem = d3.IVP([p, b, u, τ_p, τ_b1, τ_b2, τ_u1, τ_u2], namespace=locals())
problem.add_equation("div(u) + nu_inv*lift1(τ_u2,-1)@ez + τ_p = 0")
problem.add_equation("dt(u) + tau_drag*u - nu*lap(u) + grad(p) - b*ez + lift(τ_u2,-2) + lift(τ_u1,-1) = -skew(grid(u))*div(skew(u))")
problem.add_equation("dt(b) + u@grad(b0) - kappa*lap(b) + lift(τ_b2,-2) + lift(τ_b1,-1) = - u@grad(b)")
if stress_free:
    problem.add_equation("ez@(ex@e_ij(z=0)) = 0")
    problem.add_equation("ez@u(z=0) = 0")
    problem.add_equation("ez@(ex@e_ij(z=Lz)) = 0")
    problem.add_equation("ez@u(z=Lz) = 0")
else:
    problem.add_equation("u(z=0) = 0")
    problem.add_equation("u(z=Lz) = 0")
if flux_temp:
    problem.add_equation("ez@grad(b)(z=0) = 0")
else:
    problem.add_equation("b(z=0) = 0")
problem.add_equation("b(z=Lz) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time
solver.stop_iteration = stop_iter

# Initial conditions
zb, zt = zbasis.bounds
b.fill_random('g', seed=42, distribution='normal', scale=1e-5) # Random noise
b['g'] *= z * (Lz - z) # Damp noise at walls
b.low_pass_filter(scales=0.25)



KE = 0.5*u@u
PE = b+b0
ω = -div(skew(u))
flux_c = u@ez*(b0+b)
flux_c.store_last=True
flux_κ = -kappa*grad(b+b0)@ez
flux_κ.store_last=True

# Analysis
snapshots = solver.evaluator.add_file_handler(data_dir+'/snapshots', sim_dt=0.1, max_writes=10)
snapshots.add_task(b+b0, name='b')
snapshots.add_task(ω, name='vorticity')
snapshots.add_task(ω**2, name='enstrophy')

traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=0.1, max_writes=np.inf)
traces.add_task(avg(KE), name='KE')
traces.add_task(avg(PE), name='PE')
traces.add_task(np.sqrt(2*avg(KE))/nu, name='Re')
traces.add_task(avg(ω**2), name='enstrophy')
traces.add_task(1 + avg(flux_c)/avg(flux_κ), name='Nu')
traces.add_task(x_avg(np.sqrt(τ_u1@τ_u1)), name='τu1')
traces.add_task(x_avg(np.sqrt(τ_u2@τ_u2)), name='τu2')
traces.add_task(x_avg(np.sqrt(τ_b1**2)), name='τb1')
traces.add_task(x_avg(np.sqrt(τ_b2**2)), name='τb2')
traces.add_task(np.sqrt(τ_p**2), name='τp')

cadence = 10
# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=1, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=cadence)
flow.add_property(np.sqrt(u@u)/nu, name='Re')
flow.add_property(KE, name='KE')
flow.add_property(PE, name='PE')
flow.add_property(flux_c, name='f_c')
flow.add_property(flux_κ, name='f_κ')
flow.add_property(np.sqrt(τ_u1@τ_u1), name='τu1')
flow.add_property(np.sqrt(τ_u2@τ_u2), name='τu2')
flow.add_property(np.sqrt(τ_b1**2), name='τb1')
flow.add_property(np.sqrt(τ_b2**2), name='τb2')
flow.add_property(np.sqrt(τ_p**2), name='τp')

startup_iter = 10
# Main loop
try:
    good_solution = True
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed and good_solution:
        if solver.iteration == startup_iter:
            main_start = time.time()
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % cadence == 0:
            max_Re = flow.max('Re')
            avg_Re = flow.grid_average('Re')
            avg_PE = flow.grid_average('PE')
            avg_KE = flow.grid_average('KE')
            avg_Nu = 1+flow.grid_average('f_c')/flow.grid_average('f_κ')
            max_τ = np.max([flow.max('τu1'),flow.max('τu2'),flow.max('τb1'),flow.max('τb2'),flow.max('τp')])
            logger.info('Iteration={:d}, Time={:.3e}, dt={:.1e}, PE={:.3e}, KE={:.3e}, Re={:.2g}, Nu={:.2g}, τ={:.2e}'. format(solver.iteration, solver.sim_time, timestep, avg_PE, avg_KE, avg_Re, avg_Nu, max_τ))
            good_solution = np.isfinite(max_Re)
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()

    startup_time = main_start - start_time
    main_loop_time = end_time - main_start
    DOF = Nx*Nz
    niter = solver.iteration - startup_iter
    if rank==0:
        print('performance metrics:')
        print('    startup time   : {:}'.format(startup_time))
        print('    main loop time : {:}'.format(main_loop_time))
        print('    main loop iter : {:d}'.format(niter))
        print('    wall time/iter : {:f}'.format(main_loop_time/niter))
        print('          iter/sec : {:f}'.format(niter/main_loop_time))
        print('DOF-cycles/cpu-sec : {:}'.format(DOF*niter/(ncpu*main_loop_time)))
    solver.log_stats()
