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
    --run_time_simtime=<run>    How long (simtime) to run for; if not set, runs for 5 diffusion times

    --label=<label>             Additional label for run output directory
"""
import numpy as np
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
t_diffusion = np.sqrt(Rayleigh)
if args['--run_time_simtime']:
    stop_sim_time = float(args['--run_time_simtime'])
else:
    stop_sim_time = 5*t_diffusion
if args['--run_time_iter']:
    stop_iter = int(float(args['--run_time_iter']))
else:
    stop_iter = np.inf
timestepper = d3.SBDF2
Δt = max_Δt = 0.1
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
x = xbasis.local_grid(1)
z = zbasis.local_grid(1)

bases = (xbasis,zbasis)
bases_perp = xbasis
bases_ncc = zbasis
# Fields
p = dist.Field(name='p', bases=bases)
b = dist.Field(name='b', bases=bases)
u = dist.VectorField(coords, name='u', bases=bases)
τ_p = dist.Field(name='τ_p')
τ_b1 = dist.Field(name='τ_b1', bases=bases_perp)
τ_b2 = dist.Field(name='τ_b2', bases=bases_perp)
τ_u1 = dist.VectorField(coords, name='τ_u1', bases=bases_perp)
τ_u2 = dist.VectorField(coords, name='τ_u2', bases=bases_perp)

integ = lambda A: d3.Integrate(d3.Integrate(A, 'x'), 'z')
avg = lambda A: integ(A)/(Lx*Lz)
x_avg = lambda A: d3.Integrate(A, coords['x'])/(Lx)

curl = lambda A: d3.Curl(A)
grad = lambda A: d3.Gradient(A, coords)
transpose = lambda A: d3.TransposeComponents(A)

# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)

ex, ey, ez = coords.unit_vector_fields(dist)

lift_basis = zbasis
lift = lambda A, n: d3.Lift(A, lift_basis, n)

b0 = dist.Field(name='b0', bases=bases_ncc )
b0['g'] = Lz - z

e_ij = grad(u) + transpose(grad(u))

nu_inv = 1/nu
ω = curl(u)

# Problem
problem = d3.IVP([p, b, u, τ_p, τ_b1, τ_b2, τ_u1, τ_u2], namespace=locals())
problem.add_equation("div(u) + nu_inv*lift(τ_u2,-1)@ez + τ_p = 0")
problem.add_equation("dt(u) + tau_drag*u - nu*lap(u) + grad(p) - b*ez + lift(τ_u2,-2) + lift(τ_u1,-1) = cross(u, ω)")
problem.add_equation("dt(b) + u@grad(b0) - kappa*lap(b) + lift(τ_b2,-2) + lift(τ_b1,-1) = - (u@grad(b))")
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

flux_c = u@ez*(b0+b)
flux_κ = -kappa*grad(b+b0)@ez

# Analysis
snapshots = solver.evaluator.add_file_handler(data_dir+'/snapshots', sim_dt=0.1, max_writes=10)
snapshots.add_task(b+b0, name='b')
snapshots.add_task(ω, name='vorticity')
snapshots.add_task(ω@ω, name='enstrophy')
snapshots.add_task((b+b0)(z=Lz/2), layout='c', name='b midplane c')
snapshots.add_task((u)(z=Lz/2), layout='c', name='u reg midplane c')
snapshots.add_task((ez@u)(z=Lz/2), layout='c', name='u_z midplane c')
snapshots.add_task((ex@u)(z=Lz/2), layout='c', name='u_x midplane c')
snapshots.add_task((ω)(z=Lz/2), layout='c', name='ω reg midplane c')
snapshots.add_task((ey@ω)(z=Lz/2), layout='c', name='ω_y midplane c')

traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=0.1, max_writes=None)
traces.add_task(avg(KE), name='KE')
traces.add_task(avg(PE), name='PE')
traces.add_task(np.sqrt(2*avg(KE))/nu, name='Re')
traces.add_task(avg(ω@ω), name='enstrophy')
traces.add_task(1 + avg(flux_c)/avg(flux_κ), name='Nu')
traces.add_task(x_avg(np.sqrt(τ_u1@τ_u1)), name='τu1')
traces.add_task(x_avg(np.sqrt(τ_u2@τ_u2)), name='τu2')
traces.add_task(x_avg(np.sqrt(τ_b1**2)), name='τb1')
traces.add_task(x_avg(np.sqrt(τ_b2**2)), name='τb2')
traces.add_task(np.sqrt(τ_p**2), name='τp')

cadence = 10
# CFL
CFL = d3.CFL(solver, initial_dt=max_Δt, cadence=1, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_Δt)
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

# Main loop
vol = Lx*Lz
try:
    good_solution = True
    logger.info('Starting loop')
    while solver.proceed and good_solution:
        solver.step(Δt)
        if (solver.iteration-1) % cadence == 0:
            max_Re = flow.max('Re')
            avg_Re = flow.volume_integral('Re')/vol
            avg_PE = flow.volume_integral('PE')/vol
            avg_KE = flow.volume_integral('KE')/vol
            avg_Nu = 1+flow.volume_integral('f_c')/flow.volume_integral('f_κ')
            max_τ = np.max([flow.max('τu1'),flow.max('τu2'),flow.max('τb1'),flow.max('τb2'),flow.max('τp')])
            logger.info('Iteration={:d}, Time={:.3e} ({:.1e}), dt={:.1e}, PE={:.3e}, KE={:.3e}, Re={:.2g}, Nu={:.2g}, τ={:.2e}'. format(solver.iteration, solver.sim_time, solver.sim_time/t_diffusion, Δt, avg_PE, avg_KE, avg_Re, avg_Nu, max_τ))
            good_solution = np.isfinite(max_Re)
        Δt = CFL.compute_timestep()

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
