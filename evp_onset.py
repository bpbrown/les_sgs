"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.

Usage:
    evp_onset.py [options]

Options:
    --Nz=<Nz>              Vertical modes [default: 64]

    --tau_drag=<tau_drag>       1/Newtonian drag timescale; default is zero drag

    --stress_free               Use stress free boundary conditions
    --flux_temp                 Use mixed flux/temperature boundary conditions

    --Rayleigh=<Rayleigh>       Rayleigh number [default: 1e6]

    --target=<targ>   Target value for sparse eigenvalue search [default: 0]
    --eigs=<eigs>     Target number of eigenvalues to search for [default: 20]

    --dense           Solve densely for all eigenvalues (slow)
"""
import numpy as np
import sys
import os

from docopt import docopt
args = docopt(__doc__)

N_evals = int(float(args['--eigs']))
target = float(args['--target'])


# Parameters
Lz = 1
Nz = int(args['--Nz'])

stress_free = args['--stress_free']
flux_temp = args['--flux_temp']

import dedalus.public as d3

import logging
logger = logging.getLogger(__name__)
for system in ['evaluator', 'subsystems']:
    dlog = logging.getLogger(system)
    dlog.setLevel(logging.WARNING)

if args['--tau_drag']:
    tau_drag = float(args['--tau_drag'])
else:
    tau_drag = 0

Rayleigh_in = float(args['--Rayleigh'])
Prandtl = 1
dealias = 3/2
dtype = np.complex128

# Bases
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=dtype)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
z = zbasis.local_grid(1)

bases = (zbasis)
bases_perp = None
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

# control parameters
Rayleigh = dist.Field(name='Ra')
kx = dist.Field(name='kx')

dx = lambda A: 1j*kx*A
dy = lambda A: 0*A
grad = lambda A: d3.Gradient(A, coords) + dx(A)*ex + dy(A)*ey
div = lambda A:  d3.div(A) + dx(A@ex) + dy(A@ey)
lap = lambda A: d3.lap(A) + dx(dx(A)) + dy(dy(A))
transpose = lambda A: d3.TransposeComponents(A)

# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)

ex, ey, ez = coords.unit_vector_fields(dist)

lift_basis = zbasis
#lift_basis = zbasis.clone_with(a=zbasis.a+2, b=zbasis.b+2)
lift = lambda A, n: d3.Lift(A, lift_basis, n)

b0 = dist.Field(name='b0', bases=bases_ncc )
b0['g'] = Lz - z

e_ij = grad(u) + transpose(grad(u))

nu_inv = 1/nu

omega = dist.Field(name='omega')
dt = lambda A: omega*A
# Problem
problem = d3.EVP([p, b, u, τ_p, τ_b1, τ_b2, τ_u1, τ_u2], eigenvalue=omega, namespace=locals())
grad_b0 = grad(b0).evaluate()
problem.add_equation("div(u) + nu_inv*lift(τ_u2,-1)@ez + τ_p = 0")
problem.add_equation("dt(u) + tau_drag*u - nu*lap(u) + grad(p) - b*ez + lift(τ_u2,-2) + lift(τ_u1,-1) = 0")
#problem.add_equation("dt(b) + u@grad(b0) - kappa*lap(b) + lift(τ_b2,-2) + lift(τ_b1,-1) = 0")
problem.add_equation("dt(b) + u@grad_b0 - kappa*lap(b) + lift(τ_b2,-2) + lift(τ_b1,-1) = 0")
if stress_free:
    problem.add_equation("ez@(ex@e_ij(z=0)) = 0")
    problem.add_equation("ez@(ey@e_ij(z=0)) = 0")
    problem.add_equation("ez@u(z=0) = 0")
    problem.add_equation("ez@(ex@e_ij(z=Lz)) = 0")
    problem.add_equation("ez@(ey@e_ij(z=Lz)) = 0")
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
solver = problem.build_solver()



def compute_growth_rate(kx_i, Ra_i):
    kx['g'] = kx_i
    Rayleigh['g'] = Ra_i
    if args['--dense']:
        solver.solve_dense(solver.subproblems[0], rebuild_matrices=True)
        solver.eigenvalues = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
    else:
        solver.solve_sparse(solver.subproblems[0], N=N_evals, target=target, rebuild_matrices=True)
    i_evals = np.argsort(solver.eigenvalues.real)
    evals = solver.eigenvalues[i_evals]
    peak_eval = evals[-1]
    # choose convention: return the positive complex mode of the pair
    if peak_eval.imag < 0:
        peak_eval = np.conj(peak_eval)
    return peak_eval

def peak_growth_rate(*args):
    rate = compute_growth_rate(*args)
    # flip sign so minimize finds maximum
    return -1*rate.real

import scipy.optimize as sciop

#kxs = np.geomspace(0.1, 10, num=31)
#for kx_i in kxs:
kx_i = 2
for Ra in np.geomspace(5e2,1e4,num=10):
    bounds = sciop.Bounds(lb=1, ub=10)
    result = sciop.minimize(peak_growth_rate, kx_i, args=(Ra), bounds=bounds, method='Nelder-Mead', tol=1e-5)
    σ = compute_growth_rate(result.x[0], Ra)
    logger.info('peak search: start at Ra = {:.4g}, kx = {:.4g}, found σ_max = {:.2g},{:.2g}i, kx = {:.4g}'.format(Ra, kx_i, σ.real, σ.imag, result.x[0]))
    # update next guess with current value
    kx_i = result.x[0]
