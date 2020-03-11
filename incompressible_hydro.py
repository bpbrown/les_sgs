"""
Dedalus script for incompressible forced turbulence,
to test SGS methods for LES.

Usage:
    incompressible_hydro.py [options]

Options:
    --n=<n>           Fourier resolution of domain, nxnxn [default: 128]
    --k=<k>           k-value to stir domain at [default: 10]
    --Reynolds=<Re>   Reynolds number of the turbulence [default: 1000]

"""

import numpy as np
import dedalus.public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

nx = ny = nz = n = int(args['--n'])
kx = ky = kz = k = float(args['--k'])
Re = float(args['--Reynolds'])


# Bases and domain
x_basis = de.Fourier('x', nx, interval=(0, 2*np.pi))
y_basis = de.Fourier('y', ny, interval=(0, 2*np.pi))
z_basis = de.Fourier('z', ny, interval=(0, 2*np.pi))
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype='float')

prob = domain.IVP(domain, variables=['u','v','w','ϖ'])
problem.parameters['R'] = 1/Re
problem.parameters['kx'] = kx
problem.parameters['ky'] = ky
problem.parameters['kz'] = kz
problem.substitutions['Lap(A)'] = 'dx(dx(A))+dy(dy(A))+dz(dz(A))'
problem.substitutions['u.grad(A)'] = 'u*dx(A) + v*dy(A) + w*dz(A)'
problem.substitutions['fx'] = 'cos(kx*x)'
problem.substitutions['fy'] = 'cos(ky*y)'
problem.substitutions['fz'] = 'cos(kz*z)'
problem.add_equation('dt(u) + dx(ϖ) - R*Lap(u) = -u.grad(u) + fx')
problem.add_equation('dt(v) + dy(ϖ) - R*Lap(v) = -u.grad(v) + fy')
problem.add_equation('dt(w) + dz(ϖ) - R*Lap(w) = -u.grad(w) + fz')
problem.add_equation('dx(u) + dy(v) + dz(w) = 0')

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF4)
logger.info('Solver built')

# Integration parameters
solver.stop_sim_time = 10

dt = 0.2*1/n

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=10)
snapshots.add_system(solver.state)
snapshots.add_task(u, name='u_hat', layout='c')
snapshots.add_task(v, name='v_hat', layout='c')
snapshots.add_task(w, name='w_hat', layout='c')

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.4,
                     max_change=1.5, min_change=0.5, max_dt=dt, threshold=0.1)
CFL.add_velocities(('u', 'v', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("sqrt(u*u + v*v + w*w) / R", name='Re')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 1 == 0:
            logger.info('Iteration: {:d}, Time: {:f}, dt: {:f}, Re = {:g}, {:g}'.format(solver.iteration, solver.sim_time, dt, flow.max('Re'), flow.volume_average('Re'))
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
    logger.info('mode-iter/cpu-sec: {:f} (main loop only)'.format(n_iter_loop*n**3/(main_loop_time*n_cpu))
