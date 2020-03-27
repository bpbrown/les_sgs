"""
Plot spectral outputs from snapshot output .h5 file.

Usage:
    plot_spectra.py <file>... [options]

Options:
    --output=<output>    Output directory; if blank, a guess based on <file> location will be made.
"""
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib
import h5py
import time

# needed for logging to work
import dedalus.public as de

import logging
logger = logging.getLogger(__name__.split('.')[-1])
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

from docopt import docopt
args = docopt(__doc__)
files = args['<file>']

if args['--output'] is not None:
    output_path = pathlib.Path(args['--output']).absolute()
else:
    data_dir = args['<file>'][0].split('/')[0]
    data_dir += '/'
    output_path = pathlib.Path(data_dir).absolute()

u_tot_tag = r'$u_\mathrm{tot}$'
u_perp_tag = r'$u_\perp$'
w_tag = '$w$'
T_tag = '$T$'
pow = {u_perp_tag: 0, w_tag:0, u_tot_tag:0, T_tag:0}
n_t = 0
first_time = None

for file in files:
    logger.info('opening {:s}'.format(file))
    f = h5py.File(file, 'r')
    data = {}
    t = f['scales/sim_time'][:]
    kx = f['scales/kx'][:]
    ky = f['scales/ky'][:]
    x = f['scales/x/1.0'][:]
    y = f['scales/y/1.0'][:]
    for key in f['tasks']:
        data[key] = f['tasks/'+key][:,:,:,0]
    f.close()
    if not first_time:
        first_time = t[0]

    n_t += t.shape[0]
    for i in range(t.shape[0]):
        u_c = data['u midplane c'][i,:,:]
        v_c = data['v midplane c'][i,:,:]
        w_c = data['w midplane c'][i,:,:]
        T_c = data['T midplane c'][i,:,:]

        pow[u_perp_tag] += np.abs(u_c*np.conj(u_c)+v_c*np.conj(v_c))
        pow[w_tag] += np.abs(w_c*np.conj(w_c))
        pow[u_tot_tag] += np.abs(u_c*np.conj(u_c)+v_c*np.conj(v_c)+w_c*np.conj(w_c))
        pow[T_tag] += np.abs(T_c*np.conj(T_c))

last_time = t[-1]

for q in pow:
    pow[q] /= n_t

logger.info("power spectra accumulated from t={:.3g}--{:.3g} ({:.3g} total)".format(first_time, last_time, last_time-first_time))
n_ky = ky.shape[0]

ky_shift = int((n_ky+1)/2)
for q in pow:
    pow[q] = np.roll(pow[q],-ky_shift,axis=1)
ky = np.roll(ky,  -ky_shift,axis=0)

kx_g, ky_g = np.meshgrid(kx, ky, indexing='ij')
logger.info("kx {}, ky {}".format(kx.shape, ky.shape))
logger.info("kx_g {}, ky_g {}".format(kx_g.shape, ky_g.shape))
logger.info("pow {}".format(pow[u_tot_tag].shape))
n_kx = kx.shape[0]
n_theta = 2*n_kx
n_kr = n_kx

theta = np.arctan2(ky_g, kx_g)
theta_u = np.linspace(-np.pi/2, np.pi/2, n_theta)
kr = np.sqrt(kx_g*kx_g+ky_g*ky_g)
kr_u = np.linspace(0, np.max(kr), n_kr)

fig_grid, ax_grid = plt.subplots(ncols=2, nrows=2, figsize=[6,6])
ax_grid[0,0].pcolormesh(kx_g, ky_g, kx_g, shading='flat')
ax_grid[0,1].pcolormesh(kx_g, ky_g, ky_g, shading='flat')
ax_grid[1,0].pcolormesh(kx_g, ky_g, theta, shading='flat')
ax_grid[1,1].pcolormesh(kx_g, ky_g, kr, shading='flat')
fig_grid.savefig('{:s}/grids.png'.format(str(output_path)), dpi=300)

original_coords_flat = (theta.flatten(), kr.flatten())

theta_u_g, kr_u_g = np.meshgrid(theta_u, kr_u, indexing='ij')
new_coords_flat = (theta_u.flatten(), kr_u.flatten())


from scipy.interpolate import griddata

start_int = time.time()
pow_u ={}
for tag in pow:
    pow_u[tag] = griddata(original_coords_flat, pow[tag].flatten(), (theta_u_g, kr_u_g), method='cubic')
    logger.info(pow_u[tag].shape)
end_int = time.time()
print("Interpolation took {:g} seconds".format(end_int-start_int))
print(theta_u_g.shape, kr_u_g.shape)

u = data['u midplane'][-1,:,:]
v = data['v midplane'][-1,:,:]
w = data['w midplane'][-1,:,:]
T = data['T midplane'][-1,:,:]
x_g, y_g = np.meshgrid(x, y)
kx_g, ky_g = np.meshgrid(kx, ky)

fig_spectra, ax_spectra = plt.subplots(ncols=1, nrows=2, figsize=[6,3], subplot_kw=dict(polar=True))
ax_spectra[0].pcolormesh(theta_u_g, kr_u_g, np.log(pow_u[u_tot_tag]).T, shading='flat')
ax_spectra[1].pcolormesh(theta_u_g, kr_u_g, np.log(pow_u[T_tag]).T, shading='flat')
fig_spectra2d.savefig('{:s}/power_spectrum_2d_kr.png'.format(str(output_path)), dpi=300)

arrow=(slice(None,None,8),slice(None,None,8))

fig_spectra2d, ax_spectra2d = plt.subplots(ncols=2, nrows=2, figsize=[6,6])

ax_spectra2d[0,0].pcolormesh(kx_g, ky_g, np.log(pow[u_tot_tag]).T, shading='flat')
ax_spectra2d[0,0].set_xlabel(r'$k_x$')
ax_spectra2d[0,0].set_ylabel(r'$k_y$')
ax_spectra2d[0,0].set_title(u_tot_tag)
ax_spectra2d[0,1].pcolormesh(x_g, y_g, w.T, shading='flat')
ax_spectra2d[0,1].quiver(x_g[arrow], y_g[arrow], u[arrow].T, v[arrow].T, units='xy', scale=0.9)
ax_spectra2d[0,1].yaxis.tick_right()
ax_spectra2d[0,1].set_title('t={:.3g}'.format(t[-1]))
ax_spectra2d[0,1].set_xlabel(r'$x$')
ax_spectra2d[0,1].set_ylabel(r'$y$')
# ax_spectra2d[2].pcolormesh(theta_u, kr_u, np.log(pow_u).T, shading='flat', polar=True)
# ax_spectra2d[2].set_rticks([])
# ax_spectra2d[2].set_aspect(1)

ax_spectra2d[1,0].pcolormesh(kx_g, ky_g, np.log(pow[T_tag]).T, shading='flat')
ax_spectra2d[1,0].set_xlabel(r'$k_x$')
ax_spectra2d[1,0].set_ylabel(r'$k_y$')
ax_spectra2d[1,0].set_title(T_tag)
ax_spectra2d[1,1].pcolormesh(x_g, y_g, T.T, shading='flat', cmap='RdYlBu')
ax_spectra2d[1,1].yaxis.tick_right()
ax_spectra2d[1,1].set_xlabel(r'$x$')
ax_spectra2d[1,1].set_ylabel(r'$y$')


fig_spectra2d.savefig('{:s}/power_spectrum_2d.png'.format(str(output_path)), dpi=300)

fig_spectra, ax_spectra = plt.subplots()
for q in pow:
    print(pow_u[q].shape)
    avg_spectra = np.mean(pow_u[q], axis=0)
    ax_spectra.plot(kr_u, avg_spectra, label=q)
    avg_spectra = np.mean(pow[q], axis=1)
    ax_spectra.plot(kx, avg_spectra, label=q, linestyle='dashed')

norm = np.max(pow[T_tag])
logger.info('powerlaw k^(-5/3) norm is {:.3g}'.format(norm))
ax_spectra.plot(kx, norm*kx**(-5/3), color='black', linestyle='dashed', label=r'$k_\perp^{-5/3}$')

ax_spectra.legend()
ax_spectra.set_ylabel(r'Horizontal power spectrum ($u_\perp*u_\perp$)')
ax_spectra.set_xlabel(r'$k_\perp$')
ax_spectra.set_yscale('log')
ax_spectra.set_xscale('log')
fig_spectra.savefig('{:s}/power_spectrum.pdf'.format(str(output_path)))
