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
w_tag = r'$u_z$'
b_tag = r'$b$'
ω_tag = r'$\omega_y$'
pow = {u_perp_tag: 0, w_tag:0, u_tot_tag:0, b_tag:0, ω_tag:0}
n_t = 0
first_time = None

tasks = ['b midplane c', 'u_x midplane c', 'u_z midplane c', 'ω_y midplane c']

for file in files:
    logger.info('opening {:s}'.format(file))
    with h5py.File(file, mode='r') as f:
        t = np.array(f['scales/sim_time'])
        data = {}
        for i, task in enumerate(tasks):
            data[task] = f['tasks'][task][:]
            kx = f['tasks'][task].dims[1]['kx'][:]

    if not first_time:
        first_time = t[0]

    n_t += t.shape[0]
    for i in range(t.shape[0]):
        u_c = data['u_x midplane c'][i,:,:]
        v_c = 0
        w_c = data['u_z midplane c'][i,:,:]
        b_c = data['b midplane c'][i,:,:]
        ω_c = data['ω_y midplane c'][i,:,:]
        pow[u_perp_tag] += np.abs(u_c*np.conj(u_c) + v_c*np.conj(v_c))
        pow[w_tag] += np.abs(w_c*np.conj(w_c))
        pow[u_tot_tag] += np.abs(u_c*np.conj(u_c)+v_c*np.conj(v_c)+w_c*np.conj(w_c))
        pow[b_tag] += np.abs(b_c*np.conj(b_c))
        pow[ω_tag] += np.abs(ω_c*np.conj(ω_c))

last_time = t[-1]

for q in pow:
    pow[q] /= n_t

logger.info("power spectra accumulated from t={:.3g}--{:.3g} ({:.3g} total)".format(first_time, last_time, last_time-first_time))
n_kx = kx.shape[0]

kx = kx[:,0,0]
fig_spectra, ax_spectra = plt.subplots()
for q in pow:
    ax_spectra.plot(kx, pow[q][:,0,0], label=q)
min_y, max_y = ax_spectra.get_ylim()
ax_spectra.set_ylim(max_y*1e-20,max_y)

norm = np.nanmax(np.nanmean(pow[u_tot_tag], axis=0))

logger.info('powerlaw k^(-5/3) norm is {:.3g}'.format(norm))
#ax_spectra.plot(kx, norm*kx**(-5/3), color='black', linestyle='dashed', label=r'$k_\perp^{-5/3}$')

ax_spectra.legend()
ax_spectra.set_ylabel(r'Horizontal power spectrum ($u_\perp*u_\perp$)')
ax_spectra.set_xlabel(r'$k_\perp$')
ax_spectra.set_yscale('log')
ax_spectra.set_xscale('log')
fig_spectra.savefig('{:s}/power_spectrum.pdf'.format(str(output_path)))
fig_spectra.savefig('{:s}/power_spectrum.png'.format(str(output_path)), dpi=300)
