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

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from docopt import docopt
args = docopt(__doc__)
files = args['<file>']

if args['--output'] is not None:
    output_path = pathlib.Path(args['--output']).absolute()
else:
    data_dir = args['<file>'][0].split('/')[0]
    data_dir += '/'
    output_path = pathlib.Path(data_dir).absolute()

fig_spectra, ax_spectra = plt.subplots()
fig_spectra2d, ax_spectra2d = plt.subplots(ncols=2, figsize=[6,3])

for file in files:
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

    print(data['u midplane c'].shape)
    u_c = data['u midplane c'][0,:,:]
    u = data['u midplane'][0,:,:]
    v_c = data['v midplane c'][0,:,:]
    v = data['v midplane'][0,:,:]
    w = data['w midplane'][0,:,:]
    pow = np.abs(u_c*np.conj(u_c)+v_c*np.conj(v_c))
    kx_g, ky_g = np.meshgrid(kx, ky)
    x_g, y_g = np.meshgrid(x, y)

    ax_spectra2d[0].pcolormesh(kx, ky, np.log(pow).T, shading='flat')
    ax_spectra2d[0].set_xlabel(r'$k_x$')
    ax_spectra2d[0].set_ylabel(r'$k_y$')
    ax_spectra2d[1].pcolormesh(x, y, w.T, shading='flat')
    ax_spectra2d[1].quiver(x, y, u.T, v.T, units='xy', scale=0.5)

    ax_spectra2d[1].set_xlabel(r'$x$')
    ax_spectra2d[1].set_ylabel(r'$y$')
    fig_spectra2d.savefig('{:s}/power_spectrum_2d.png'.format(str(output_path)), dpi=300)

    avg_spectra = np.mean(pow, axis=1)
    ax_spectra.plot(kx, avg_spectra)
    ax_spectra.set_ylabel('Power spectrum (u*u)')
    ax_spectra.set_xlabel(r'$k_\perp$')
    ax_spectra.set_yscale('log')
    ax_spectra.set_xscale('log')
    fig_spectra.savefig('{:s}/power_spectrum.pdf'.format(str(output_path)))
