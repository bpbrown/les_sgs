"""
Plot scalar outputs from scalar output .h5 file.
Assumes all scalar output is in a single file.

Usage:
    plot_scalar.py <file> [options]

Options:
    --times=<times>      Range of times to plot over; pass as a comma separated list with t_min,t_max.  Default is whole timespan.
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
file = args['<file>']

if args['--output'] is not None:
    output_path = pathlib.Path(args['--output']).absolute()
else:
    data_dir = args['<file>'].split('/')[0]
    data_dir += '/'
    output_path = pathlib.Path(data_dir).absolute()

fig_KE, ax_KE = plt.subplots(nrows=2)
fig_Re, ax_Re = plt.subplots(nrows=2)
fig_ens, ax_ens = plt.subplots(nrows=2)
fig_Nu, ax_Nu = plt.subplots(nrows=2)

f = h5py.File(file, 'r')
data = {}
t = f['scales/sim_time'][:]
for key in f['tasks']:
    data[key] = f['tasks/'+key][:,0,0,0]
f.close()


if args['--times']:
    subrange = True
    t_min, t_max = args['--times'].split(',')
    t_min = float(t_min)
    t_max = float(t_max)
    print("plotting over range {:g}--{:g}, data range {:g}--{:g}".format(t_min, t_max, min(t), max(t)))
else:
    subrange = False

ax_KE[0].plot(t, data['KE'], label='KE')
ax_KE[1].plot(t, data['KE'], label='KE')
ax_KE[1].set_xlabel('t')
ax_KE[1].set_ylabel('KE')
ax_KE[1].set_yscale('log')

ax_Re[0].plot(t, data['Re'], label='Re')
ax_Re[1].plot(t, data['Re'], label='Re')
ax_Re[1].set_xlabel('t')
ax_Re[1].set_ylabel('Re')
ax_Re[1].set_yscale('log')

ax_ens[0].plot(t, data['enstrophy'], label='ens')
ax_ens[1].plot(t, data['enstrophy'], label='ens')
ax_ens[1].set_xlabel('t')
ax_ens[1].set_ylabel(r'$\omega^2$')
ax_ens[1].set_yscale('log')

fig_KE.savefig('{:s}/energies.pdf'.format(str(output_path)))
fig_Re.savefig('{:s}/Reynolds.pdf'.format(str(output_path)))
fig_ens.savefig('{:s}/enstrophy.pdf'.format(str(output_path)))

if 'Nu' in data:
    ax_ens[0].plot(t, data['enstrophy'], label='ens')
    ax_ens[1].plot(t, data['enstrophy'], label='ens')
    ax_ens[1].set_xlabel('t')
    ax_ens[1].set_ylabel(r'$\omega^2$')
    ax_ens[1].set_yscale('log')
    fig_Nu.savefig('{:s}/Nusselt.pdf'.format(str(output_path)))
