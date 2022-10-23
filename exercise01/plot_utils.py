import numpy as np
from casadi import SX, Function

from math import isnan, sqrt, ceil, inf
import matplotlib

import matplotlib.pyplot as plt
from numpy.ma.core import where

def plot_cost(s0s, costs):

    latexify()
    plt.figure()
    plt.plot(s0s, costs, '.')

    plt.grid()
    plt.xlabel(r'$s_1$')
    plt.ylabel(r'$V\left((s_1, 0)\right)$')

def plot_trajectories(s_trajs, a_trajs, labels):

    if not isinstance(s_trajs, list):
        s_trajs = [s_trajs]
        a_trajs = [a_trajs]
        labels = [labels]
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    N = a_trajs[0].size
    ts = np.arange(0, N+1)

    latexify()

    plt.figure(figsize=(5, 8))
    plt.subplot(3, 1, 1)
    for i, x_traj in enumerate(s_trajs):
        plt.plot(ts, x_traj[0, :].T, '-', alpha=0.7, color=colors[i], label=labels[i])
    plt.ylabel(r'$\theta_k$')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 2)
    for i, x_traj in enumerate(s_trajs):
        plt.plot(ts, x_traj[1, :].T, '-', alpha=0.7, color=colors[i], label=labels[i])
    plt.ylabel(r'$\omega_k$')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 3)
    for i, u_traj in enumerate(a_trajs):
        plt.step(ts[:-1], u_traj.T, alpha=0.7, color=colors[i], label=labels[i], where='post')
    plt.grid()
    plt.xlabel(r'time step $k$')
    plt.ylabel(r'$a_k$')
    plt.legend()

def plot_cost_to_go(J_cost, S1, S2, k, N, ax=None, fig=None, vmin=0, vmax=1000):
    
    pc = ax.pcolormesh(S1, S2, J_cost, shading='nearest', vmin=vmin, vmax=vmax, cmap='inferno_r')

    indx = N-k-1
    if indx == 0 or indx == 5:
        ax.set_ylabel(r'$s_2$') 
    if indx >= 5:
        ax.set_xlabel(r'$s_1$')

    ax.set_title('k = %d' % k)
    if k == N - 1 :
        cbar_ax = fig.add_axes([0.9, 0.25, 0.03, 0.5])
        fig.colorbar(pc, cax=cbar_ax)

def plot_control_maps(u_map_dp, u_map_lqr, S1, S2, k, amin=0, amax=1000, titles=None):
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,3), sharey=True, sharex=True)
    fig.subplots_adjust(right=0.8, hspace=0.1, wspace=0.1) 

    if titles == None:
        titles = ['DP state-to-control map, k = %d' % k, 
                  'LQR state-to-control map']

    pc = axes[0].pcolormesh(S1, S2, u_map_dp, shading='nearest', vmin=amin, vmax=amax, cmap='RdBu')
    pc = axes[1].pcolormesh(S1, S2, u_map_lqr, shading='nearest', vmin=amin, vmax=amax, cmap='RdBu')
    axes[0].set_xlabel(r'$s_1$')
    axes[0].set_ylabel(r'$s_2$'); 
    axes[0].set_title(titles[0])
    axes[1].set_xlabel(r'$s_1$')
    axes[1].set_title(titles[1])

    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(pc, cax=cbar_ax)

def plot_cost_maps(J_dp, J_lqr, S1, S2, k, Jmin=None, Jmax=None, log_scale=False, both=True):
    
    if both:
        ncols = 2
    else:
        ncols = 1

    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(4*ncols,3), sharey=True, sharex=True)
    fig.subplots_adjust(right=0.8, hspace=0.1, wspace=0.1) 

    if Jmin==None:
        Jmin=min(np.nanmin(J_dp), np.nanmin(J_lqr))
    if Jmax == None:
        Jmax=max(np.nanmax(J_dp), np.nanmax(J_lqr))

    if log_scale:
        Jmin = 1e-8
        norm = matplotlib.colors.LogNorm(vmin=Jmin, vmax=Jmax)
    else:
        norm=  matplotlib.colors.Normalize(vmin=Jmin, vmax=Jmax)

    if ncols == 2:
        pc = axes[0].pcolormesh(S1, S2, J_dp, norm=norm, shading='nearest', cmap='inferno_r')
        pc = axes[1].pcolormesh(S1, S2, J_lqr, norm=norm, shading='nearest', cmap='inferno_r')
        axes[0].set_xlabel(r'$s_1$')
        axes[0].set_ylabel(r'$s_2$'); 
        axes[0].set_title('DP cost-to-go, k = %d' % k)

        axes[1].set_xlabel(r'$s_1$')
        axes[1].set_title('LQR cost')
    else:
        pc = axes.pcolormesh(S1, S2, J_dp, norm=norm, shading='nearest', cmap='inferno_r')
        axes.set_xlabel(r'$s_1$')
        axes.set_ylabel(r'$s_2$'); 
        axes.set_title('DP cost-to-go, k = %d' % k)

    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(pc, cax=cbar_ax)

def plot_cost_slices(J_dp, J_lqr, s1_grid, s2_grid, indices, vary_s1=True):

    N = len(indices)
    fig, axes = plt.subplots(nrows=1, ncols=N, figsize=(N*2.5,4), sharex=True, sharey=True)
    cmap = matplotlib.cm.get_cmap('plasma')
    colors = [cmap(i/N) for i in range(N)]

    for i in range(N):
        if vary_s1:
            title = r'$s_2 = %f$' % s2_grid[indices[i]]
            xlabel = r'$s_1$'
            xvals = s1_grid
            yvals_dp = J_dp[:, indices[i]]
            yvals_lqr = J_lqr[:, indices[i]]
        else:
            title = r'$s_1 = %f$' % s1_grid[indices[i]]
            xlabel = r'$s_2$'
            xvals = s2_grid
            yvals_dp = J_dp[indices[i], :]
            yvals_lqr = J_lqr[indices[i], :]

        axes[i].plot(xvals, yvals_dp, '-', color=colors[i], label='DP')
        axes[i].plot(xvals, yvals_lqr, '--', color=colors[i], label='LQR')
        axes[i].grid()
        axes[i].set_title(title)
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel('cost')
        axes[i].legend()


def latexify(fig_width=None, fig_height=None):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    if fig_width is None:
        fig_width = 5  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches


    params = {'backend': 'ps',
              'text.latex.preamble': r"\usepackage{gensymb} \usepackage{amsmath}",
              'axes.labelsize': 10, # fontsize for x and y labels (was 10)
              'axes.titlesize': 10,
              'legend.fontsize': 10, # was 10
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)
