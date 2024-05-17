import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

## increasing font sizes for the figures
plt.rc( 'axes', titlesize=17 ) 
plt.rc( 'axes', labelsize=15 ) 
#plt.rc( 'lines', linewidth=2.2 ) 
plt.rc( 'xtick', labelsize=12 ) 
plt.rc( 'ytick', labelsize=12 )
plt.rc( 'legend',fontsize=12 ) 


def get_successes( counts ):
    '''
    Function that returns the cumulated number of successes by a given episode.
    counts : list of durations of the episodes.
    '''
    result = np.zeros(len(counts))
    result = counts < 200
    return np.cumsum(result)

def plot_random( data, eps, fig_path ):
    plt.figure(layout='tight')
    plt.scatter(eps, data['duration'])
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.savefig( f'{fig_path}/duration.png')
    return

def plot_dqn( data, eps, fig_path ):
    smoothing = 10 # to smooth the rewards
    ep_env_reward = np.convolve(data['ep_env_reward'], np.ones(smoothing)/smoothing, mode='valid')
    ep_aux_reward = np.convolve(data['ep_aux_reward'], np.ones(smoothing)/smoothing, mode='valid')
    ep_loss = np.convolve(data['ep_loss'], np.ones(smoothing)/smoothing, mode='valid')
    successes = get_successes( data['duration'] )
    cumsum_aux_reward = np.cumsum(data['ep_aux_reward'])
    cumsum_env_reward = np.cumsum(data['ep_env_reward'])
    cumsum_reward = cumsum_aux_reward + cumsum_env_reward

    fig, ax = plt.subplots( 1, 2, figsize=(11, 6), layout='tight' )
    ax[0].scatter(eps, data['duration'])
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Duration')
    offset = eps[-1] - successes.iloc[-1]
    ax[1].plot(eps, eps-offset, linestyle='--', color='k', label='Slope 1')
    ax[1].scatter(eps, successes, label='Observed')
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Successes by then')
    plt.savefig( f'{fig_path}/duration.png' )

    fig, ax_r = plt.subplots( 1, 2, figsize=(11, 6), layout='tight' )
    ax_r[0].plot(ep_env_reward, label='Environment')
    ax_r[0].plot(ep_aux_reward, label='Auxiliary')
    ax_r[0].plot(ep_env_reward + ep_aux_reward, label='Total')
    ax_r[0].set_xlabel('Episode')
    ax_r[0].set_ylabel('Episode Reward (smoothed)')
    ax_r[1].plot(cumsum_env_reward, label='Environment')
    ax_r[1].plot(cumsum_aux_reward, label='Auxiliary')
    ax_r[1].plot(cumsum_reward, label='Total')
    ax_r[1].set_xlabel('Episode')
    ax_r[1].set_ylabel('Cumulative Reward')
    ax_r[1].legend()
    plt.savefig( f'{fig_path}/reward.png')

    plt.figure(layout='tight')
    plt.plot(ep_loss)
    plt.xlabel('Episode')
    plt.ylabel('Loss (smoothed)')
    plt.savefig( f'{fig_path}/loss.png')

    return


def plot_dyna( data, eps, fig_path ):
    pass


def gen_plots(run_path, agent):

    fig_path = f'{run_path}/figs' 
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # load the data from the .h5 file
    data = pd.read_hdf(f'{run_path}/metrics.h5', key='data')
    with h5.File(f'{run_path}/metrics.h5', 'r') as f:
        eps = f['eps'][:]
        duration = f['duration'] # duration of the training in mins

    if agent[0:3] == 'dqn':
        plot_dqn( data, eps, fig_path )
    elif agent == 'random':
        plot_random( data, eps, fig_path )
    elif agent == 'dyna':
        plot_dyna( data, eps, fig_path )
    


