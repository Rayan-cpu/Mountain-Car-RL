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
    ep_loss = np.convolve(ep_loss, np.ones(smoothing)/smoothing, mode='valid')
    successes = get_successes( data['duration'] )

    fig, ax = plt.subplots( 2, 1, figsize=(6, 10), sharex=True, layout='tight' )
    ax[0].scatter(eps, data['duration'])
    ax[0].set_ylabel('Duration')
    ax[1].scatter(eps, successes)
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Successes by then')
    plt.savefig( f'{fig_path}/duration.png' )

    plt.figure(layout='tight')
    plt.plot(ep_env_reward, label='Environment')
    plt.plot(ep_aux_reward, label='Auxiliary')
    plt.plot(ep_env_reward + ep_aux_reward, label='Total')
    plt.xlabel('Episode')
    plt.ylabel('Normalised Reward (smoothed)')
    plt.legend()
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
    


