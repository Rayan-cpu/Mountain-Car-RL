import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml


def gen_plots(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    runs_dir = config['Files']['out_dir']
    run_dir = config['Files']['out_file']
    run_path = f'{runs_dir}/{run_dir}'

    fig_path = f'{run_path}/figs' 
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # load the data from the .h5 file
    with h5.File(f'{run_path}/metrics.h5', 'r') as f:
        eps = f['eps'][:]
        count = f['count'][:]
        norm_ep_env_reward = f['norm_ep_env_r'][:]
        norm_ep_aux_reward = f['norm_ep_aux_r'][:]
        ep_loss = f['ep_loss'][:]


    def get_successes( counts ):
        result = np.zeros(len(counts))
        result = counts < 200
        return np.cumsum(result)

    smoothing = 10 # to smooth the rewards
    norm_ep_env_reward = np.convolve(norm_ep_env_reward, np.ones(smoothing)/smoothing, mode='valid')
    norm_ep_aux_reward = np.convolve(norm_ep_aux_reward, np.ones(smoothing)/smoothing, mode='valid')
    ep_loss = np.convolve(ep_loss, np.ones(smoothing)/smoothing, mode='valid')

    fig, ax = plt.subplots( 2, 1, figsize=(6, 10), sharex=True )
    ax[0].scatter(eps, count)
    ax[0].set_ylabel('Duration')
    # plot the cumulative sum of the successes
    successes = get_successes( count )
    ax[1].scatter(range(len(successes)), successes)
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Successes by then')
    plt.savefig( f'{fig_path}/duration.png' )
    plt.figure()
    # will sizes match ?
    plt.plot(norm_ep_env_reward, label='Environment')
    plt.plot(norm_ep_aux_reward, label='Auxiliary')
    plt.plot(norm_ep_env_reward + norm_ep_aux_reward, label='Total')
    plt.xlabel('Episode')
    plt.legend()
    plt.ylabel('Reward')
    plt.savefig( f'{fig_path}/reward.png')
    plt.figure()
    plt.plot(ep_loss)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.savefig( f'{fig_path}/loss.png')
    #plt.show()

