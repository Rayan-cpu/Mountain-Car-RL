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

def round_to_three_significant_digits(number):
    digits_left_of_decimal = int(np.floor(np.log10(abs(number)))) + 1
    factor = 10 ** (3 - digits_left_of_decimal)
    rounded_number = round(number * factor) / factor # rounding
    return rounded_number

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

def plot_vanilla_dqn( data, eps, fig_path ):
    smoothing = 10 
    ep_loss = np.convolve( data['ep_loss'], np.ones(smoothing)/smoothing, mode='valid' )
    ep_env_reward = np.convolve(data['ep_env_reward'], np.ones(smoothing)/smoothing, mode='valid')

    fig, ax = plt.subplots( 1, 3, figsize=(16, 5), layout='tight' )
    ax[0].scatter(eps, data['duration'])
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Duration')

    ax[1].plot(ep_env_reward)
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Episode Reward (smoothed)')

    ax[2].plot(ep_loss)
    ax[2].set_xlabel('Episode')
    ax[2].set_ylabel('Loss (smoothed)')
    plt.savefig( f'{fig_path}/full_results.png')

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

    fig, ax = plt.subplots( 1, 2, figsize=(11, 5), layout='tight' )

    ax[0].scatter(eps, data['duration'])
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Duration')
    offset = eps[-1] - successes.iloc[-1]
    ax[1].plot(eps, eps-offset, linestyle='--', color='k', label='Slope 1')
    ax[1].scatter(eps, successes, label='Observed')
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Successes by then')
    plt.savefig( f'{fig_path}/duration.png' )

    fig, ax_r = plt.subplots( 1, 2, figsize=(11, 5), layout='tight' )
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

def heuristic_comparison( r_factor ):
    degree = 2
    update_tau = 3
    run_dir = f'../runs/dqn_heuristic'
    fig_path = f'{run_dir}'

    fig, ax = plt.subplots( 1, 2, figsize=(11, 5), layout='tight' )
    marker = ['.', '^']
    for i,r_factor_ in enumerate(r_factor):
        run_path = f'{run_dir}/up-tau={update_tau}_d={degree}_frac={r_factor_}'
        data = pd.read_hdf(f'{run_path}/metrics.h5', key='data')
        
        duration = data['duration']
        successes = get_successes( duration )
        eps = 1 + np.arange(len(duration))

        label = r'$\rho'
        label = f'{label}={r_factor_:.2f}$'
        l = ax[1].plot(eps, successes, label=label)
        ax[0].scatter(eps, duration, s=10, marker=marker[i], facecolors='none', edgecolors=l[0].get_color())
  
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Duration')
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Successes by then')
    ax[1].legend()
    plt.savefig( f'{fig_path}/heuristic_comparison.png' )


def dyna_comparison( size_factors ):
    k = 3
    run_dir = f'../runs/dyna'
    fig_path = f'{run_dir}'

    fig, ax = plt.subplots( 1, 2, figsize=(11, 5), layout='tight' )
    marker = ['.', '>','<']
    for i,ss_factor_ in enumerate(size_factors):
        run_path = f'{run_dir}/dyna-k={k}-ss_coef={ss_factor_}'
        data = pd.read_hdf(f'{run_path}/metrics.h5', key='data')
        
        duration = data['duration']
        successes = get_successes( duration )
        eps = 1 + np.arange(len(duration))

        label = f'{ss_factor_:.2f}'
        l = ax[1].plot(eps, successes, label=label)
        ax[0].scatter(eps, duration, s=10, marker=marker[i], facecolors='none', edgecolors=l[0].get_color())
  
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Duration')
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Successes by then')
    ax[1].legend(title=r'Size factor $\alpha$')
    plt.savefig( f'{fig_path}/dyna_comparison.png' )



def plot_dyna( data, eps, fig_path,characteristic_trajectory_1,characteristic_trajectory_2,characteristic_trajectory_3,characteristic_trajectory_4):
    successes = get_successes( data['duration'] )

    fig, ax = plt.subplots( 1, 2, figsize=(11, 5), layout='tight' )
    ax[0].scatter(eps, data['duration'])
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Duration')
    offset = eps[-1] - successes.iloc[-1]
    ax[1].plot(eps, eps-offset, linestyle='--', color='k', label='Slope 1')
    ax[1].scatter(eps, successes, label='Observed')
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Successes by then')
    plt.savefig( f'{fig_path}/duration.png' )

    # the logarithm of the Q values 
    smoothing_Q_values = 20 # to smooth the q-values
    Q_values_changes_smooth = np.convolve(data['ep_Q_values_change'], np.ones(smoothing_Q_values)/smoothing_Q_values, mode='valid')
    plt.figure(figsize=(11, 5),layout='tight')
    plt.plot(Q_values_changes_smooth)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Episode')
    plt.ylabel(r'$\Delta Q$')
    plt.savefig( f'{fig_path}/log_Q_value_changes.png' )

    # the normal Q values
    smoothing_Q_values = 20 # to smooth the q-values
    Q_values_changes_smooth = np.convolve(data['ep_Q_values_change'], np.ones(smoothing_Q_values)/smoothing_Q_values, mode='valid')
    plt.figure(figsize=(11, 5),layout='tight')
    plt.plot(Q_values_changes_smooth)
    plt.xlabel('Episode')
    plt.ylabel(r'$\Delta Q$')
    plt.savefig( f'{fig_path}/Q_value_changes.png' )

    # draw the total reward per episode !
    smoothing_reward = 10 # to smooth the reward
    avg_reward_changes_smooth = np.convolve(data['ep_env_reward'], np.ones(smoothing_reward)/smoothing_reward, mode='valid')
    plt.figure(figsize=(11, 5),layout='tight')
    plt.plot(avg_reward_changes_smooth)
    plt.xlabel('Episode')
    plt.ylabel(r'Reward per episode')
    plt.savefig( f'{fig_path}/Rewards.png' )

    # draw the cumulated reward per episode !
    cumsum_aux_reward = np.cumsum(data['ep_env_reward'])
    smoothing_reward = 10 # to smooth the q-values
    cum_changes_smooth = np.convolve(cumsum_aux_reward, np.ones(smoothing_reward)/smoothing_reward, mode='valid')
    plt.figure(figsize=(11, 5),layout='tight')
    plt.plot(cum_changes_smooth)
    plt.xlabel('Episode')
    plt.ylabel(r'Cumulated rewards per episode')
    plt.savefig( f'{fig_path}/Cumulated_Rewards.png' )


    # Plot the four trajectories : 
    x_min = -1.2
    x_max = 0.6
    y_min = -0.07
    y_max = 0.07
    plt.figure(figsize=(6, 4),layout='tight')
    #plt.plot(characteristic_trajectory_1[:,0],characteristic_trajectory_1[:,1],color='r',label='1')
    plt.plot(characteristic_trajectory_2[:,0],characteristic_trajectory_2[:,1],color='b',label='1')
    plt.plot(characteristic_trajectory_3[:,0],characteristic_trajectory_3[:,1],color='g',label='2')
    plt.plot(characteristic_trajectory_4[:,0],characteristic_trajectory_4[:,1],color='m',label='3')
    #plt.scatter(characteristic_trajectory_1[0,0],characteristic_trajectory_1[0,1],color='r')
    plt.scatter(characteristic_trajectory_2[0,0],characteristic_trajectory_2[0,1],color='b')
    plt.scatter(characteristic_trajectory_3[0,0],characteristic_trajectory_3[0,1],color='g')
    plt.scatter(characteristic_trajectory_4[0,0],characteristic_trajectory_4[0,1],color='m')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.legend(title=r'$Trajectory$')
    plt.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], color='black')
    plt.xlim([x_min-0.1,x_max+0.1])
    plt.ylim([y_min-0.02,y_max+0.02])
    plt.axvline(x=0.5, color='r', linestyle='--', label=f'Terminal state')
    plt.axvline(x=-0.5, color='k', linestyle='--',alpha=0.2, label=f'Terminal state')
    plt.savefig(f'{fig_path}/special_trajectories.png')

    pass

def plot_additional_dyna(eps,fig_path,pos_axis_plot,vel_axis_plot,characteristic_Q_1,characteristic_Q_2,characteristic_Q_3,final_Q_matrix,characteristic_Count_1,characteristic_Count_2,characteristic_Count_3,Count_matrix):

    # Round tick values to 3 significant digits
    pos_ticks_indices = np.linspace(0, len(pos_axis_plot) - 1, 3).astype(int)
    vel_ticks_indices = np.linspace(0, len(vel_axis_plot) - 1, 3).astype(int)

    # Get the tick values corresponding to the selected indices
    pos_ticks_values = [pos_axis_plot[i] for i in pos_ticks_indices]
    vel_ticks_values = [vel_axis_plot[i] for i in vel_ticks_indices]
    pos_axis_labels = [round_to_three_significant_digits(val) for val in pos_ticks_values]
    vel_axis_labels = [round_to_three_significant_digits(val) for val in vel_ticks_values]

    # Figure Q_value_1 ----------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    plt.subplots_adjust(wspace=0.3)
    im0 = ax[0].imshow(characteristic_Q_1.T, cmap='viridis', aspect='auto')
    fig.colorbar(im0, ax=ax[0])  # Add a colorbar to the first subplot
    ax[0].set_title('Q values')
    ax[0].set_xlabel('position')
    ax[0].set_ylabel('velocity')
    ax[0].set_xticks(ticks=pos_ticks_indices)
    ax[0].set_xticklabels(labels=pos_axis_labels)
    ax[0].set_yticks(ticks=vel_ticks_indices)
    ax[0].set_yticklabels(labels=vel_axis_labels)

    characteristic_visited_1=np.where(characteristic_Count_1!=0,1,0)
    im1 = ax[1].imshow(characteristic_visited_1.T, cmap='viridis', aspect='auto')
    fig.colorbar(im1, ax=ax[1])  # Add a colorbar to the second subplot
    ax[1].set_title('Visited states')
    ax[1].set_xlabel('position')
    ax[1].set_xticks(ticks=pos_ticks_indices)
    ax[1].set_xticklabels(labels=pos_axis_labels)
    ax[1].set_yticks(ticks=vel_ticks_indices)
    ax[1].set_yticklabels(labels=vel_axis_labels)

    im2 = ax[2].imshow(characteristic_Count_1.T, cmap='viridis', aspect='auto')
    fig.colorbar(im2, ax=ax[2])  # Add a colorbar to the second subplot
    ax[2].set_title('Count matrix')
    ax[2].set_xlabel('position')
    ax[2].set_xticks(ticks=pos_ticks_indices)
    ax[2].set_xticklabels(labels=pos_axis_labels)
    ax[2].set_yticks(ticks=vel_ticks_indices)
    ax[2].set_yticklabels(labels=vel_axis_labels)
    plt.savefig(f'{fig_path}/Q1_matrix.png')

    # Figure Q_value_2 ----------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    plt.subplots_adjust(wspace=0.3)
    im0 = ax[0].imshow(characteristic_Q_2.T, cmap='viridis', aspect='auto')
    fig.colorbar(im0, ax=ax[0])  # Add a colorbar to the first subplot
    ax[0].set_title('Q values')
    ax[0].set_xlabel('position')
    ax[0].set_ylabel('velocity')
    ax[0].set_xticks(ticks=pos_ticks_indices)
    ax[0].set_xticklabels(labels=pos_axis_labels)
    ax[0].set_yticks(ticks=vel_ticks_indices)
    ax[0].set_yticklabels(labels=vel_axis_labels)

    characteristic_visited_2=np.where(characteristic_Count_2!=0,1,0)
    im1 = ax[1].imshow(characteristic_visited_2.T, cmap='viridis', aspect='auto')
    fig.colorbar(im1, ax=ax[1])  # Add a colorbar to the second subplot
    ax[1].set_title('Visited states')
    ax[1].set_xlabel('position')
    ax[1].set_xticks(ticks=pos_ticks_indices)
    ax[1].set_xticklabels(labels=pos_axis_labels)
    ax[1].set_yticks(ticks=vel_ticks_indices)
    ax[1].set_yticklabels(labels=vel_axis_labels)

    im2 = ax[2].imshow(characteristic_Count_2.T, cmap='viridis', aspect='auto')
    fig.colorbar(im2, ax=ax[2])  # Add a colorbar to the second subplot
    ax[2].set_title('Count matrix')
    ax[2].set_xlabel('position')
    ax[2].set_xticks(ticks=pos_ticks_indices)
    ax[2].set_xticklabels(labels=pos_axis_labels)
    ax[2].set_yticks(ticks=vel_ticks_indices)
    ax[2].set_yticklabels(labels=vel_axis_labels)
    plt.savefig(f'{fig_path}/Q2_matrix.png')

    # Figure Q_value_3 ----------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    plt.subplots_adjust(wspace=0.3)
    im0 = ax[0].imshow(characteristic_Q_3.T, cmap='viridis', aspect='auto')
    fig.colorbar(im0, ax=ax[0])  # Add a colorbar to the first subplot
    ax[0].set_title('Q values')
    ax[0].set_xlabel('position')
    ax[0].set_ylabel('velocity')
    ax[0].set_xticks(ticks=pos_ticks_indices)
    ax[0].set_xticklabels(labels=pos_axis_labels)
    ax[0].set_yticks(ticks=vel_ticks_indices)
    ax[0].set_yticklabels(labels=vel_axis_labels)

    characteristic_visited_3=np.where(characteristic_Count_3!=0,1,0)
    im1 = ax[1].imshow(characteristic_visited_3.T, cmap='viridis', aspect='auto')
    fig.colorbar(im1, ax=ax[1])  # Add a colorbar to the second subplot
    ax[1].set_title('Visited states')
    ax[1].set_xlabel('position')
    ax[1].set_xticks(ticks=pos_ticks_indices)
    ax[1].set_xticklabels(labels=pos_axis_labels)
    ax[1].set_yticks(ticks=vel_ticks_indices)
    ax[1].set_yticklabels(labels=vel_axis_labels)

    im2 = ax[2].imshow(characteristic_Count_3.T, cmap='viridis', aspect='auto')
    fig.colorbar(im2, ax=ax[2])  # Add a colorbar to the second subplot
    ax[2].set_title('Count matrix')
    ax[2].set_xlabel('position')
    ax[2].set_xticks(ticks=pos_ticks_indices)
    ax[2].set_xticklabels(labels=pos_axis_labels)
    ax[2].set_yticks(ticks=vel_ticks_indices)
    ax[2].set_yticklabels(labels=vel_axis_labels)
    plt.savefig(f'{fig_path}/Q3_matrix.png')

    # Figure Q_value_final ----------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    plt.subplots_adjust(wspace=0.3)
    im0 = ax[0].imshow(final_Q_matrix.T, cmap='viridis', aspect='auto')
    fig.colorbar(im0, ax=ax[0])  # Add a colorbar to the first subplot
    ax[0].set_title('Q values')
    ax[0].set_xlabel('position')
    ax[0].set_ylabel('velocity')
    ax[0].set_xticks(ticks=pos_ticks_indices)
    ax[0].set_xticklabels(labels=pos_axis_labels)
    ax[0].set_yticks(ticks=vel_ticks_indices)
    ax[0].set_yticklabels(labels=vel_axis_labels)

    visited_matrix=np.where(Count_matrix!=0,1,0)
    im1 = ax[1].imshow(visited_matrix.T, cmap='viridis', aspect='auto')
    fig.colorbar(im1, ax=ax[1])  # Add a colorbar to the second subplot
    ax[1].set_title('Visited states')
    ax[1].set_xlabel('position')
    ax[1].set_xticks(ticks=pos_ticks_indices)
    ax[1].set_xticklabels(labels=pos_axis_labels)
    ax[1].set_yticks(ticks=vel_ticks_indices)
    ax[1].set_yticklabels(labels=vel_axis_labels)

    # clamp the matrix for visibility : 
    min_value = Count_matrix.min()
    max_value_60_percent = 0.5 * Count_matrix.max()
    Count_matrix_modified = np.clip(Count_matrix, min_value, max_value_60_percent)
    im2 = ax[2].imshow(Count_matrix_modified.T, cmap='viridis', aspect='auto')
    fig.colorbar(im2, ax=ax[2])  # Add a colorbar to the second subplot
    ax[2].set_title('Count matrix')
    ax[2].set_xlabel('position')
    ax[2].set_xticks(ticks=pos_ticks_indices)
    ax[2].set_xticklabels(labels=pos_axis_labels)
    ax[2].set_yticks(ticks=vel_ticks_indices)
    ax[2].set_yticklabels(labels=vel_axis_labels)
    plt.savefig(f'{fig_path}/Q_final_matrix.png')

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
        if agent[4:] == 'vanilla':
            plot_vanilla_dqn( data, eps, fig_path )
        else :
            plot_dqn( data, eps, fig_path )
    elif agent == 'random':
        plot_random( data, eps, fig_path )
    elif agent == 'dyna':
        # import the 4 vectors : 
        with h5.File(f'{run_path}/Additional_data.hdf5', 'r') as f:
            final_Q_matrix = f['final_Q_matrix'][:]
            pos_axis_plot = f['pos_axis_plot'][:]
            vel_axis_plot = f['vel_axis_plot'][:]
            Count_matrix = f['Count_matrix'][:]
            characteristic_trajectory_1 = f['characteristic_trajectory_1'][:]
            characteristic_trajectory_2 = f['characteristic_trajectory_2'][:]
            characteristic_trajectory_3 = f['characteristic_trajectory_3'][:]
            characteristic_trajectory_4 = f['characteristic_trajectory_4'][:]
            characteristic_Q_1 = f['characteristic_Q_1'][:]
            characteristic_Q_2 = f['characteristic_Q_2'][:]
            characteristic_Q_3 = f['characteristic_Q_3'][:]
            characteristic_Count_1 = f['characteristic_Count_1'][:]
            characteristic_Count_2 = f['characteristic_Count_2'][:]
            characteristic_Count_3 = f['characteristic_Count_3'][:]

        plot_dyna( data, eps, fig_path,characteristic_trajectory_1,characteristic_trajectory_2,characteristic_trajectory_3,characteristic_trajectory_4)
        plot_additional_dyna(eps,fig_path,pos_axis_plot,vel_axis_plot,characteristic_Q_1,characteristic_Q_2,characteristic_Q_3,final_Q_matrix,characteristic_Count_1,characteristic_Count_2,characteristic_Count_3,Count_matrix)
        dyna_comparison( [0.55, 1.5,4.5] )
    


