import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

## increasing font sizes for the figures
fs_fact = 1.35
plt.rc( 'axes', titlesize=fs_fact*17 ) 
plt.rc( 'axes', labelsize=fs_fact*15 ) 
#plt.rc( 'lines', linewidth=2.2 ) 
plt.rc( 'xtick', labelsize=fs_fact*12 ) 
plt.rc( 'ytick', labelsize=fs_fact*12 )
plt.rc( 'legend',fontsize=fs_fact*12 ) 

def transform_range(values, a, b, c, d):
    new_values = np.zeros(len(values))
    for i,value in enumerate(values): 
        scale = (d - c) / (b - a)
        new_values[i] = c + (value - a) * scale
    return new_values

def transform_range(values, a, b, c, d):
    new_values = np.zeros(len(values))
    for i,value in enumerate(values): 
        scale = (d - c) / (b - a)
        new_values[i] = c + (value - a) * scale
    return new_values

def round_to_three_significant_digits(number):
    rounded_number = np.round(1000*number)/1000
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

    fig, ax = plt.subplots( 1, 3, figsize=(16, 4.5), layout='tight' )
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

    fig, ax = plt.subplots( 1, 2, figsize=(11, 4.5), layout='tight' )

    ax[0].scatter(eps, data['duration'])
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Duration')
    offset = eps[-1] - successes.iloc[-1]
    ax[1].plot(eps, eps-offset, linestyle='--', color='k', label='Slope 1')
    ax[1].scatter(eps, successes, label='Observed')
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Successes by then')
    plt.savefig( f'{fig_path}/duration.png' )

    fig, ax_r = plt.subplots( 1, 2, figsize=(11, 4.5), layout='tight' )
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

    fig, ax = plt.subplots( 1, 2, figsize=(11, 4.5), layout='tight' )
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

    fig, ax = plt.subplots( 1, 2, figsize=(11, 4.5), layout='tight' )
    marker = ['.', '>','<']
    for i,ss_factor_ in enumerate(size_factors):
        run_path = f'{run_dir}/dyna-k={k}-ss_coef={ss_factor_}_forcomparison'
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



def plot_dyna( data, eps, fig_path,characteristic_trajectory_2,characteristic_trajectory_3,characteristic_trajectory_4):
    successes = get_successes( data['duration'] )

    fig, ax = plt.subplots( 1, 2, figsize=(11, 4.5), layout='tight' )
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
    smoothing_Q_values = 30 # to smooth the q-values
    Q_values_changes_smooth = np.convolve(data['ep_Q_values_change'], np.ones(smoothing_Q_values)/smoothing_Q_values, mode='valid')
    plt.figure(figsize=(11, 4.5),layout='tight')
    plt.plot(Q_values_changes_smooth)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Episode')
    plt.ylabel(r'$\Delta Q$')
    plt.savefig( f'{fig_path}/log_Q_value_changes.png' )
    

    # the normal Q values
    smoothing_Q_values = 20 # to smooth the q-values
    Q_values_changes_smooth = np.convolve(data['ep_Q_values_change'], np.ones(smoothing_Q_values)/smoothing_Q_values, mode='valid')
    plt.figure(figsize=(11, 4.5),layout='tight')
    plt.plot(Q_values_changes_smooth)
    plt.xlabel('Episode')
    plt.ylabel(r'$\Delta Q$')
    plt.savefig( f'{fig_path}/Q_value_changes.png' )

    # draw the total reward per episode ! !!!!
    smoothing_reward = 20 # to smooth the reward
    avg_reward_changes_smooth = np.convolve(data['ep_env_reward'], np.ones(smoothing_reward)/smoothing_reward, mode='valid')
    plt.figure(figsize=(11, 4.5),layout='tight')
    plt.plot(avg_reward_changes_smooth)
    plt.xlabel('Episode')
    plt.ylabel(r'Reward per episode')
    plt.savefig( f'{fig_path}/Rewards.png' )

    # draw the cumulated reward per episode ! 
    cumsum_aux_reward = np.cumsum(data['ep_env_reward'])
    cum_changes_smooth = np.convolve(cumsum_aux_reward, np.ones(smoothing_reward)/smoothing_reward, mode='valid')
    plt.figure(figsize=(11, 4.5),layout='tight')
    plt.plot(cum_changes_smooth)
    plt.xlabel('Episode')
    plt.ylabel(r'Cumulative reward')
    plt.savefig( f'{fig_path}/Cumulated_Rewards.png' )

    # draw both rewards : 
    fig, ax = plt.subplots( 1, 2, figsize=(11, 6), layout='tight' )
    ax[0].plot(avg_reward_changes_smooth)
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel(r'Reward per episode')
    ax[1].plot(cum_changes_smooth)
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel(r'Cumulative reward')
    plt.savefig( f'{fig_path}/reward_both_figs.png' )

    # Plot the four trajectories : 
    x_min = -1.2
    x_max = 0.6
    y_min = -0.07
    y_max = 0.07
    plt.figure(figsize=(6, 4),layout='tight')
    plt.plot(characteristic_trajectory_2[:,0],characteristic_trajectory_2[:,1],color='b',label='1')
    plt.plot(characteristic_trajectory_3[:,0],characteristic_trajectory_3[:,1],color='g',label='2')
    plt.plot(characteristic_trajectory_4[:,0],characteristic_trajectory_4[:,1],color='m',label='3')
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

def plot_additional_dyna(eps,fig_path,pos_axis_plot,vel_axis_plot,characteristic_Q_1,characteristic_Q_2,characteristic_Q_3,final_Q_matrix,characteristic_Count_1,characteristic_Count_2,characteristic_Count_3,Count_matrix,characteristic_trajectory_2,characteristic_trajectory_3,characteristic_trajectory_4):

    # Round tick values to 3 significant digits
    pos_ticks_indices = np.linspace(0, len(pos_axis_plot) - 1, 3).astype(int)
    vel_ticks_indices = np.linspace(0, len(vel_axis_plot) - 1, 3).astype(int)

    # Get the tick values corresponding to the selected indices
    pos_ticks_values = [pos_axis_plot[i] for i in pos_ticks_indices]
    vel_ticks_values = [vel_axis_plot[i] for i in vel_ticks_indices]

    pos_axis_labels = [round_to_three_significant_digits(val) for val in pos_ticks_values]
    vel_axis_labels = [round_to_three_significant_digits(val) for val in vel_ticks_values]

    # Figure Q_value_1 ----------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
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
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
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
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
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
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    # translate positions in -1.2,0.6 to 0, number of positions
    characteristic_trajectory_2_pos=transform_range(characteristic_trajectory_2[:,0],-1.2,0.6,0,len(pos_axis_plot)-1)
    characteristic_trajectory_3_pos=transform_range(characteristic_trajectory_3[:,0],-1.2,0.6,0,len(pos_axis_plot)-1)
    characteristic_trajectory_4_pos=transform_range(characteristic_trajectory_4[:,0],-1.2,0.6,0,len(pos_axis_plot)-1)
    characteristic_trajectory_2_vel=-transform_range(characteristic_trajectory_2[:,1],-0.07,0.07,len(vel_axis_plot)-1,0)
    characteristic_trajectory_3_vel=-transform_range(characteristic_trajectory_3[:,1],-0.07,0.07,len(vel_axis_plot)-1,0)
    characteristic_trajectory_4_vel=-transform_range(characteristic_trajectory_4[:,1],-0.07,0.07,len(vel_axis_plot)-1,0)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    plt.subplots_adjust(wspace=0.3)
    im0 = ax[0].imshow(final_Q_matrix.T, cmap='viridis', aspect='auto')
    fig.colorbar(im0, ax=ax[0]) # Add a colorbar to the first subplot
    ax[0].set_title('Q values')
    ax[0].set_xlabel('position')
    ax[0].set_ylabel('velocity')
    ax[0].set_xticks(ticks=pos_ticks_indices)
    ax[0].set_xticklabels(labels=pos_axis_labels)
    ax[0].set_yticks(ticks=vel_ticks_indices)
    ax[0].set_yticklabels(labels=vel_axis_labels)

    ax[0].scatter(characteristic_trajectory_2_pos,characteristic_trajectory_2_vel+(len(vel_axis_plot)-1)*np.ones(len(characteristic_trajectory_2_vel)),color='b',label='T1')
    ax[0].scatter(characteristic_trajectory_3_pos,characteristic_trajectory_3_vel+(len(vel_axis_plot)-1)*np.ones(len(characteristic_trajectory_3_vel)),color='yellow',label='T2')
    ax[0].scatter(characteristic_trajectory_4_pos,characteristic_trajectory_4_vel+(len(vel_axis_plot)-1)*np.ones(len(characteristic_trajectory_4_vel)),color='m',label='T3')
    limits = transform_range([0.5,-0.5],-1.2,0.6,1,len(pos_axis_plot))
    ax[0].axvline(x=limits[0], color='r', linestyle='--', )
    ax[0].axvline(x=limits[1], color='k', linestyle='--',alpha=0.2)
    ax[0].legend()
    #ax[0].set_xlim([0,len(pos_axis_plot)])
    #ax[0].set_ylim([0,len(vel_axis_plot)])
    #ax[0].set_xlim([min(characteristic_trajectory_2_pos),max(characteristic_trajectory_2_pos)])
    #ax[0].set_ylim([min(characteristic_trajectory_2_vel),max(characteristic_trajectory_2_vel)])
    

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
    #min_value = Count_matrix.min()
    #max_value_60_percent = 0.5 * Count_matrix.max()
    #Count_matrix = np.clip(Count_matrix, min_value, max_value_60_percent)
    #Count_matrix = 1/(max(Count_matrix))*Count_matrix
    im2 = ax[2].imshow(Count_matrix.T, cmap='viridis', aspect='auto')
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

        plot_dyna( data, eps, fig_path,characteristic_trajectory_2,characteristic_trajectory_3,characteristic_trajectory_4)
        plot_additional_dyna(eps,fig_path,pos_axis_plot,vel_axis_plot,characteristic_Q_1,characteristic_Q_2,characteristic_Q_3,final_Q_matrix,characteristic_Count_1,characteristic_Count_2,characteristic_Count_3,Count_matrix,characteristic_trajectory_2,characteristic_trajectory_3,characteristic_trajectory_4)
        # dyna_comparison( [0.55, 1.5,4.5] )
    


