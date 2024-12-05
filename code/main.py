import gymnasium as gym
import agents 
import pandas as pd
import h5py as h5
import time
import yaml
import argparse
import numpy as np
import os
import shutil
import seaborn
import analyse 
import matplotlib.pyplot as plt

def get_successes( counts ):
    '''
    Function returning cumulated number of successes by a given episode.
    counts : list of durations of the episodes.
    '''
    result = np.zeros(len(counts))
    result = counts < 200
    return np.cumsum(result)

def init_agent( configs ):
    '''
    Initialize the agent and the path to save the results according to provided config file.
    configs : dictionary containing the configuration parameters.
    '''
    runs_dir = configs['Files']['runs_dir']
    agent_name = configs['General']['agent_type']
    run_dir = f'{runs_dir}/{agent_name}'
    run_path = ''
    
    bool_dyna = False
    if agent_name == 'random':
        just_for_syntax = configs['General']['n_episodes']
        run_path = f'{run_dir}/n_eps={just_for_syntax}'
        return agents.RandomAgent(), run_path, bool_dyna 
    elif agent_name == 'dqn_vanilla':
        up_tau = configs['DQN']['Qs_NN_update_period']
        run_path = f'{run_dir}/up-tau={up_tau}'
        return agents.DQNVanilla( update_period=up_tau ), run_path, bool_dyna
    elif agent_name == 'dqn_heuristic':
        up_tau = configs['DQN']['Qs_NN_update_period']
        degree = configs['Heuristic']['degree']
        frac = configs['Heuristic']['reward_scale']
        run_path = f'{run_dir}/up-tau={up_tau}_d={degree}_frac={frac}'
        return agents.DQNAgentHeuristic( degree=degree, frac=frac, update_period=up_tau ), run_path,bool_dyna
    elif agent_name == 'dqn_rnd':
        up_tau = configs['DQN']['Qs_NN_update_period']
        reward_factor = configs['RND']['reward_factor']
        run_path = f'{run_dir}/up-tau={up_tau}_r-fact={reward_factor}'
        return agents.DQNAgentRND( reward_factor=reward_factor,update_period=configs['DQN']['Qs_NN_update_period'] ), run_path,bool_dyna
    elif agent_name == 'dyna':
        bool_dyna = True
        k_value = configs['Dyna']['k']
        step_size_coef = configs['Dyna']['step_size_coef']
        x_step= step_size_coef*0.025
        v_step= step_size_coef*0.005
        run_path = f'{run_dir}/dyna-k={k_value}-ss_coef={step_size_coef}'
        return agents.DynaAgent(k = k_value,x_step=x_step,v_step=v_step,load_from=None), run_path, bool_dyna
    else:
        raise ValueError(f'Agent {agent_name} not found')

def main(config_file, colab):
    '''
    Generate the results of the training of the agent according to the provided config file.
    config_file : path to the configuration file.
    colab : boolean indicating if the code is run on google colab.
    '''
    print('Running')
    
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)  
    agent, run_path, bool_dyna = init_agent( config ) 
    
    if colab:
        run_path = f'rl-project-Rayan-Tara/code/{run_path}'
    
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    shutil.copy(config_file, f'{run_path}/config.yml')

    n_eps = config['General']['n_episodes']
    sampling = n_eps // 10   # we give an update every n_eps // 10 episodes
    results = []   # table to store the outcome of each episode results
    additional_results = {}
    env = gym.make('MountainCar-v0')

    # to print(l'avancement, the percentage of episodes done !)
    print(f'Starting to train ...')
    start = time.time()
    bool_final = False 
    for i in range(n_eps):
        bool_final = (i == n_eps-1)
        result_at_ep = agent.run_episode(env)
        results.append( result_at_ep ) # append the list of the results with a dictionary 
        if bool_dyna and bool_final: # add the if we are using dyna ! 
            additional_results['final_Q_matrix'], additional_results['pos_axis_plot'], additional_results['vel_axis_plot'], additional_results['Count_matrix'],additional_results['characteristic_trajectory_1'],additional_results['characteristic_trajectory_2'],additional_results['characteristic_trajectory_3'],additional_results['characteristic_trajectory_4'],additional_results['characteristic_Q_1'], additional_results['characteristic_Q_2'], additional_results['characteristic_Q_3'],additional_results['characteristic_Count_1'],additional_results['characteristic_Count_2'],additional_results['characteristic_Count_3'] = agent.end_episode()
        if i % sampling == 0:
            print(f'{i/n_eps*100:.1f} % of episodes done')
    end = time.time()
    duration = end - start
    print(f'Training took: {(end-start)/60:.3} min')

    # save data :
    df = pd.DataFrame(results) # write results to file
    df.to_hdf(f'{run_path}/metrics.h5', key='data', mode='w')

    #if config['General']['agent_type'][:3] == 'dqn':
    agent.save_training(f'{run_path}/trained_model')

    # save additional data in case we are dealing with dyna :
    if bool_dyna:
        additional_data_path = f'{run_path}/Additional_data'
        os.makedirs(additional_data_path, exist_ok=True)
        with h5.File(additional_data_path + '.hdf5', 'w') as file : 
            file.create_dataset('final_Q_matrix',data = additional_results['final_Q_matrix'])
            file.create_dataset('pos_axis_plot',data = additional_results['pos_axis_plot'])
            file.create_dataset('vel_axis_plot',data = additional_results['vel_axis_plot'])
            file.create_dataset('Count_matrix',data = additional_results['Count_matrix'])
            file.create_dataset('characteristic_trajectory_1',data = additional_results['characteristic_trajectory_1'])
            file.create_dataset('characteristic_trajectory_2',data = additional_results['characteristic_trajectory_2'])
            file.create_dataset('characteristic_trajectory_3',data = additional_results['characteristic_trajectory_3'])
            file.create_dataset('characteristic_trajectory_4',data = additional_results['characteristic_trajectory_4'])
            file.create_dataset('characteristic_Q_1',data = additional_results['characteristic_Q_1'])
            file.create_dataset('characteristic_Q_2',data = additional_results['characteristic_Q_2'])
            file.create_dataset('characteristic_Q_3',data = additional_results['characteristic_Q_3'])
            file.create_dataset('characteristic_Count_1',data = additional_results['characteristic_Count_1'])
            file.create_dataset('characteristic_Count_2',data = additional_results['characteristic_Count_2'])
            file.create_dataset('characteristic_Count_3',data = additional_results['characteristic_Count_3'])
        
    # add simulation information to the file
    with h5.File(f'{run_path}/metrics.h5', 'a') as f:
        f.create_dataset('eps', data=range(n_eps))
        f.create_dataset('duration', data=duration)

    print(f'Starting to plot ...')
    analyse.gen_plots(run_path, config['General']['agent_type'])
    print(f'Done plotting !')


def compare_performances( n_eps=1000 ):
    dqn_heuristic = agents.DQNAgentHeuristic( load_from='../runs/dqn_heuristic/up-tau=3_d=2_frac=0.7/trained_model')
    #dqn_vanilla = agents.DQNVanilla( load_from='../runs/dqn_vanilla/up-tau=3/trained_model')
    dqn_rnd = agents.DQNAgentRND( load_from='../runs/dqn_rnd/up-tau=1_r-fact=10.0/trained_model')
    step_size_coef = 1.5
    x_step= step_size_coef*0.025
    v_step= step_size_coef*0.005
    dyna = agents.DynaAgent(x_step=x_step,v_step=v_step,k=3,load_from='../runs/dyna/dyna-k=3-ss_coef=1.5/trained_model')

    env = gym.make('MountainCar-v0')
    seeds = np.arange(n_eps)
    results = np.zeros((n_eps, 3))
    sampling = n_eps // 10
    print('Starting comparison ...')

    for i in range(n_eps):
        seed = int(seeds[i])
        env.reset( seed=seed )
        results[i, 0] = dqn_heuristic.run_episode(env)['duration']
        env.reset( seed=seed )
        #results[i, 1] = dqn_vanilla.run_episode(env)['duration']
        results[i, 1] = dqn_rnd.run_episode(env)['duration']
        env.reset( seed=seed )
        results[i, 2] = dyna.run_episode(env)['duration']
        if i % sampling == 0:
            print(f'{i/n_eps*100:.1f} % of episodes done')
    print('Done with comparison !')

    fig, ax = plt.subplots( 1, 3, figsize=(16, 4.5), layout='tight' )
    marker = ['o', '>','<']

    duration_dqn_heuristic = results[:, 0]
    eps = 1 + np.arange(len(duration_dqn_heuristic))
    ax[1].scatter(eps, duration_dqn_heuristic, s=15, marker=marker[0])
        
    duration_dqn_rnd = results[:, 1]
    eps = 1 + np.arange(len(duration_dqn_rnd))
    ax[1].scatter(eps, duration_dqn_rnd, s=10, marker=marker[1])

    duration_dyna = results[:, 2]
    eps = 1 + np.arange(len(duration_dyna))
    ax[1].scatter(eps, duration_dyna, s=10, marker=marker[2])
  
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Duration')

    ax[2]=seaborn.kdeplot(duration_dqn_heuristic, label='DQN heuristic')
    ax[2]=seaborn.kdeplot(duration_dqn_rnd, label='DQN RND')
    ax[2]=seaborn.kdeplot(duration_dyna, label='Dyna')
    ax[2].set_xlabel('Duration')
    ax[2].set_ylabel('Density')

    # plot of rewards of the interesting simulations !
    data_heuristic = pd.read_hdf('../runs/dqn_heuristic/up-tau=3_d=2_frac=0.7/metrics.h5', key='data')
    data_dqn = pd.read_hdf('../runs/dqn_rnd/up-tau=1_r-fact=10.0/metrics.h5', key='data')
    data_dyna = pd.read_hdf('../runs/dyna/dyna-k=3-ss_coef=1.5/metrics.h5', key='data')
    smoothing = 30
    reward_heuristic = np.convolve(data_heuristic['ep_env_reward'], np.ones(smoothing)/smoothing, mode='valid')
    reward_dqn = np.convolve(data_dqn['ep_env_reward'], np.ones(smoothing)/smoothing, mode='valid')
    reward_dyna = np.convolve(data_dyna['ep_env_reward'], np.ones(smoothing)/smoothing, mode='valid')
    ax[0].plot(reward_heuristic, label='DQN heuristic')
    ax[0].plot(reward_dqn, label='DQN RND')
    ax[0].plot(reward_dyna, label='Dyna')
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Reward')
    ax[0].legend()
    plt.savefig('comparison.png', bbox_inches='tight')

    analyse.heuristic_comparison([0.01,0.7,35.0])

    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config-file', type=str, help='Path to the configuration file', required=True)
    parser.add_argument('-c', '--comparison', type=bool, help='Whether to run the comparison', required=False, default=False)
    args = parser.parse_args()

    main(args.config_file, False)
    if args.comparison:
        compare_performances()

