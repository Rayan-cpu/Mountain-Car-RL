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
import analyse # to generate the plots

print('Running')

def init_agent( configs ):
    runs_dir = configs['Files']['runs_dir'] # la ou on conserve les resultas
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

def main(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file) # lire le config file  

    agent, run_path, bool_dyna = init_agent( config ) 
    env = gym.make('MountainCar-v0')

    # si le path nexiste pas alors cree un folder 
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    # copie moi le fichier config dans le fichier des resultats. 
    shutil.copy(config_file, f'{run_path}/config.yml')

    n_eps = config['General']['n_episodes']   # number of episodes to rum
    sampling = n_eps // 10   # //k : we will have an update every 1/k fraction of completion
    results = []   # table to store the outcome of each episode resutls
    additional_results = {}

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

    if config['General']['agent_type'][:3] == 'dqn':
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
    dqn_heuristic = agents.DQNAgentHeuristic( load_from='runs/dqn_heuristic/up-tau=3_d=2_frac=0.01/trained_model')
    dqn_vanilla = agents.DQNVanilla( load_from='runs/dqn_vanilla/up-tau=3/trained_model')
    #dqn_rnd = agents.DQNAgentRND( load_from='runs/dqn_rnd/up-tau=3_r-fact=0.01/trained_model')
    #dyna = agents.DynaAgent( load_from='runs/dyna/dyna-k=5-ss_coef=0.1/trained_model')

    env = gym.make('MountainCar-v0')
    seeds = np.arange(n_eps)
    results = np.zeros((n_eps, 3))

    for i in range(n_eps):
        env.reset( seed=seeds[i] )
        results[i, 0] = dqn_heuristic.run_episode(env)['duration']
        env.reset( seed=seeds[i] )
        results[i, 1] = dqn_vanilla.run_episode(env)['duration']
        #results[i, 1] = dqn_rnd.run_episode(env)['duration']
        #env.reset( seed=seeds[i] )
        #results[i, 2] = dyna.run_episode(env)['duration']

    pass

if __name__ == '__main__':
    # faire arriver les arugments du config 
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('-f', '--config-file', type=str, help='Path to the configuration file', required=True)
    args = parser.parse_args()
    # args.config_file est un nom de file 
    main(args.config_file)



'''
done = False
episode_reward = 0
states = np.zeros((1000, 2))
states[0] = state
count = 0
images = []
while not done:
    action = agent.select_action(state,env) 
    next_state, reward, terminated, truncated, _ = env.step(action)
    agent.observe(state, action, next_state, reward)
    agent.update()

    episode_reward += reward
    state = next_state
    states[count+1] = state

    done = terminated or truncated
    count += 1
    images.append(env.render())
env.close()



import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# plot the images with a .1 second delay

fig = plt.figure()
# updates the data and graph
def update(frame):
    # creating a new graph or updating the graph
    plt.imshow(frame)
 
anim = FuncAnimation(fig, update, frames = images, cache_frame_data=False, interval=50)
plt.show()
'''

'''
fig = plt.figure()
for i in range(len(images)):
    plt.imshow(images[i])
    plt.pause(.1)
    plt.draw()
'''
