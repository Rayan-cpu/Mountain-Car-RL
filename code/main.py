import gymnasium as gym
import agents 
import pandas as pd
import h5py as h5
import time
import yaml
import argparse
import os
import shutil
import analyse # to generate the plots

def init_agent( configs ):
    runs_dir = configs['Files']['runs_dir']
    agent_name = configs['General']['agent_type']
    run_dir = f'{runs_dir}/{agent_name}'
    run_path = ''
    
    if agent_name == 'random':
        run_path = f'{run_dir}/n_eps={configs['General']['n_episodes']}'
        return agents.RandomAgent(), run_path
    elif agent_name == 'dqn_heuristic':
        up_tau = configs['DQN']['Qs_NN_update_period']
        degree = configs['Heuristic']['degree']
        frac = configs['Heuristic']['reward_scale']
        run_path = f'{run_dir}/up-tau={up_tau}_d={degree}_frac={frac}'
        return agents.DQNAgentHeuristic( degree=degree, frac=frac, update_period=up_tau ), run_path
    elif agent_name == 'dqn_rnd':
        up_tau = configs['DQN']['Qs_NN_update_period']
        reward_factor = configs['RND']['reward_factor']
        run_path = f'{run_dir}/up-tau={up_tau}_r-fact={reward_factor}'
        return agents.DQNAgentRND( reward_factor=reward_factor,update_period=configs['DQN']['Qs_NN_update_period'] ), run_path
    else:
        raise ValueError(f'Agent {agent_name} not found')

def main(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    agent, run_path = init_agent( config )
    env = gym.make('MountainCar-v0')

    #runs_dir = config['Files']['runs_dir']
    #run_path = f'{runs_dir}/{run_name}' 
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    shutil.copy(config_file, f'{run_path}/config.yml')

    n_eps = config['General']['n_episodes']
    sampling = n_eps // 20
    results = []

    print(f'Starting to train ...')
    start = time.time()
    for i in range(n_eps):
        results.append( agent.run_episode(env) )
        if i % sampling == 0:
            print(f'{i/n_eps*100:.1f} % of episodes done')
    end = time.time()
    duration = end - start
    print(f'Training took: {(end-start)/60:.3} min')

    df = pd.DataFrame(results) # write results to file
    df.to_hdf(f'{run_path}/{'metrics'}.h5', key='data', mode='w') 

    # add simulation information to the file
    with h5.File(f'{run_path}/{'metrics'}.h5', 'a') as f:
        f.create_dataset('eps', data=range(n_eps))
        f.create_dataset('duration', data=duration)

    print(f'Starting to plot ...')
    analyse.gen_plots(run_path, config['General']['agent_type'])
    print(f'Done plotting !')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('-f', '--config-file', type=str, help='Path to the configuration file', required=True)
    args = parser.parse_args()

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
