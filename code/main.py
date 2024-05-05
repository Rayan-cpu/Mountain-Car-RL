import gymnasium as gym
import agents 
import numpy as np
import h5py as h5
import time
import yaml
import argparse
import os
import shutil
import analyse # to generate the plots

def init_agent( configs ):
    agent_name = configs['General']['agent_type']
    if agent_name == 'random':
        return agents.RandomAgent()
    elif agent_name == 'dqn_heuristic':
        cf = configs['Heuristic']
        return agents.DQNAgentHeuristic( degree=cf['degree'], frac=cf['reward_scale'], update_period=configs['DQN']['update_tau'] )
    elif agent_name == 'dqn_rnd':
        cf = configs['RND']

        return agents.DQNAgentRND( reward_factor=cf['reward_factor'],update_period=configs['DQN']['update_tau'] )
    else:
        raise ValueError(f'Agent {agent_name} not found')

def main(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    runs_dir = config['Files']['out_dir']
    run_dir = config['Files']['out_file']
    
    run_path = f'{runs_dir}/{run_dir}' 
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    shutil.copy(config_file, f'{run_path}/config.yml')

    agent = init_agent( config )
    env = gym.make('MountainCar-v0')
    #env = gym.make('MountainCar-v0', render_mode='rgb_array')
    #agent = agents.DQNAgentHeuristic( degree=3, frac=1.e-1, update_period=3 )
    #agent = agents.DQNAgentRND( reward_factor=1.0, pre_train_steps=1000 )

    n_eps = config['General']['n_eps']
    sampling = n_eps // 20
    count = np.zeros(n_eps)
    norm_ep_env_r = np.zeros(n_eps)
    norm_ep_aux_r = np.zeros(n_eps)
    ep_loss = np.zeros(n_eps)

    print(f'Starting to train ...')
    start = time.time()
    for i in range(n_eps):
        count[i], norm_ep_env_r[i], norm_ep_aux_r[i], ep_loss[i] = agent.run_episode(env)
        if i % sampling == 0:
            print(f'{i/n_eps*100:.1f} % of episodes done')
    end = time.time()
    duration = end - start
    print(f'Training took: {(end-start)/60:.3} min')

    with h5.File(f'{run_path}/{'metrics'}.h5', 'w') as f:
        f.create_dataset('eps', data=range(n_eps))
        f.create_dataset('count', data=count)
        f.create_dataset('norm_ep_env_r', data=norm_ep_env_r)
        f.create_dataset('norm_ep_aux_r', data=norm_ep_aux_r)
        f.create_dataset('ep_loss', data=ep_loss)
        f.create_dataset('duration', data=duration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("-f", "--config-file", type=str, help="Path to the configuration file", required=True)
    args = parser.parse_args()

    main(args.config_file)
    print(f'Starting to plot ...')
    analyse.gen_plots(args.config_file) # generate the plots
    print(f'Done plotting !')


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
