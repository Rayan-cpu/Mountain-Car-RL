import random
import torch 
import torch.nn as nn
import yaml

class MLP( nn.Module ):
    '''
    Multi-layer perceptron class. It is a simple feed-forward neural network with 2 hidden layers of size 64 and ReLU activation functions.
    '''
    def __init__( self, in_dim, out_dim ):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear( in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward( self, x ):
        logits = self.linear_relu_stack( x )
        return logits

class ReplayBuffer():
    '''
    Class for the replay buffer. It stores the samples in a tensor and provides methods to easily update its content and generate random batches.
    '''
    def __init__(self, buffer_len):
        self.values = torch.zeros( [buffer_len, 6] ) # (x,v, action, reward, x',v')
        self.len = buffer_len
        self.state_mean = torch.tensor([0.0, 0.0])
        self.state_var = torch.tensor([1.0, 1.0])
        self.index = 0
    
    def update(self, state, action, next_state, reward):
        state_tensor = torch.tensor(state)
        sample = torch.cat( [state_tensor, torch.tensor([action]), torch.tensor([reward]) , torch.tensor(next_state)] )

        if self.index < self.len:
            self.values[self.index,:] = sample
            self.state_mean = (self.state_mean * self.index + state_tensor) / (self.index + 1)
            self.state_var = (self.state_var * self.index + (state_tensor - self.state_mean) ** 2) / (self.index + 1)
            self.index += 1
            return
        
        self.state_mean += (state_tensor - self.values[0,0:2]) / self.len
        self.state_var += ( (state_tensor - self.state_mean) ** 2 - (self.values[0,0:2] - self.state_mean) ** 2 ) / self.len
        self.values = torch.roll( self.values, -1, dims=0 ) # replace oldest sample 
        self.values[-1,:] = sample   
        return
    
    def new_batch( self, batch_size ):
        batch_ind = random.sample( range(self.len), batch_size ) 
        batch = self.values[batch_ind,:]
        return batch
    
def conda_to_pip_requirements(yaml_file, requirements_file):
    with open(yaml_file, 'r') as stream:
        try:
            env_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return

    dependencies = env_data.get('dependencies', [])
    pip_deps = []
    for dep in dependencies:
        if isinstance(dep, dict) and 'pip' in dep:
            pip_deps.extend(dep['pip'])
        elif isinstance(dep, str):
            pip_deps.append(dep)

    with open(requirements_file, 'w') as req_file:
        for dep in pip_deps:
            req_file.write(dep + '\n')

import json
import os



if __name__ == '__main__':
    #conda_to_pip_requirements('../requirements.yml', '../requirements.txt')
# Load the conda environment file
    file_path = '../requirements.yml'
    with open(file_path, 'r') as file:
        conda_env = file.read()

    # Parse the dependencies
    dependencies = conda_env.split('dependencies:')[-1].strip().split('\n')
    formatted_dependencies = []

    for dep in dependencies:

        if dep.startswith('- ') or dep.startswith('  - '):
            if dep.startswith('- '):
                dep = dep[2:].strip()
            elif dep.startswith('  -'):
                dep = dep[4:].strip()
            if '=' in dep:
                parts = dep.split('=')
                if len(parts) >= 2:
                    package_name = parts[0].strip()
                    package_version = parts[1].strip()
                    formatted_dependencies.append(f"{package_name}=={package_version}")
       # print(dep)

    # Write the formatted dependencies to requirements.txt
    requirements_file_path = '../requirements.txt'
    with open(requirements_file_path, 'w') as file:
        for dep in formatted_dependencies:
            file.write(f"{dep}\n")
