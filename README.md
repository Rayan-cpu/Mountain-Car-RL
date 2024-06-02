# Read me 

## Git repo problem
We decided to work with git to deal with merges easily. Everything went smoothly untill yesterday, when we faced a problem which we could not solve. Some files were currupted doing one of Rayan's pushes and Tara was from then on not able to pull anymore. We managed to work around this as there was little left to do, but thought it was worth mentioning. Some things are therefore a bit more wanky than if everything went according to plan. For example, the conda requirements were not tested with the last version of the code (as Rayan uses pip). 

## Set up Conda environement
Use conda env create --name myenv -f requirements.yml to create the associated environment on a new machine. If you just want to update to the new version of the environment, use conda env update -f requirements.yml --prune.  

## Run training run
All the configuration files are found in the ..\configs folder. Run the python main.py -f ..\configs\<config-name>.yml command from the \code folder to start the training run. A copy of used configuration file along with the results (training data and figures) will be stored in the in the ..\runs\<run_name> folder. The <run_name> here will be given from the parameters associated to the run and should therefore be easy to identify.  
Note that the configuration files do not represent an exhaustive list of the training runs we made. 

## Generating all the results 
This can be done in the generate_results.ipynb file. Open it up for more instructions in case you were interested. Notice that running the whole file takes more than an hour.

## Implementation details
* The utility.py file contains the definitions of the MLP (multi layer perceptron) and ReplayBuffer classes. These are used as attributes for the DQN agent classes. 
* The agents.py file contains the definitions of the different agents which can be used. We have a base abstract class Agent which is used to declare the methods which should be provided for every agent (no instances can be created if these are not provided). Then, we have RandomAgent, DQNAgent and DynaAgent as sub-classes, which inherit from Agent. The analyse.py file then takes care to plot the relevant quantities for each of the agents, by calling the right functions