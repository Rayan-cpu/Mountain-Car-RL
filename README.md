# Read me 

## Git repo problem
We decided to work with `git` to avoid 

## Set up Conda environement
Use `conda env export > requirements.yml` to update the requirements and `conda env create --name myenv -f requirements.yml` to create the associated environment on a new machine. I you just want to update to the new version of the environment, use `conda env update -f requirements.yml --prune`.  

## Run training run
Modify a configuration file from the ones in the `..\configs` folder. Use the file whose name corresponds to the agent you want to use. Input the associated parameters for the run. Run the `python main.py -f ..\configs\<config-name>.yml` command from the `\code` folder. A copy of used configuration file along with the results (training metrics and figures) will be stored in the in the `..\runs\<run_name>` folder. The `<run_name>` here will be given from the parameters associated to the run and should therefore be easy to identify.  
 
## Implementation details
* The `utility.py` file contains the definitions of the `MLP` (multi layer perceptron) and `ReplayBuffer` classes. These are used as attributes for the `DQN` agent classes. 
* The `agents.py` file contains the definitions of the different agents which can be used. We have a base abstract class `Agent` which is used to declare the methods which should be provided for every agent (no instances can be created if these are not provided). Then, we have `RandomAgent`, `DQNAgent` and `DynaAgent` as sub-classes, which inherit from `Agent`. The `analyse.py` file then takes care to plot the relevant quantities for each of the agents, by calling the right functions.  
  

## Open questions 
* how to deal with terminal states ? (hint from pdf)
  * currently : we stop as soon as s(t+1) is terminal, so we never play from terminal 
  * problem would only be present if continued playing, no ? 
    * as would be creating state-action which make no sense ? (of which reward would have to be 0)
  * maybe problem if we store it in replay buffer ??
* Loss looks silly ?? (to low at the start ??)


