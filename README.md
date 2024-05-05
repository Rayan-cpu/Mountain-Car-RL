# Read me 
- available actions : TA "not needeed due to only consider environements with constant number of moves "

## Set up Conda environement
Use `conda env export > requirements.yml` to update the requirements and `conda env create -f requirements.yml` to create the associated environment on a new machine. I you just want to update to the new version of the environment, use `conda env update -f requirements.yml --prune`.  

## Implementation details 
* handling gradients 
  * they are set to zero after each update
  * we use target network to only include part of the loss in the differenciation 
* dataloaders are usually used for training (to shuffle data and create batches)
  * can't add samples to it, and should not re-define it from scratch 
    * we just keep a list instead, sampling with random indices 
* we only start training once the replay buffer is full 
* implementation of target net was done with help of chat gpt (command to import the weights from another net)



## Open questions 
* how to deal with terminal states ? (hint from pdf)
  * currently : we stop as soon as s(t+1) is terminal, so we never play from terminal 
  * problem would only be present if continued playing, no ? 
    * as would be creating state-action which make no sense ? (of which reward would have to be 0)
  * maybe problem if we store it in replay buffer ??


