# Read me 
- available actions : TA "not needeed due to only consider environements with constant number of moves "

## Set up Conda environement
Use `conda env export > requirements.yml` to update the requirements and `conda env create -f requirements.yml` to create the associated environment on a new machine. I you just want to update to the new version of the environment, use `conda env update -f requirements.yml --prune`.  

## Implementation details 
* handling gradients 
  * they are set to zero after each update
  * we use no-grad to only include part of the loss in the differenciation 
* dataloaders are usually used for training (to shuffle data and create batches)
  * can't add samples to it, and should not re-define it from scratch 
    * we just keep a list instead, sampling with random indices 
* we only start training once the replay buffer is full 
* 

## Open questions 
* how to deal with terminal states ? (hint from pdf)
  * currently : we stop as soon as s(t+1) is terminal, so we never play from terminal 
  * problem would only be present if continued playing, no ? 
    * as would be creating state-action which make no sense ? (of which reward would have to be 0)
* currently we update the target after each GD step (i.e. we use the same network, just using no-grad), but in class had target which only updated every C GD steps
  * should we do as in class ? 
  * if yes, does that require a deep copy ? (to avoid sharing the gradients and the optimiser ?)
*  "run for 1000 episodes and report its loss and average cumulative reward per episode"
   *  loss per episode over the episodes ? (to track reduction ?)
      *  this would require to have access to weather the episode is done or not : can we add a "bool done" argument to the update() function to compute the average over the episode ? -> otherwise can add "return_loss argument" and deal with computations in the episode loop
   *  average cumulative reward per episode -> as in scalar : the average over the episodes ? (to answer "do we on avg. get to the 100 ?") 

