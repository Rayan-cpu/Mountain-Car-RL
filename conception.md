# Conception file 
loss per episode over the samples ? 
would expect reduction 
actually the task is getting harder while we explore more
the loss trend is not expected to be decreasing before the steady state ?
    what is this steady state ?
## Heuristic reward function
$$
f(x) = A\mathbb I[x>\bar x\_0]x^n
$$
for $A>0$ some constant and a given $n\in\mathbb N^\star$. In practice we most often take $n=3$ and $A=0.1$.  
    * eg. if take degree=1 not trivial that we actually learn ??
    * could even have instabilities ??
    * 1.e-1 seems to work well 
    * test 1.e0 and 2.e-2 ?? (sort of dicotomy + prior knowledge search)
    * test the higher reward with low degree func ? 
    * be on the lookout for instabilities 
## RND reward
### Normalisation  
* over what should we normalise ?
    * can use whole buffer or batch 
    * can do it for no extra expense by storing buffer mean and std and using update rule for mean quantities $f\_{n+1}(\vec x)=f\_n(\vec x\_n)+\frac{1}{n}(x\_{n+1}-x\_0)$ 
    * they say not to compute at "few initial steps" but can wait until buffer is full 
        * we wait for the buffer to be full to start training as we want representative input to be given (and buffer length is not huge compared to training time)  
        * we could try and a posteriori compute the RND reward for the initial buffer but it turns out to mess up the gradients as it requires modifying variables that are used for the training of the RND net
* handling gradients 
  * they are set to zero after each update
  * we use target network to only include part of the loss in the differenciation 
* dataloaders are usually used for training (to shuffle data and create batches)
  * can't add samples to it, and should not re-define it from scratch 
  * we just keep a list instead, sampling with random indices 
* we only start training once the replay buffer is full 
* implementation of target net was done with help of chat gpt (command to import the weights from another net)
* adding the auxiliary rewards to the buffer so that the agent can learn from them led to an error when calling backward(). The autodiff notices that the variable changed value (as batch will have changed through the buffer). Since training time >> buffer update time, no solution to this was found.
