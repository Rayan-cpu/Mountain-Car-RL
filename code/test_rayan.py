import gymnasium as gym
import agents 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

mat = np.ones((10,14))
mat[0,0]=0
plt.imshow(mat, cmap='viridis', aspect='auto')
plt.scatter(-0.5,-0.5)
plt.scatter(0,0)
plt.scatter(0.5,0.5)
plt.scatter(1,1)
plt.savefig("hey.png")