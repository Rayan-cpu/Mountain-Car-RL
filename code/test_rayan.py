import gymnasium as gym
import agents 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

matrix = np.random.rand(10, 5)  # Replace with your 2D matrix

# Example tick labels
pos_axis_plot = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']  # Replace with your tick labels
vel_axis_plot = ['v1', 'v2', 'v3', 'v4', 'v5']  # Replace with your tick labels

# Create the plot
plt.imshow(matrix, cmap='viridis', aspect='auto')
plt.colorbar()  # Add a colorbar to the plot

# Set custom ticks and labels
plt.xticks(ticks=np.arange(len(pos_axis_plot)), labels=pos_axis_plot)
plt.yticks(ticks=np.arange(len(vel_axis_plot)), labels=vel_axis_plot)

# Add title and labels
plt.title('2D Matrix Plot with Colorbar')
plt.xlabel('Position Axis')
plt.ylabel('Velocity Axis')

# Show the plot
plt.show()