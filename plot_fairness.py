import csv
from fairness_metric import Fairness
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

fair = Fairness()

num_agents = 4
prices=[deque(maxlen=600) for _ in range(num_agents)]
plot_prices=[[] for _ in range(num_agents)]
csv_file = ['Group-1 prices.csv','Group-2 prices.csv','Group-3 prices.csv','Group-4 prices.csv']
plot_fairness = []
for i in range(num_agents):
    j=1
    with open(csv_file[i], mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            f=float(row[0])
            prices[i].append(f)
            j+=1
            if j%600==0:
                plot_prices[i].append(np.mean(prices[i]))
            
for i in range(len(plot_prices[0])):
    plot_fairness.append(fair.jain_index([plot_prices[0][i],plot_prices[1][i],plot_prices[2][i],plot_prices[3][i]]))            

                             
fig, axs = plt.subplots(5, 1, figsize=(10, 8))

# Plot data in each subplot
axs[0].plot(plot_fairness, color='magenta', label='Fairness Measure')
axs[0].set_title('Fairness Reward')
axs[0].legend()

axs[1].plot(plot_prices[0], color='red', label='Inv-Sigmoid Demand')
axs[1].set_title('Group 1')
axs[1].legend()

axs[2].plot(plot_prices[1], color='blue', label='Sigmoid Demand')
axs[2].set_title('Group 2')
axs[2].legend()


axs[3].plot(plot_prices[2], color='green', label='Price-Inelastic Demand')
axs[3].set_title('Group 3')
axs[3].legend()

axs[4].plot(plot_prices[3], color='orange', label='Log-Linear Demand')
axs[4].set_title('Group 4')
axs[4].legend()


# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot as an image file
plt.savefig('Rewards vs Episode.png')