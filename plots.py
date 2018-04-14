import numpy as np
import matplotlib.pyplot as plt

with open('reward.txt') as f:
    lines = f.readlines()
    x = [line.split()[0] for line in lines]
    y = [line.split()[1] for line in lines]

fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.set_title("Reward over time")    
ax1.set_xlabel('Time')
ax1.set_ylabel('Reward')

ax1.plot(x,y, c='r', label='the data')

leg = ax1.legend()

plt.show()
