import sys
import os
import numpy as np
import matplotlib.pyplot as plt

return_dir_list = [
    sys.path[0] + '/models/dqn/LunarLander-v2',
    sys.path[0] + '/models/ddpg/LunarLanderContinuous-v2',
    sys.path[0] + '/models/ppo/LunarLanderContinuous-v2',
    sys.path[0] + '/models/ppo_plus_ent/LunarLanderContinuous-v2'
]
names = ['dqn', 'ddpg', 'ppo', 'ppo_entropy']
colors = ['limegreen', 'lightcoral', 'purple', 'violet']

ave_rets = np.zeros((len(return_dir_list), int(1e6)))
fig, ax = plt.subplots()
x = np.arange(ave_rets.shape[-1])

for i, d in enumerate(return_dir_list):
    sed_rets = np.load(os.path.join(d, 'sedimentary_returns.npy'))
    milestones = np.load(os.path.join(d, 'episodic_steps.npy'))
    ave_rets[i,:milestones[0]] = sed_rets[0]
    for j in range(milestones.shape[-1]-1):
        ave_rets[i,milestones[j]:milestones[j+1]] = sed_rets[j+1]
    ave_rets[i,milestones[-1]:] = sed_rets[-1]
    ax.plot(x, ave_rets[i], colors[i], label=names[i])

ax.grid()
ax.legend()
plt.show()

