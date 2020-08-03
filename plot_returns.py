import sys
import os
import numpy as np
import matplotlib.pyplot as plt

return_dir_list = [
    sys.path[0] + '/models/dqn/LunarLander-v2',
    sys.path[0] + '/models/ddpg/LunarLanderContinuous-v2',
    sys.path[0] + '/models/ppo/LunarLanderContinuous-v2',
    sys.path[0] + '/models/ppo_plus_ent/LunarLanderContinuous-v2',
    sys.path[0] + '/models/sac/LunarLanderContinuous-v2',
    sys.path[0] + '/models/sac_fix_ent/LunarLanderContinuous-v2',
    sys.path[0] + '/models/sac_auto_ent/LunarLanderContinuous-v2'
]
names = ['dqn', 'ddpg', 'ppo', 'ppo_entropy', 'sac', 'sac_fixed_entropy', 'sac_auto_entropy']
colors = ['limegreen', 'grey', 'deeppink', 'violet', 'royalblue', 'dodgerblue', 'deepskyblue']

ave_rets = np.zeros((len(return_dir_list), int(1e6)))
fig, ax = plt.subplots(figsize=(12,6))
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
plt.tight_layout()
plt.show()

