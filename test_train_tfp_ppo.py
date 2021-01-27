import sys
import os
import numpy as np
import time
import gym
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.signal
import tensorflow as tf
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

from agents.tfp_ppo import PPOAgent

################################################################
"""
On-policy Replay Buffer for PPO
"""
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.lpa_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, lpa):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr <= self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.lpa_buf[self.ptr] = lpa
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr
        # self.ptr, self.path_start_idx = 0, 0

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, lpa=self.lpa_buf)
        return {k: tf.convert_to_tensor(v, dtype=tf.float32) for k,v in data.items()}
################################################################
RANDOM_SEED = 0
# instantiate env
env = gym.make('LunarLander-v2')
dim_obs = env.observation_space.shape[0]
num_act = env.action_space.n
dim_act = 1
# instantiate actor-critic and replay buffer
agent = PPOAgent(target_kld=.02, beta=0.)
replay_buffer = PPOBuffer(dim_obs, dim_act, size=5000)
save_dir = './saved_models/'+env.spec.id+'/ppo/'+datetime.now().strftime("%Y-%m-%d-%H-%M")+'/'
if not os.path.exists(save_dir):
    try:
        os.makedirs(save_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
policy_net_path = os.path.join(save_dir, 'policy_net')
value_net_path = os.path.join(save_dir, 'value_net')
# set seed
tf.random.set_seed(RANDOM_SEED)
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
env.action_space.seed(RANDOM_SEED)
# paramas
num_trains = 50
train_epochs = 80
save_freq = 20
# prepare for interaction with environment
obs, ep_ret, ep_len = env.reset(), 0, 0
ep_cntr, st_cntr = 0, 0
stepwise_rewards, episodic_returns, sedimentary_returns = [], [], []
episodic_steps = []
start_time = time.time()
# main loop
for t in range(num_trains):
    for s in range(replay_buffer.max_size):
        act, val, logp = agent.make_decision(np.expand_dims(obs,0))
        n_obs, rew, done, _ = env.step(act)
        ep_ret += rew
        ep_len += 1
        replay_buffer.store(obs, act, rew, val, logp)
        obs = n_obs
        # handle episode termination
        timeout = (ep_len==env.spec.max_episode_steps)
        terminal = done or timeout
        epoch_ended = (s==replay_buffer.max_size-1)
        if terminal or epoch_ended:
            if epoch_ended and not(terminal):
                print('Warning: trajectory cut off by epoch at {} steps.'.format(ep_len), flush=True)
            if timeout or epoch_ended:
                _, val, _ = agent.make_decision(np.expand_dims(obs,0))
            else:
                val = 0
            replay_buffer.finish_path(val)
            if terminal:
                ep_cntr += 1
                episodic_returns.append(ep_ret)
                sedimentary_returns.append(sum(episodic_returns)/ep_cntr)
                episodic_steps.append(st_cntr)
                print("\n====\nTotalSteps: {} \nEpisode: {}, Step: {}, EpReturn: {}, EpLength: {} \n====\n".format(st_cntr, ep_cntr, s+1, ep_ret, ep_len))
            obs, ep_ret, ep_len = env.reset(), 0, 0
    data = replay_buffer.get()
    loss_pi, loss_v, loss_info = agent.train(data, train_epochs)
    print("\n====\nTraining: {} \nTotalSteps: {} \nDataSize: {} \nAveReturn: {} \nLossPi: {} \nLossV: {} \nKLDivergence: {} \nEntropy: {} \nTimeElapsed: {}\n====\n".format(t+1, st_cntr, data['ret'].shape[0], sedimentary_returns[-1], loss_pi, loss_v, loss_info['kld'], loss_info['entropy'], time.time()-start_time))


# plot returns
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle('Averaged Returns')
ax.plot(sedimentary_returns)
plt.show()


