import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


# core.py
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()


# sac.py
from copy import deepcopy
import itertools
import numpy as np
from torch.optim import Adam
import gym
import time

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


# setup agent
env = gym.make('LunarLanderContinuous-v2')
obs_dim = env.observation_space.shape
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0]
ac = MLPActorCritic(env.observation_space, env.action_space)
ac_targ = deepcopy(ac)
# List of parameters for both Q-networks (save this for convenience)
q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
# Count variables (protip: try to get a feel for how different size networks behave!)
var_counts = tuple(count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
# Set up function for computing SAC Q-losses
def compute_loss_q(data):
    o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

    q1 = ac.q1(o,a)
    q2 = ac.q2(o,a)

    # Bellman backup for Q functions
    with torch.no_grad():
        # Target actions come from *current* policy
        a2, logp_a2 = ac.pi(o2)

        # Target Q-values
        q1_pi_targ = ac_targ.q1(o2, a2)
        q2_pi_targ = ac_targ.q2(o2, a2)
        q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
        backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)
    # MSE loss against Bellman backup
    loss_q1 = ((q1 - backup)**2).mean()
    loss_q2 = ((q2 - backup)**2).mean()
    loss_q = loss_q1 + loss_q2
    # Useful info for logging
    q_info = dict(Q1Vals=q1.detach().numpy(),
                  Q2Vals=q2.detach().numpy())
    print("loss_q: {}".format(loss_q))

    return loss_q, q_info

# Set up function for computing SAC pi loss
def compute_loss_pi(data):
    o = data['obs']
    pi, logp_pi = ac.pi(o)
    q1_pi = ac.q1(o, pi)
    q2_pi = ac.q2(o, pi)
    q_pi = torch.min(q1_pi, q2_pi)
    # Entropy-regularized policy loss
    loss_pi = (alpha * logp_pi - q_pi).mean()
    # Useful info for logging
    pi_info = dict(LogPi=logp_pi.detach().numpy())
    print("loss_pi: {}".format(loss_pi))

    return loss_pi, pi_info

def update(data):
    # First run one gradient descent step for Q1 and Q2
    q_optimizer.zero_grad()
    loss_q, q_info = compute_loss_q(data)
    loss_q.backward()
    q_optimizer.step()
    # Freeze Q-networks so you don't waste computational effort
    # computing gradients for them during the policy learning step.
    for p in q_params:
        p.requires_grad = False
    # Next run one gradient descent step for pi.
    pi_optimizer.zero_grad()
    loss_pi, pi_info = compute_loss_pi(data)
    loss_pi.backward()
    pi_optimizer.step()
    # Unfreeze Q-networks so you can optimize it at next DDPG step.
    for p in q_params:
        p.requires_grad = True
    # Finally, update target networks by polyak averaging.
    with torch.no_grad():
        for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
            # NB: We use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors.
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)

def get_action(o, deterministic=False):
    return ac.act(torch.as_tensor(o, dtype=torch.float32),
                  deterministic)


# get prepared
steps_per_epoch=4000
epochs=100
replay_size=int(1e6)
gamma=0.99
polyak=0.995
lr=1e-3
alpha=0.2
batch_size=100
start_steps=10000
update_after=1000
update_every=50
num_test_episodes=10
max_ep_len=1000
save_freq = 1
total_steps = steps_per_epoch * epochs
start_time = time.time()
# Set up optimizers for policy and q-function
pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
q_optimizer = Adam(q_params, lr=lr)
# Freeze target networks with respect to optimizers (only update via polyak averaging)
for p in ac_targ.parameters():
    p.requires_grad = False
# Experience buffer
replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

# main loop
o, ep_ret, ep_len = env.reset(), 0, 0
for t in range(total_steps):
    # Until start_steps have elapsed, randomly sample actions
    # from a uniform distribution for better exploration. Afterwards,
    # use the learned policy.
    if t > start_steps:
        a = get_action(o)
    else:
        a = env.action_space.sample()
    # Step the env
    o2, r, d, _ = env.step(a)
    ep_ret += r
    ep_len += 1
    # Ignore the "done" signal if it comes from hitting the time
    # horizon (that is, when it's an artificial terminal signal
    # that isn't based on the agent's state)
    d = False if ep_len==max_ep_len else d
    # Store experience to replay buffer
    replay_buffer.store(o, a, r, o2, d)
    # Super critical, easy to overlook step: make sure to update
    # most recent observation!
    o = o2
    # End of trajectory handling
    if d or (ep_len == max_ep_len):
        # logger.store(EpRet=ep_ret, EpLen=ep_len)
        # o, ep_ret, ep_len = env.reset(), 0, 0
        o = env.reset()

    # Update handling
    if t >= update_after and t % update_every == 0:
        for j in range(update_every):
            batch = replay_buffer.sample_batch(batch_size)
            update(data=batch)
    # End of epoch handling
    if (t+1) % steps_per_epoch == 0:
        epoch = (t+1) // steps_per_epoch
        print(
            "\n================================================================\nEpoch: {}, Step: {} \nEpReturns: {} \nAveEpReturn: {} \nTime: {} \n================================================================\n".format(
                epoch,
                t+1,
                ep_ret,
                ep_ret/ep_len,
                time.time()-start_time
            )
        )
        # Save model
        # if (epoch % save_freq == 0) or (epoch == epochs):
        #     logger.save_state({'env': env}, None)


# Test trained model
input("Press ENTER to test lander...")
num_episodes = 10
num_steps = env.spec.max_episode_steps
ep_rets, ave_rets = [], []
for ep in range(num_episodes):
    obs, done, rewards = env.reset(), False, []
    for st in range(num_steps):
        env.render()
        act = get_action(obs)
        next_obs, rew, done, info = env.step(act)
        rewards.append(rew)
        print("\n-\nepisode: {}, step: {} \naction: {} \nobs: {}, \nreward: {}".format(ep+1, st+1, act, obs, rew))
        obs = next_obs.copy()
        if done:
            ep_rets.append(sum(rewards))
            ave_rets.append(sum(ep_rets)/len(ep_rets))
            print("\n---\nepisode: {} \nepisode return: {}, averaged return: {} \n---\n".format(ep+1, ep_rets[-1], ave_rets[-1]))
            break