import sys
import os
from copy import deepcopy
import numpy as np
import scipy.signal
import random
import time
import gym
import matplotlib.pyplot as plt
from datetime import datetime
import logging

import tensorflow as tf
print(tf.__version__)
import tensorflow_probability as tfp
tfd = tfp.distributions
# # for the sake of debugging, import torch
# import torch
# import torch.nn as nn
# from torch.distributions.normal import Normal
################################################################
"""
Unnecessary initial settings
"""
# restrict GPU and memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
# set log level
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)
################################################################


################################################################
"""
instantiate env
"""
env = gym.make('LunarLanderContinuous-v2')
# env = gym.make('Pendulum-v0')
################################################################


################################################################
"""
Build actor_net, critic_net
"""
def mlp(sizes, activation, output_activation=None):
    inputs = tf.keras.Input(shape=(sizes[0],))
    x = tf.keras.layers.Dense(sizes[1], activation=activation)(inputs)
    for i in range(2,len(sizes)-1):
        x = tf.keras.layers.Dense(sizes[i], activation=activation)(x)
    outputs = tf.keras.layers.Dense(sizes[-1], activation=output_activation)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

# # debugging torch version mlp
# def torch_mlp(sizes, activation, output_activation=nn.Identity):
#     layers = []
#     for j in range(len(sizes)-1):
#         act = activation if j < len(sizes)-2 else output_activation
#         layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
#     return nn.Sequential(*layers)

class Actor(tf.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def __call__(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = tf.Variable(log_std)
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = tf.squeeze(self.mu_net(obs))
        std = tf.math.exp(self.log_std)
        return tfd.Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return tf.math.reduce_sum(pi.log_prob(act), axis=-1)

class MLPCritic(tf.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    @tf.function
    def __call__(self, obs):
        return tf.squeeze(self.v_net(obs), axis=-1)

class MLPActorCritic(tf.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64,64), activation='tanh'):
        super().__init__()
        self.actor = MLPGaussianActor(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=hidden_sizes, activation=activation)
        self.critic = MLPCritic(obs_dim=obs_dim, hidden_sizes=hidden_sizes, activation=activation)

    def step(self, obs):
        with tf.GradientTape() as t:
            with t.stop_recording():
                pi_dist = self.actor._distribution(obs)
                a = pi_dist.sample()
                logp_a = self.actor._log_prob_from_distribution(pi_dist, a)
                v = self.critic(obs)

        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
################################################################


################################################################
"""
Compute losses and gradients
"""
def compute_actor_gradients(data):
    obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
    with tf.GradientTape() as tape:
        tape.watch(ac.actor.trainable_variables)
        pi, logp = ac.actor(obs, act)
        # print("pi: {} \nlogp: {}".format(pi, logp))
        ratio = tf.math.exp(logp - logp_old)
        clip_adv = tf.math.multiply(tf.clip_by_value(ratio, 1-clip_ratio, 1+clip_ratio), adv)
        ent = tf.math.reduce_sum(pi.entropy(), axis=-1)
        loss = -tf.math.minimum(tf.math.multiply(ratio, adv), clip_adv) #+ .01*ent
        loss_pi = tf.math.reduce_mean(loss)
        # useful info
        approx_kl = tf.math.reduce_mean(logp_old - logp, axis=-1)
        entropy = tf.math.reduce_mean(ent)
        pi_info = dict(kl=approx_kl, ent=entropy)
    actor_grads = tape.gradient(loss_pi, ac.actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, ac.actor.trainable_variables))

    return loss_pi, pi_info

def compute_critic_gradients(data):
    obs, ret = data['obs'], data['ret']
    with tf.GradientTape() as tape:
        tape.watch(ac.critic.trainable_variables)
        loss_v = tf.keras.losses.MSE(ret, ac.critic(obs))
    critic_grads = tape.gradient(loss_v, ac.critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grads, ac.critic.trainable_variables))

    return loss_v

def update(buffer):
    data = buffer.get()
    for i in range(train_pi_iters):
        loss_pi, pi_info = compute_actor_gradients(data)
        kl = pi_info['kl']
        if kl > 1.5 * target_kl:
                print('Early stopping at step %d due to reaching max kl.'%i)
                break
    for j in range(train_v_iters):
        loss_v = compute_critic_gradients(data)

    return loss_pi, pi_info, loss_v
################################################################


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
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr <= self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
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
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: tf.convert_to_tensor(v, dtype=tf.float32) for k,v in data.items()}
################################################################


################################################################
"""
Main
"""
# paramas
steps_per_epoch=4000
epochs=200
gamma=0.99
clip_ratio=0.2
pi_lr=3e-4
vf_lr=1e-3
train_pi_iters=80
train_v_iters=80
lam=0.97
max_ep_len=1000
target_kl=0.01
save_freq=10
# instantiate actor-critic and replay buffer
obs_dim=env.observation_space.shape[0]
act_dim=env.action_space.shape[0]
ac = MLPActorCritic(obs_dim=obs_dim, act_dim=act_dim)
buffer = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)
# create optimizer
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
# Prepare for interaction with environment
model_dir = './training_models/ppo/'
start_time = time.time()
obs, ep_ret, ep_len = env.reset(), 0, 0
episodes, total_steps, ave_rets = 0, 0, []
# main loop
for ep in range(epochs):
    for st in range(steps_per_epoch):
        act, val, logp = ac.step(obs.reshape(1,-1))
        next_obs, rew, done, _ = env.step(act)
        ep_ret += rew
        ep_len += 1
        total_steps += 1
        buffer.store(obs, act, rew, val, logp)
        obs = next_obs # SUPER CRITICAL!!!
        # handle episode termination
        timeout = (ep_len==env.spec.max_episode_steps)
        terminal = done or timeout
        epoch_ended = (st==steps_per_epoch-1)
        if terminal or epoch_ended:
            if epoch_ended and not(terminal):
                print('Warning: trajectory cut off by epoch at {} steps.'.format(ep_len), flush=True)
            if timeout or epoch_ended:
                _, val, _ = ac.step(obs.reshape(1,-1))
            else:
                val = 0
            buffer.finish_path(val)
            if terminal:
                episodes += 1
                ave_rets.append(ep_ret/ep_len)
                print("\nTotalSteps: {} \nEpisode: {}, Step: {}, EpReturn: {}, EpLength: {}".format(total_steps, episodes, st+1, ep_ret, ep_len))
            obs, ep_ret, ep_len = env.reset(), 0, 0
    # Save model
    if not ep%save_freq or (ep==epochs-1):
        model_path = os.path.join(model_dir, env.spec.id, 'models', str(ep))
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        tf.saved_model.save(ac, model_path)

    # update actor-critic
    loss_pi, pi_info, loss_v = update(buffer)
    print("\n================================================================\nEpoch: {} \nStep: {} \nAveReturn: {} \nLossPi: {} \nLossV: {} \nKLDivergence: {} \n Entropy: {} \nTimeElapsed: {}\n================================================================\n".format(ep+1, st+1, ave_rets[-1], loss_pi, loss_v, pi_info['kl'], pi_info['ent'], time.time()-start_time))
################################################################


# Test trained model
input("Press ENTER to test lander...")
num_episodes = 10
num_steps = env.spec.max_episode_steps
ep_rets, ave_rets = [], []
for ep in range(num_episodes):
    obs, done, rewards = env.reset(), False, []
    for st in range(num_steps):
        env.render()
        act, _, _ = ac.step(obs.reshape(1,-1))
        next_obs, rew, done, info = env.step(act)
        rewards.append(rew)
        # print("\n-\nepisode: {}, step: {} \naction: {} \nobs: {}, \nreward: {}".format(ep+1, st+1, act, obs, rew))
        obs = next_obs.copy()
        if done:
            ep_rets.append(sum(rewards))
            ave_rets.append(sum(ep_rets)/len(ep_rets))
            print("\n---\nepisode: {} \nepisode return: {}, averaged return: {} \n---\n".format(ep+1, ep_rets[-1], ave_rets[-1]))
            break
