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

def mlp(sizes, activation, output_activation=None):
    inputs = tf.keras.Input(shape=(sizes[0],))
    x = tf.keras.layers.Dense(sizes[1], activation=activation)(inputs)
    for i in range(2,len(sizes)-1):
        x = tf.keras.layers.Dense(sizes[i], activation=activation)(x)
    outputs = tf.keras.layers.Dense(sizes[-1], activation=output_activation)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

class Critic(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, **kwargs):
        super(Critic, self).__init__(name='critic', **kwargs)
        inputs = tf.keras.Input(shape=(obs_dim+act_dim,))
        x = tf.keras.layers.Dense(hidden_sizes[0], activation=activation)(inputs)
        for i in range(1, len(hidden_sizes)):
            x = tf.keras.layers.Dense(hidden_sizes[i], activation=activation)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        self.q_net = tf.keras.Model(inputs=inputs, outputs=outputs)
        
    def call(self, obs, act):
        qval = self.q_net(tf.concat([obs, act], axis=-1))
        return tf.squeeze(qval, axis=-1)

class Actor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_lim, **kwargs):
        super(Actor, self).__init__(name='actor', **kwargs)
        inputs = tf.keras.Input(shape=(obs_dim,))
        x = tf.keras.layers.Dense(hidden_sizes[0], activation=activation)(inputs)
        for i in range(1, len(hidden_sizes)):
            x = tf.keras.layers.Dense(hidden_sizes[i], activation=activation)(x)
        mu_layer = tf.keras.layers.Dense(act_dim)(x) 
        log_std_layer = tf.keras.layers.Dense(act_dim)(x) 
        self.policy_net = tf.keras.Model(inputs=inputs, outputs=[mu_layer, log_std_layer])
        self.act_lim = act_lim

    def call(self, obs, deterministic=False, with_logprob=True):
        mu, log_std = self.policy_net(obs)
        log_std = tf.clip_by_value(log_std, -20, 2)
        std = tf.math.exp(log_std)
        pi_distribution = tfd.Normal(mu, std)
        if deterministic:
            action = mu # only use for evaluation
        else:
            eps = tf.random.normal(shape=mu.shape)
            action = mu + eps*std # reparameterization trick
        if with_logprob:
            # arXiv 1801.01290, appendix C
            logp_pi = tf.math.reduce_sum(pi_distribution.log_prob(action), axis=-1)
            logp_pi -= tf.math.reduce_sum(2*(np.log(2) - action - tf.math.softplus(-2*action)), axis=-1)
        else:
            logp_pi = None
        action = tf.math.tanh(action)
        action = self.act_lim*action

        return action, logp_pi
        
class SoftActorCritic(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, act_lim=1, hidden_sizes=(256,256), activation='relu', gamma = 0.99, alpha=0.2,
                 critic_lr=3e-4, actor_lr=1e-4, **kwargs):
        super(SoftActorCritic, self).__init__(name='sac', **kwargs)
        # params
        name = 'sac_agent'
        self.alpha = alpha # entropy temperature
        self.gamma = gamma # discount rate
        #
        self.pi = Actor(obs_dim, act_dim, hidden_sizes, activation, act_lim)
        self.q0 = Critic(obs_dim, act_dim, hidden_sizes, activation) 
        self.q1 = Critic(obs_dim, act_dim, hidden_sizes, activation) 
        self.targ_q0 = Critic(obs_dim, act_dim, hidden_sizes, activation)
        self.targ_q1 = Critic(obs_dim, act_dim, hidden_sizes, activation)
        self.critic_optimizer = tf.keras.optimizers.Adam(lr=critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(lr=actor_lr)

    def act(self, obs, deterministic=False):
        a, _ = self.pi(obs, deterministic, False)
        return a.numpy()

    def train_one_batch(self, data):
        # update critic
        with tf.GradientTape() as tape:
            tape.watch(self.q0.trainable_weights+self.q1.trainable_weights)
            pred_qval0 = self.q0(data['obs'], data['act'])
            pred_qval1 = self.q1(data['obs'], data['act'])
            nact, nlogp = self.pi(data['nobs'])
            nqval0 = self.targ_q0(data['nobs'], nact) # compute qval for next step
            nqval1 = self.targ_q1(data['nobs'], nact)
            pessi_nqval = tf.math.minimum(nqval0, nqval1) # pessimistic value
            targ_qval = data['rew'] + self.gamma*(1 - data['done'])*(pessi_nqval - self.alpha*nlogp)
            loss_q0 = tf.keras.losses.MSE(y_true=targ_qval, y_pred=pred_qval0)
            loss_q1 = tf.keras.losses.MSE(y_true=targ_qval, y_pred=pred_qval1)
            loss_q = loss_q0 + loss_q1
        grads_critic = tape.gradient(loss_q, self.q0.trainable_weights+self.q1.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(grads_critic, self.q0.trainable_weights+self.q1.trainable_weights))
        # update actor
        with tf.GradientTape() as tape:
            tape.watch(self.pi.trainable_weights)
            act, logp = self.pi(data['obs'])
            qval0 = self.q0(data['obs'], act)
            qval1 = self.q1(data['obs'], act)
            pessi_qval = tf.math.minimum(qval0, qval1)
            loss_pi = tf.math.reduce_mean(self.alpha*logp - pessi_qval)
        grads_actor = tape.gradient(loss_pi, self.pi.trainable_weights)
        self.actor_optimizer.apply_gradients(zip(grads_actor, self.pi.trainable_weights))

        return loss_q, loss_pi
    
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.nobs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, done, nobs):
        self.obs_buf[self.ptr] = obs
        self.nobs_buf[self.ptr] = nobs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=tf.convert_to_tensor(self.obs_buf[idxs]),
                     nobs=tf.convert_to_tensor(self.nobs_buf[idxs]),
                     act=tf.convert_to_tensor(self.act_buf[idxs]),
                     rew=tf.convert_to_tensor(self.rew_buf[idxs]),
                     done=tf.convert_to_tensor(self.done_buf[idxs]))
        return batch


if __name__=='__main__':
    env = gym.make('LunarLanderContinuous-v2')
    max_episode_steps = env.spec.max_episode_steps
    update_freq = 50
    sac = SoftActorCritic(obs_dim=8, act_dim=2)
    replay_buffer = ReplayBuffer(obs_dim=8, act_dim=2, size=int(1e6)) 
    obs, ep_len = env.reset(), 0
    for t in range(1000):
        env.render()
        act = np.squeeze(sac.act(obs.reshape(1,-1)))
        nobs, rew, done, _ = env.step(act)
        ep_len += 1
        done = False if ep_len == max_episode_steps else done
        replay_buffer.store(obs, act, rew, done, nobs)
        obs = nobs
        if done or (ep_len==max_episode_steps):
            obs, ep_len = env.reset(), 0
        if not t%update_freq:
            for _ in range(update_freq):
                minibatch = replay_buffer.sample_batch()
                loss_q, loss_pi = sac.train_one_batch(data=minibatch)
                print("\nloss_q: {} \nloss_pi: {}".format(loss_q, loss_pi))


