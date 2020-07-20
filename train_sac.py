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
    def __init__(self, obs_dim, act_dim, act_lim=1, hidden_sizes=(256,256), activation='relu', **kwargs):
        super(SoftActorCritic, self).__init__(name='sac', **kwargs)
        self.pi = Actor(obs_dim, act_dim, hidden_sizes, activation, act_lim)
        self.q0 = Critic(obs_dim, act_dim, hidden_sizes, activation) 
        self.q1 = Critic(obs_dim, act_dim, hidden_sizes, activation) 

    def act(self, obs, deterministic=False):
        a, _ = self.pi(obs, deterministic, False)
        return a.numpy()




if __name__=='__main__':
    env = gym.make('LunarLanderContinuous-v2')
    sac = SoftActorCritic(obs_dim=8,act_dim=2)
    o = env.reset()
    for t in range(200):
        env.render()
        a = np.squeeze(sac.act(o.reshape(1,-1)))
        o2, r, d, _ = env.step(a)
        o = o2
        if d:
            break

