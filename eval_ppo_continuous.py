import sys
import os
import numpy as np
import random
import gym
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)
from train_ppo_continuous import PPOActorCritic


# Create LunarLander env
env = gym.make('LunarLanderContinuous-v2')

# load model
# model_path = './training_models/ppo/LunarLanderContinuous-v2/models/199'
model_path = './models/ppo/LunarLanderContinuous-v2/199'
ppo = PPOActorCritic(obs_dim=8, act_dim=2, beta=0.)
ppo.actor.mu_net = tf.keras.models.load_model(model_path)
# params
num_episodes = 10
ep_rets, ave_rets = [], []
num_steps = env.spec.max_episode_steps
# Test trained model
ep_rets, ave_rets = [], []
for ep in range(num_episodes):
    obs, done, rewards = env.reset(), False, []
    for st in range(num_steps):
        env.render()
        # act = ppo.act(obs.reshape(1,-1))
        act = ppo.actor._distribution(obs.reshape(1,-1)).mean().numpy()
        next_obs, rew, done, info = env.step(act)
        rewards.append(rew)
        # print("\n-\nepisode: {}, step: {} \naction: {} \nobs: {}, \nreward: {}".format(ep+1, st+1, act, obs, rew))
        obs = next_obs.copy()
        if done:
            ep_rets.append(sum(rewards))
            print("\n---\nepisode: {} \nepisode return: {}\n---\n".format(ep+1, ep_rets[-1]))
            break
