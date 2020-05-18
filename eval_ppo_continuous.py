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


# Create LunarLander env
env = gym.make('LunarLanderContinuous-v2')

# load model
# model_path = './training_models/ppo/LunarLanderContinuous-v2/models/199'
model_path = './training_models/ppo/test/LunarLanderContinuous-v2/models/9'
ac = tf.saved_model.load(model_path)
# params
num_episodes = 10
num_steps = 1000
ep_rets, ave_rets = [], []
if __name__ == '__main__':
    num_episodes = 10
    num_steps = env.spec.max_episode_steps
    ep_rets, ave_rets = [], []
    for ep in range(num_episodes):
        obs, done, rewards = env.reset(), False, []
        for st in range(num_steps):
            env.render()
            act, _, _ = ac.step(obs.reshape(1,-1))
            next_obs, rew, done, info = env.step(act.numpy())
            rewards.append(rew)
            # print("\n-\nepisode: {}, step: {} \naction: {} \nobs: {}, \nreward: {}".format(ep+1, st+1, act, obs, rew))
            obs = next_obs.copy()
            if done:
                ep_rets.append(sum(rewards))
                ave_rets.append(sum(ep_rets)/len(ep_rets))
                print("\n---\nepisode: {} \nepisode return: {}, averaged return: {} \n---\n".format(ep+1, ep_rets[-1], ave_rets[-1]))
                break
