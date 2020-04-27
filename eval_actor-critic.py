import sys
import os
import numpy as np
import random
import gym
import matplotlib.pyplot as plt
import tensorflow as tf

import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)


# Create LunarLander env
env = gym.make('LunarLander-v2')

# load model
model_path = './training_models/actor-critic/actor_net/2000.h5'
actor_net = tf.keras.models.load_model(model_path)
# params
num_episodes = 10
num_steps = 1000
ep_rets, ave_rets = [], []
if __name__ == '__main__':
    for ep in range(num_episodes):
        obs, done, rewards = env.reset(), False, []
        for st in range(num_steps):
            env.render()
            act = np.squeeze(tf.random.categorical(logits=tf.nn.log_softmax(actor_net(obs.reshape(1,-1))), num_samples=1))
            next_obs, rew, done, info = env.step(act)
            rewards.append(rew)
            logging.info("\n-\nepisode: {}, step: {} \naction: {} \nobs: {}, \nreward: {}".format(ep+1, st+1, act, obs, rew))
            obs = next_obs.copy()
            if done:
                ep_rets.append(sum(rewards))
                ave_rets.append(sum(ep_rets)/len(ep_rets))
                logging.info("\n---\nepisode: {} \nepisode return: {}, averaged return: {} \n---\n".format(ep+1, ep_rets[-1], ave_rets[-1]))
                break

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel("Episode")
ax.set_ylabel("Return")
ax.plot(ep_rets)
plt.show()
