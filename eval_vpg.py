import sys
import os
import numpy as np
import random
import gym
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)


# Create LunarLander env
# env = gym.make('LunarLander-v2')
env = gym.make('CartPole-v0')

# load model
# # Create Policy Network
# actor_net = tf.keras.Sequential(
#     [
#         tf.keras.layers.InputLayer(input_shape=env.observation_space.shape),
#         tf.keras.layers.Dense(32, activation='tanh'),
#         tf.keras.layers.Dense(32, activation='tanh'),
#         tf.keras.layers.Dense(env.action_space.n, activation='softmax')
#     ]
# )
saved_path = './training_models/vpg/CartPole-v0/model'
actor_net = tf.saved_model.load(saved_path)
# actor_net.summary()
# params
num_episodes = 10
num_steps = env.spec.max_episode_steps
ep_rets, ave_rets = [], []
if __name__ == '__main__':
    for ep in range(num_episodes):
        obs, done, rewards = env.reset(), False, []
        for st in range(num_steps):
            env.render()
            logprob = tf.stop_gradient(tf.nn.log_softmax(actor_net(obs.reshape(1,-1))))
            act = np.squeeze(tf.random.categorical(logits=logprob, num_samples=1)) # squeeze (1,1) to (1,)
            next_obs, rew, done, info = env.step(act)
            rewards.append(rew)
            logging.debug("\n-\nepisode: {}, step: {} \naction: {} \nobs: {}, \nreward: {}".format(ep+1, st+1, act, obs, rew))
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
