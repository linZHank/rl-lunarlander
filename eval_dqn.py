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

# # Create Policy Network
# qnet_active = tf.keras.Sequential(
#     [
#         tf.keras.layers.InputLayer(input_shape=env.observation_space.shape),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(env.action_space.n)
#     ]
# )
# qnet_active.summary()

# # Restore checkpoint
# optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
# checkpoint_dir = './training_checkpoints/dqn'
# ckpt = tf.train.Checkpoint(optimizer=optimizer, model=qnet_active)
# ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir))

def epsilon_greedy(qvals, epsilon):
    if np.random.uniform() > epsilon:
        act_id = np.argmax(qvals)
    else:
        act_id = np.random.randint(env.action_space.n)

    return act_id

# load model
model_path = './training_models/dqn/2020-04-24-00-19/1850000.h5'
qnet_active = tf.keras.models.load_model(model_path)
# params
num_episodes = 10
num_steps = 1000
epsilon = .01
ep_rets, ave_rets = [], []
if __name__ == '__main__':
    for ep in range(num_episodes):
        obs, done, rewards = env.reset(), False, []
        for st in range(num_steps):
            env.render()
            qvals = np.squeeze(qnet_active(obs.reshape(1,-1)).numpy())
            action = epsilon_greedy(qvals, epsilon)
            next_obs, rew, done, info = env.step(action)
            rewards.append(rew)
            logging.info("\n-\nepisode: {}, step: {} \naction: {} \nobs: {}, \nreward: {}".format(ep+1, st+1, action, obs, rew))
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
