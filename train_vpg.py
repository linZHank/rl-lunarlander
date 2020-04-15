import sys
import os
import numpy as np
import gym
import matplotlib.pyplot as plt

import tensorflow as tf
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
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)


# create LunarLander env
env = gym.make('LunarLander-v2')

# Create Policy Network
policy_net = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=env.observation_space.shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(env.action_space.n, activation='softmax')
    ]
)
policy_net.summary()

# Create an Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

def grad(policy_net, batch_obs, batch_acts, batch_rets):
    with tf.GradientTape() as tape:
        batch_prob_acts = policy_net(tf.convert_to_tensor(batch_obs))
        batch_acts_onehot = tf.one_hot(tf.convert_to_tensor(batch_acts), depth=env.action_space.n)
        batch_pis = tf.math.multiply(batch_prob_acts, batch_acts_onehot)
        batch_reduce_pis = tf.math.reduce_sum(batch_pis, axis=1)
        batch_logpis = tf.math.log(batch_reduce_pis)
        batch_weighted_logpis = tf.math.multiply(batch_logpis, tf.convert_to_tensor(batch_rets))
        pg_loss = -tf.math.reduce_mean(batch_weighted_logpis)

    return pg_loss, tape.gradient(pg_loss, policy_net.trainable_variables)

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


################################### Test grad ###################################
# obs = env.reset()
# batch_obs = []
# batch_acts = []
# batch_rets = []
# ep_rewards = []
# for _ in range(16):
#     prob_acts = policy_net(obs.reshape(1,-1))
#     action = np.squeeze(tf.random.categorical(logits=prob_acts, num_samples=1))
#     obs, rew, done, _ = env.step(action)
#     batch_obs.append(obs.copy())
#     batch_acts.append(action)
#     ep_rewards.append(np.float32(rew))
#     if done:
#         break
# ep_ret, ep_len = np.sum(ep_rewards), len(ep_rewards)
# batch_rets += [ep_ret]*ep_len
# #
# batch_prob_acts = policy_net(tf.convert_to_tensor(batch_obs))
# # print(batch_prob_acts) # debug
# batch_acts_onehot = tf.one_hot(tf.convert_to_tensor(batch_acts), depth=env.action_space.n)
# # print(batch_acts_onehot) # debug
# batch_pis = tf.math.multiply(batch_prob_acts, batch_acts_onehot)
# # print(batch_pis) # debug
# batch_reduce_pis = tf.math.reduce_sum(batch_pis, axis=1)
# # print(batch_reduce_pis) # debug
# batch_logpis = tf.math.log(batch_reduce_pis)
# # print(batch_logpis) # debug
# # print(tf.convert_to_tensor(batch_rets)) # debug
# loss = -tf.math.multiply(batch_logpis, tf.convert_to_tensor(batch_rets))
# print(loss) # debug
##################################################################################


# params
num_batches = 4096
batch_size = 8192
if __name__ == '__main__':
    obs, done = env.reset(), False
    ep_rets, ave_rets = [], []
    batch = 0
    episode = 0
    step = 0
    for s in range(num_batches):
        batch_obs = []
        batch_acts = []
        batch_rtgs = []
        rewards = []
        while True:
            # env.render()
            prob_a = policy_net(obs.reshape(1,-1))
            # print(prob_a) # debug
            action = np.squeeze(tf.random.categorical(logits=prob_a, num_samples=1)) # squeeze (1,1) to (1,)
            # print(action) # debug
            obs, rew, done, info = env.step(action)
            batch_obs.append(obs.copy())
            batch_acts.append(action)
            rewards.append(np.float32(rew))
            step += 1
            # print(obs, rew, done, info)
            logging.debug("\n-\nbatch: {}, episode: {}, step: {}, batch length: {} \naction: {} \nobs: {}".format(batch+1, episode+1, step+1, len(batch_obs), action, obs))
            if done:
                episode += 1
                step = 0
                ep_rets.append(sum(rewards))
                ave_rets.append(sum(ep_rets)/len(ep_rets))
                batch_rtgs += list(reward_to_go(rewards))
                logging.info("\n---\nbatch: {}, episode: {} \nreturn: {} \n".format(batch+1, episode+1, ep_rets[-1]))
                obs, done, rewards = env.reset(), False, []
                if len(batch_obs) > batch_size:
                    batch += 1
                    break
        batch_rtgs = np.array(batch_rtgs)-np.sum(ep_rets)/(episode+1) # subtract baseline
        pg_loss, grads = grad(policy_net, batch_obs, batch_acts, batch_rtgs)
        optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))
        logging.info("\n====\nbatch: {}, episode: {} \nloss: {} \nmean return: {} \n====\n".format(batch+1, episode+1, pg_loss, np.mean(batch_rtgs)))

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle('Averaged Returns')
ax.plot(ave_rets)
plt.show()
