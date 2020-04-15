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
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)


# Create LunarLander env
env = gym.make('LunarLander-v2')

# Create Policy Network
actor_net = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=env.observation_space.shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(env.action_space.n, activation='softmax')
    ]
)
actor_net.summary()

# Create value Network
critic_net = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=env.observation_space.shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ]
)
critic_net.summary()

def grad_actor(actor_net, batch_obs, batch_acts, batch_rtgs, batch_dones, batch_next_obs):
    with tf.GradientTape() as tape:
        batch_probs = actor_net(tf.convert_to_tensor(batch_obs))
        batch_acts_onehot = tf.one_hot(tf.convert_to_tensor(batch_acts), depth=env.action_space.n)
        batch_pis = tf.math.multiply(batch_probs, batch_acts_onehot)
        batch_reduce_pis = tf.math.reduce_sum(batch_pis, axis=1)
        batch_logpis = tf.math.log(batch_reduce_pis)
        batch_vals = tf.squeeze(critic_net(tf.convert_to_tensor(batch_obs)))
        batch_next_vals = tf.squeeze(critic_net(tf.convert_to_tensor(batch_next_obs)))
        batch_advs = tf.convert_to_tensor(batch_rtgs) + (1.-np.array(batch_dones))*gamma*batch_next_vals - batch_vals
        loss_actor = -tf.math.reduce_mean(tf.math.multiply(batch_logpis, batch_advs))

    return loss_actor, tape.gradient(loss_actor, actor_net.trainable_variables)

def grad_critic(critic_net, batch_obs, batch_rtgs):
    with tf.GradientTape() as tape:
        vals_pred = tf.squeeze(critic_net(tf.convert_to_tensor(batch_obs)))
        vals_target = tf.convert_to_tensor(batch_rtgs)
        loss_critic = tf.keras.losses.MSE(vals_target, vals_pred)

    return loss_critic, tape.gradient(loss_critic, critic_net.trainable_variables)

def reward_to_go(rews, gamma):
    """
    discount considered
    """
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (gamma*rtgs[i+1] if i+1 < n else 0)
    return rtgs


################################### Test Loss ###################################
# gamma = 0.99
# obs = env.reset()
# batch_obs = []
# batch_acts = []
# batch_dones = []
# # batch_rets = []
# batch_rtgs = []
# batch_next_obs = []
# rewards = []
# for ep in range(4):
#     for st in range(100):
#         prob_acts = actor_net(obs.reshape(1,-1))
#         action = np.squeeze(tf.random.categorical(logits=prob_acts, num_samples=1))
#         next_obs, rew, done, _ = env.step(action)
#         batch_obs.append(obs.copy())
#         batch_acts.append(action)
#         batch_dones.append(done)
#         batch_next_obs.append(next_obs)
#         rewards.append(np.float32(gamma**st*rew))
#         obs = next_obs.copy()
#     batch_rtgs += list(reward_to_go(rewards))
#     rewards = []
#     obs = env.reset()
#
# # ep_ret, ep_len = np.sum(ep_rewards), len(ep_rewards)
# # batch_rets += [ep_ret]*ep_len
# # compute loss
# batch_prob_acts = actor_net(tf.convert_to_tensor(batch_obs))
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
# batch_vals = tf.squeeze(critic_net(tf.convert_to_tensor(batch_obs)))
# # print(batch_vals)
# batch_next_vals = tf.squeeze(critic_net(tf.convert_to_tensor(batch_next_obs)))
# # print(batch_next_vals)
# # print(tf.convert_to_tensor(batch_rtgs))
# batch_advs = tf.convert_to_tensor(batch_rtgs) + (1.-np.array(batch_dones))*gamma*batch_next_vals - batch_vals
# # print(batch_advs)
# pg_loss = -tf.math.reduce_mean(tf.math.multiply(batch_logpis, batch_advs))
# # print(pg_loss)
# vals_pred = tf.squeeze(critic_net(tf.convert_to_tensor(batch_obs)))
# # print(vals_pred)
# vals_target = tf.convert_to_tensor(batch_rtgs)
# # print(vals_target)
# loss_critic = tf.keras.losses.MSE(vals_target, vals_pred)
# # print(loss_critic)

##################################################################################


# params
num_batches = 256
batch_size = 8192
lr_actor = 3e-4
lr_critic = 1e-3
gamma = 0.99
val_train_iters = 32
# Create Optimizers
optimizer_actor = tf.keras.optimizers.Adam(learning_rate=lr_actor)
optimizer_critic = tf.keras.optimizers.Adam(learning_rate=lr_critic)
if __name__ == '__main__':
    obs, done = env.reset(), False
    ep_rets, ave_rets = [], []
    batch = 0
    episode = 0
    step = 0
    for s in range(num_batches):
        batch_obs = []
        batch_acts = []
        batch_drtgs = [] # discounted reward to go
        batch_dones = []
        batch_next_obs = []
        rewards = []
        while True:
            # env.render()
            prob_a = actor_net(obs.reshape(1,-1))
            # print(prob_a) # debug
            action = np.squeeze(tf.random.categorical(logits=prob_a, num_samples=1)) # squeeze (1,1) to (1,)
            # print(action) # debug
            next_obs, rew, done, info = env.step(action)
            batch_obs.append(obs.copy())
            batch_acts.append(action)
            rewards.append(np.float32(rew))
            batch_dones.append(done)
            batch_next_obs.append(next_obs.copy())
            step += 1
            # print(obs, rew, done, info)
            logging.debug("\n-\nbatch: {}, episode: {}, step: {}, batch length: {} \naction: {} \nobs: {}".format(batch+1, episode+1, step+1, len(batch_obs), action, obs))
            obs = next_obs.copy()
            if done:
                episode += 1
                step = 0
                ep_rets.append(sum(rewards))
                ave_rets.append(sum(ep_rets)/len(ep_rets))
                batch_drtgs += list(reward_to_go(rewards, gamma))
                logging.info("\n---\nbatch: {}, episode: {} \nreturn: {} \n".format(batch+1, episode+1, ep_rets[-1]))
                obs, done, rewards = env.reset(), False, []
                if len(batch_obs) > batch_size:
                    batch += 1
                    break
        loss_actor, grads_actor = grad_actor(actor_net, batch_obs, batch_acts, batch_drtgs, batch_dones, batch_next_obs)
        optimizer_actor.apply_gradients(zip(grads_actor, actor_net.trainable_variables))
        for i in range(val_train_iters):
            loss_critic, grads_critic = grad_critic(critic_net, batch_obs, batch_drtgs)
            optimizer_critic.apply_gradients(zip(grads_critic, critic_net.trainable_variables))

        logging.info("\n====\nbatch: {}, episode: {} \nloss_actor: {}, loss_critic: {} \nmean return: {} \n====\n".format(batch, episode, loss_actor, loss_critic, np.mean(batch_drtgs)))

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle('Averaged Returns')
ax.plot(ave_rets)
plt.show()
