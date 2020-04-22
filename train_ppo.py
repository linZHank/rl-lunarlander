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


# Create LunarLander env
env = gym.make('LunarLander-v2')

# Create Policy Network
actor_net = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=env.observation_space.shape),
        tf.keras.layers.Dense(128, activation='tanh'),
        tf.keras.layers.Dense(128, activation='tanh'),
        tf.keras.layers.Dense(env.action_space.n)
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

def compute_loss_actor(actor_net, batch_obs, batch_logprobs, batch_acts, batch_advs, epsilon=0.2):
    tensor_obs = tf.convert_to_tensor(batch_obs)
    # print(tensor_obs)
    tensor_logprobs = tf.nn.log_softmax(actor_net(tensor_obs))
    # print(tensor_logprobs)
    tensor_new_samples = tf.squeeze(tf.random.categorical(tensor_logprobs, num_samples=1))
    # print(tensor_new_samples)
    tensor_logpi = tf.math.multiply(tensor_logprobs, tf.one_hot(tensor_new_samples, depth=env.action_space.n))
    # print(tensor_logpi)
    tensor_reduce_logpi = tf.math.reduce_sum(tensor_logpi, axis=1)
    # print(tensor_reduce_logpi)
    tensor_logprobs_old = tf.convert_to_tensor(batch_logprobs)
    # print(tensor_logprobs_old)
    tensor_acts = tf.convert_to_tensor(batch_acts)
    tensor_logpi_old = tf.math.multiply(tensor_logprobs_old, tf.one_hot(tensor_acts, depth=env.action_space.n))
    tensor_reduce_logpi_old = tf.math.reduce_sum(tensor_logpi_old, axis=1)
    # print(tensor_reduce_logpi_old)
    tensor_ratio = tf.math.exp(tensor_reduce_logpi - tensor_reduce_logpi_old)
    # print(tensor_ratio)
    tensor_advs = tf.convert_to_tensor(batch_advs)
    # print(tensor_advs)
    tensor_ratio_clip = tf.clip_by_value(tensor_ratio, clip_value_min=1-epsilon, clip_value_max=1+epsilon)
    tensor_objs_clip = tf.math.multiply(tensor_ratio_clip, tensor_advs)
    # print("clipped objective: {}".format(tensor_objs_clip))
    tensor_objs = tf.math.minimum(tf.math.multiply(tensor_ratio, tensor_advs), tensor_objs_clip)
    # print("min obj: {}".format(tensor_objs))
    loss_actor = -tf.math.reduce_mean(tensor_objs)
    # print(loss_actor)

    return loss_actor

def compute_loss_critic(critic_net, batch_obs, batch_rets):
    vals_pred = tf.squeeze(critic_net(tf.convert_to_tensor(batch_obs)))
    vals_target = tf.convert_to_tensor(batch_rets)
    loss_critic = tf.keras.losses.MSE(vals_target, vals_pred)
    # print(loss_critic)

    return loss_critic

def grad_actor(actor_net, batch_obs, batch_logprobs, batch_acts, batch_advs):
    with tf.GradientTape() as tape:
        loss_actor = compute_loss_actor(actor_net, batch_obs, batch_logprobs, batch_acts, batch_advs)

    return loss_actor, tape.gradient(loss_actor, actor_net.trainable_variables)

def grad_critic(critic_net, batch_obs, batch_rets):
    with tf.GradientTape() as tape:
        loss_critic = compute_loss_critic(critic_net, batch_obs, batch_rets)

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


# params
num_batches = 256
batch_size = 8192
lr_actor = 3e-4
lr_critic = 1e-3
gamma = 0.99
actor_train_iters = 64
critic_train_iters = 64
# Create Optimizers
optimizer_actor = tf.keras.optimizers.Adam(learning_rate=lr_actor)
optimizer_critic = tf.keras.optimizers.Adam(learning_rate=lr_critic)

#
obs, done = env.reset(), False
ep_rets, ave_rets = [], []
batch = 0
episode = 0
step = 0
for s in range(num_batches):
    batch_obs = []
    batch_vals = []
    batch_logprobs = []
    batch_acts = []
    batch_rets = [] # discounted reward to go
    batch_dones = []
    batch_next_obs = []
    batch_advs = []
    rewards = []
    while True:
        # env.render()
        val = critic_net(obs.reshape(1,-1))
        logprob = tf.nn.log_softmax(actor_net(obs.reshape(1,-1))) # shape=(1, action_space.n)
        # print(logprob) # debug
        action = np.squeeze(tf.random.categorical(logits=logprob, num_samples=1)) # squeeze (1,1) to (1,)
        # print(action) # debug
        next_obs, rew, done, info = env.step(action)
        batch_obs.append(obs.copy())
        batch_vals.append(np.squeeze(val.numpy()))
        batch_logprobs.append(np.squeeze(logprob.numpy()))
        batch_acts.append(action)
        rewards.append(np.float32(rew))
        batch_dones.append(done)
        batch_next_obs.append(next_obs.copy())
        # update obs
        obs = next_obs.copy() # Critical
        step += 1
        # print(obs, rew, done, info)
        logging.debug("\n-\nbatch: {}, episode: {}, step: {}, batch length: {} \naction: {} \nobs: {}".format(batch+1, episode+1, step+1, len(batch_obs), action, obs))
        if done:
            episode += 1
            step = 0
            ep_rets.append(sum(rewards))
            ave_rets.append(sum(ep_rets)/len(ep_rets))
            rtgs = reward_to_go(rewards, gamma)
            vals = batch_vals[-len(rewards):]
            nvals = vals[1:]+[0]
            advs = [np.float32(rtgs[i]+gamma*nvals[i]-vals[i]) for i in range(len(rewards))]
            batch_rets += list(rtgs)
            batch_advs += advs
            # print("rewards: {} \ndiscounted reward to go: {} \nvalues: {} \nnext values: {} \nadvantages: {}".format(rewards, rtgs, vals, nvals, advs))
            # logging.info("\n---\nbatch: {}, episode: {} \nreturn: {} \n".format(batch+1, episode, ep_rets[-1]))
            obs, done, rewards = env.reset(), False, []
            if len(batch_obs) > batch_size:
                batch += 1
                break
    # compute_loss_actor(actor_net, batch_obs, batch_logprobs, batch_acts, batch_advs)
    # update actor_net and critic_net
    for i in range(actor_train_iters):
        loss_actor, grads_actor = grad_actor(actor_net, batch_obs, batch_logprobs, batch_acts, batch_advs)
        optimizer_actor.apply_gradients(zip(grads_actor, actor_net.trainable_variables))
    for i in range(critic_train_iters):
        loss_critic, grads_critic = grad_critic(critic_net, batch_obs, batch_rets)
        optimizer_critic.apply_gradients(zip(grads_critic, critic_net.trainable_variables))
    # log batch
    logging.info("\n====\nbatch: {}, episode: {} \nloss_actor: {}, loss_critic: {} \nmean return: {} \n====\n".format(batch, episode, loss_actor, loss_critic, ave_rets))

# plot returns
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle('Averaged Returns')
ax.plot(ave_rets)
plt.show()
