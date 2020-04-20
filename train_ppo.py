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

def compute_loss_actor(batch_obs, batch_logprobs, batch_acts):
    tensor_obs = tf.convert_to_tensor(batch_obs)
    # print(tensor_obs)
    tensor_logprobs = tf.convert_to_tensor(batch_logprobs)
    # print(tensor_logprobs)
    tensor_new_samples = tf.squeeze(tf.random.categorical(tensor_logprobs, num_samples=1))
    # print(tensor_new_samples)
    tensor_logpi = tf.math.multiply(tensor_logprobs, tf.one_hot(tensor_new_samples, depth=env.action_space.n))
    # print(tensor_logpi)
    tensor_reduce_logpi = tf.math.reduce_sum(tensor_logpi, axis=1)
    # print(tensor_reduce_logpi)
    tensor_acts = tf.convert_to_tensor(batch_acts)
    tensor_logpi_old = tf.math.multiply(tensor_logprobs, tf.one_hot(tensor_acts, depth=env.action_space.n))
    tensor_reduce_logpi_old = tf.math.reduce_sum(tensor_logpi_old, axis=1)
    # print(tensor_reduce_logpi_old)
    tensor_ratio = tf.math.exp(tensor_reduce_logpi - tensor_reduce_logpi_old)
    print(tensor_ratio)



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

def compute_advantage():
    pass



# params
num_batches = 2 #56
batch_size = 16 #8192
lr_actor = 3e-5
lr_critic = 5e-3
gamma = 0.99
val_train_iters = 32
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
    batch_logprobs = []
    batch_acts = []
    batch_rets = [] # discounted reward to go
    batch_dones = []
    batch_next_obs = []
    batch_advs = []
    rewards = []
    while True:
        env.render()
        logprob = tf.nn.log_softmax(actor_net(obs.reshape(1,-1))) # shape=(1, action_space.n)
        # print(logprob) # debug
        action = np.squeeze(tf.random.categorical(logits=logprob, num_samples=1)) # squeeze (1,1) to (1,)
        # print(action) # debug
        next_obs, rew, done, info = env.step(action)
        batch_obs.append(obs.copy())
        batch_logprobs.append(tf.squeeze(logprob))
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
            # batch_drtgs += list(reward_to_go(rewards, gamma))
            logging.info("\n---\nbatch: {}, episode: {} \nreturn: {} \n".format(batch+1, episode, ep_rets[-1]))
            obs, done, rewards = env.reset(), False, []
            if len(batch_obs) > batch_size:
                batch += 1
                break
    compute_loss_actor(batch_obs, batch_logprobs, batch_acts)
#     loss_actor, grads_actor = grad_actor(actor_net, batch_obs, batch_acts, batch_drtgs, batch_dones, batch_next_obs)
#     optimizer_actor.apply_gradients(zip(grads_actor, actor_net.trainable_variables))
#     for i in range(val_train_iters):
#         loss_critic, grads_critic = grad_critic(critic_net, batch_obs, batch_drtgs)
#         optimizer_critic.apply_gradients(zip(grads_critic, critic_net.trainable_variables))
#
#     logging.info("\n====\nbatch: {}, episode: {} \nloss_actor: {}, loss_critic: {} \nmean return: {} \n====\n".format(batch, episode, loss_actor, loss_critic, np.mean(batch_drtgs)))
#
# fig, ax = plt.subplots(figsize=(8, 6))
# fig.suptitle('Averaged Returns')
# ax.plot(ave_rets)
# plt.show()
