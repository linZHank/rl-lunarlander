import sys
import os
import numpy as np
import time
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


# create LunarLander env
# env = gym.make('CartPole-v1')
env = gym.make('LunarLander-v2')

# Create Policy Network
actor_net = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=env.observation_space.shape),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(env.action_space.n)
    ]
)
actor_net.summary()

def compute_loss_actor(actor_net, buffer_obs, buffer_logprobs, buffer_acts, buffer_rets):
    tensor_obs = tf.convert_to_tensor(buffer_obs)
    # print("obs: {}".format(tensor_obs))
    tensor_acts = tf.convert_to_tensor(buffer_acts)
    # print("acts: {}".format(tensor_acts))
    tensor_rets = tf.cast(tf.convert_to_tensor(buffer_rets), tf.float32)
    # print("rets: {}".format(tensor_rets))
    logits = actor_net(tensor_obs)
    pis = tf.nn.softmax(logits)
    # print("pis: {}".format(pis))
    logprobs = tf.nn.log_softmax(logits)
    # print("logprobs: {}".format(logprobs))
    acts_onehot = tf.one_hot(tensor_acts, depth=env.action_space.n)
    logpis = tf.math.multiply(logprobs, acts_onehot)
    # print("logpis: {}".format(logpis))
    sum_logpis = tf.math.reduce_sum(logpis, axis=1)
    # print("sum_logpis: {}".format(sum_logpis))
    obj = tf.math.multiply(sum_logpis, tensor_rets)
    loss_actor = -tf.math.reduce_mean(obj)
    # compute KL-divergence and Entropy
    logprobs_old = tf.convert_to_tensor(buffer_logprobs)
    kld_approx = tf.math.reduce_mean(logprobs_old - logprobs)
    entropy = tf.math.reduce_sum(tf.math.multiply(-pis, logprobs))
    actor_info = dict(kl=kld_approx, entropy=entropy)

    return loss_actor, actor_info

def grad_actor(actor_net, buffer_obs, buffer_logprobs, buffer_acts, buffer_rets):
    with tf.GradientTape() as tape:
        loss_actor, _ = compute_loss_actor(actor_net, buffer_obs, buffer_logprobs, buffer_acts, buffer_rets)

    return loss_actor, tape.gradient(loss_actor, actor_net.trainable_variables)

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
model_dir = './training_models/vpg/'
save_freq = 100
num_epochs = 2000
buffer_size = 4096
lr_actor = 1e-4
gamma = 0.99
# Create Optimizers
optimizer_actor = tf.keras.optimizers.Adam(learning_rate=lr_actor)
# prepare for train
obs, done = env.reset(), False
ep_rets, ave_rets = [], []
epoch = 0
episode = 0
step = 0
start_time = time.time()
for e in range(num_epochs):
    buffer_obs = []
    buffer_acts = []
    buffer_advs = []
    buffer_rews = []
    buffer_rets = [] # discounted reward to go
    buffer_logprobs = []
    while True:
        # env.render()
        logprob = tf.stop_gradient(tf.nn.log_softmax(actor_net(obs.reshape(1,-1))))
        # logging.debug("action log probility: {}".format(logprob))
        act = np.squeeze(tf.random.categorical(logits=logprob, num_samples=1)) # squeeze (1,1) to (1,)
        # logging.debug("sampled action: {}".format(act))
        next_obs, rew, done, info = env.step(act)
        # store experience into buffer
        buffer_obs.append(obs.copy())
        buffer_acts.append(act)
        buffer_rews.append(rew)
        buffer_logprobs.append(np.squeeze(logprob.numpy()))
        logging.debug("\n-\nepoch: {}, episode: {}, step: {}, epoch length: {} \nobs: {} \naction: {} \nnext obs: {}".format(epoch+1, episode+1, step+1, len(buffer_obs), obs, act, next_obs))
        obs = next_obs.copy()
        step += 1
        if done:
            episode += 1
            # compute discounted reward to go
            rews = buffer_rews[-step:] # rewards in last episode
            rtgs = reward_to_go(rews, gamma)
            buffer_rets += list(rtgs)
            ep_rets.append(sum(rews))
            ave_rets.append(sum(ep_rets)/len(ep_rets))
            logging.debug("\n---\nEpoch: {}, TotalEpisode: {} \nEpisodeReturn: {} \nAveReturn: {} \n".format(epoch, episode, ep_rets[-1], ave_rets[-1]))
            obs, done, rew = env.reset(), False, []
            step = 0
            if len(buffer_obs) > buffer_size:
                epoch += 1
                break
    # run policy gradient
    loss_actor_old, actor_info_old = compute_loss_actor(actor_net, buffer_obs, buffer_logprobs, buffer_acts, buffer_rets)
    loss_actor, grads_actor = grad_actor(actor_net, buffer_obs, buffer_logprobs, buffer_acts, buffer_rets)
    optimizer_actor.apply_gradients(zip(grads_actor, actor_net.trainable_variables))
    loss_actor, actor_info = compute_loss_actor(actor_net, buffer_obs, buffer_logprobs, buffer_acts, buffer_rets)
    logging.info(
        "\n================================================================\nEpoch: {} \nLossPiOld: {}, LossPiUpdated: {} \nKLDivergence: {} \nEntropy: {} \nAveEpReturn: {} \nTime: {} \n================================================================\n".format(
            epoch,
            loss_actor_old,
            loss_actor,
            actor_info['kl'],
            actor_info['entropy'],
            ave_rets[-1],
            time.time()-start_time
        )
    )

# save model
model_path = os.path.join(model_dir, env.spec.id, 'models', str(epoch)+'.h5')
if not os.path.exists(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))
actor_net.save(model_path)
# plot averaged returns
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle('Averaged Returns')
ax.plot(ave_rets)
plt.show()

# Test trained model
input("Press ENTER to test lander...")
num_episodes = 10
num_steps = env.spec.max_episode_steps
ep_rets, ave_rets = [], []
for ep in range(num_episodes):
    obs, done, rewards = env.reset(), False, []
    for st in range(num_steps):
        env.render()
        logprob = tf.stop_gradient(tf.nn.log_softmax(actor_net(obs.reshape(1,-1))))
        act = np.squeeze(tf.random.categorical(logits=logprob, num_samples=1)) # squeeze (1,1) to (1,)
        next_obs, rew, done, info = env.step(act)
        rewards.append(rew)
        logging.info("\n-\nepisode: {}, step: {} \naction: {} \nobs: {}, \nreward: {}".format(ep+1, st+1, act, obs, rew))
        obs = next_obs.copy()
        if done:
            ep_rets.append(sum(rewards))
            ave_rets.append(sum(ep_rets)/len(ep_rets))
            logging.info("\n---\nepisode: {} \nepisode return: {}, averaged return: {} \n---\n".format(ep+1, ep_rets[-1], ave_rets[-1]))
            break
