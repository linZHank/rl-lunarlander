import sys
import os
import numpy as np
import gym
import time
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
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
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
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(env.action_space.n)
    ]
)
actor_net.summary()

# Create value Network
critic_net = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=env.observation_space.shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ]
)
critic_net.summary()

def compute_loss_actor(actor_net, buffer_obs, buffer_logprobs, buffer_acts, buffer_advs):
    tensor_obs = tf.convert_to_tensor(buffer_obs)
    tensor_acts = tf.convert_to_tensor(buffer_acts)
    advs = tf.convert_to_tensor(buffer_advs)
    # next 3 lines apply advantage normalization trick
    mean_advs = tf.math.reduce_mean(advs)
    std_advs = tf.math.reduce_std(advs)
    tensor_advs = (advs-mean_advs)/std_advs
    # print("tensor_obs: {}".format(tensor_obs))
    # print("tensor_acts: {}".format(tensor_acts))
    # print("tensor_advs: {}".format(tensor_advs))
    logits = actor_net(tensor_obs)
    probs = tf.nn.softmax(logits)
    logprobs = tf.math.log(probs)
    # logprobs = tf.nn.log_softmax(logits)
    # print("acts log probs: {}".format(logprobs))
    acts_onehot = tf.one_hot(tensor_acts, depth=env.action_space.n)
    logpis = tf.math.multiply(logprobs, acts_onehot)
    tensor_logpis = tf.math.reduce_sum(logpis, axis=1)
    # print("log pi: {}".format(tensor_logpis))
    obj = tf.math.multiply(tensor_logpis, tensor_advs)
    # print("objective: {}".format(obj))
    loss_actor = -tf.math.reduce_mean(obj)
    # print("loss_pi: {}".format(loss_actor))
    # compute KL-divergence and Entropy
    logprobs_old = tf.convert_to_tensor(buffer_logprobs)
    kld_approx = tf.math.reduce_mean(logprobs_old - logprobs)
    entropy = tf.math.reduce_sum(tf.math.multiply(-probs, logprobs))
    actor_info = dict(kl=kld_approx, entropy=entropy)

    return loss_actor, actor_info

def grad_actor(actor_net, buffer_obs, buffer_logprobs, buffer_acts, buffer_advs):
    with tf.GradientTape() as tape:
        loss_actor, _ = compute_loss_actor(actor_net, buffer_obs, buffer_logprobs, buffer_acts, buffer_advs)

    return loss_actor, tape.gradient(loss_actor, actor_net.trainable_variables)

def compute_loss_critic(critic_net, buffer_obs, buffer_rets):
    vals_pred = tf.squeeze(critic_net(tf.convert_to_tensor(buffer_obs)))
    vals_target = tf.convert_to_tensor(buffer_rets)
    loss_critic = tf.keras.losses.MSE(vals_target, vals_pred)
    # print(loss_critic)

    return loss_critic

def grad_critic(critic_net, buffer_obs, buffer_rets):
    with tf.GradientTape() as tape:
        loss_critic = compute_loss_critic(critic_net, buffer_obs, buffer_rets)

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
model_dir = './training_models/actor-critic'
save_freq = 100
num_epochs = 2000
buffer_size = 4096
lr_actor = 3e-4
lr_critic = 1e-3
gamma = 0.999
lam = 0.97
val_train_iters = 80
# Create Optimizers
optimizer_actor = tf.keras.optimizers.Adam(learning_rate=lr_actor)
optimizer_critic = tf.keras.optimizers.Adam(learning_rate=lr_critic)

#
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
    buffer_vals = []
    buffer_logprobs = []
    while True:
        # env.render()
        val = critic_net(obs.reshape(1,-1))
        # logging.debug("value: {}".format(val))
        logprob = tf.nn.log_softmax(actor_net(obs.reshape(1,-1)))
        # logging.debug("action log probility: {}".format(logprob))
        act = np.squeeze(tf.random.categorical(logits=logprob, num_samples=1)) # squeeze (1,1) to (1,)
        # logging.debug("sampled action: {}".format(act))
        next_obs, rew, done, info = env.step(act)
        # store experience into buffer
        buffer_obs.append(obs.copy())
        buffer_acts.append(act)
        buffer_rews.append(rew)
        buffer_vals.append(np.squeeze(val.numpy()))
        buffer_logprobs.append(np.squeeze(logprob.numpy()))
        logging.debug("\n-\nepoch: {}, episode: {}, step: {}, epoch length: {} \nobs: {} \naction: {} \nnext obs: {}".format(epoch+1, episode+1, step+1, len(buffer_obs), obs, act, next_obs))
        obs = next_obs.copy()
        step += 1
        if done:
            episode += 1
            # compute discounted reward to go
            rews = buffer_rews[-step:] # rewards in last episode
            rtgs = reward_to_go(rews, gamma)
            # compute advantages
            vals = buffer_vals[-step:]
            nvals = vals[1:]+[0] # values of next states
            advs = [np.float32(rtgs[i]+gamma*lam*nvals[i]-vals[i]) for i in range(len(rews))]
            buffer_rets += list(rtgs)
            # logging.debug("discounted returns: {}".format(buffer_rets))
            buffer_advs += advs
            # logging.debug("values: {} \nadvantages: {}".format(buffer_vals, buffer_advs))
            ep_rets.append(sum(rews))
            ave_rets.append(sum(ep_rets)/len(ep_rets))
            # batch_drtgs += list(reward_to_go(rewards, gamma))
            logging.debug("\n---\nEpoch: {}, TotalEpisode: {} \nEpisodeReturn: {} \nAveReturn: {} \n".format(epoch, episode, ep_rets[-1], ave_rets[-1]))
            obs, done, rew = env.reset(), False, []
            step = 0
            if len(buffer_obs) > buffer_size:
                epoch += 1
                break
    loss_pi_old, actor_info_old = compute_loss_actor(actor_net, buffer_obs, buffer_logprobs, buffer_acts, buffer_advs)
    loss_v_old = compute_loss_critic(critic_net, buffer_obs, buffer_rets)
    # run policy gradient
    loss_actor, grads_actor = grad_actor(actor_net, buffer_obs, buffer_logprobs, buffer_acts, buffer_advs)
    optimizer_actor.apply_gradients(zip(grads_actor, actor_net.trainable_variables))
    loss_actor, actor_info = compute_loss_actor(actor_net, buffer_obs, buffer_logprobs, buffer_acts, buffer_advs)
    # fit value
    for i in range(val_train_iters):
        loss_critic, grads_critic = grad_critic(critic_net, buffer_obs, buffer_rets)
        optimizer_critic.apply_gradients(zip(grads_critic, critic_net.trainable_variables))
        loss_critic = compute_loss_critic(critic_net, buffer_obs, buffer_rets)
#
    logging.info(
        "\n================================================================\nEpoch: {} \nLossPiOld: {}, LossPiUpdated: {} \nKLDivergence: {} \nEntropy: {} \nLossVOld: {}, LossVUpdated: {} \nAveEpReturn: {} \nTime: {} \n================================================================\n".format(
            epoch,
            loss_pi_old,
            loss_actor,
            actor_info['kl'],
            actor_info['entropy'],
            loss_v_old,
            loss_critic,
            ave_rets[-1],
            time.time()-start_time
        )
    )
    # save models
    if not epoch % save_freq or epoch==num_epochs:
        actor_net_path = os.path.join(model_dir, 'actor_net', str(epoch)+'.h5')
        critic_net_path = os.path.join(model_dir, 'critic_net', str(epoch)+'.h5')
        if not os.path.exists(os.path.dirname(actor_net_path)):
            os.makedirs(os.path.dirname(actor_net_path))
        if not os.path.exists(os.path.dirname(critic_net_path)):
            os.makedirs(os.path.dirname(critic_net_path))
        actor_net.save(actor_net_path)
        critic_net.save(critic_net_path)
        logging.info("\n$$$$$$$$$$$$$$$$\nmodels saved at {}\n$$$$$$$$$$$$$$$$\n".format(model_dir))


fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle('Averaged Returns')
ax.plot(ave_rets)
plt.show()
