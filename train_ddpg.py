import sys
import os
import numpy as np
import random
import time
import gym
import matplotlib.pyplot as plt
from datetime import datetime

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
env = gym.make('LunarLanderContinuous-v2')

# Create policy network
actor_net = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=env.observation_space.shape),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(env.action_space.shape[0], activation='tanh')
    ]
)
actor_net_stable = tf.keras.models.clone_model(actor_net)
# Create Q Network
critic_net = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(env.observation_space.shape[0]+env.action_space.shape[0])),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1)
    ]
)
critic_net.summary()
critic_net_stable = tf.keras.models.clone_model(critic_net)

# Create Loss Object
loss_object = tf.keras.losses.MeanSquaredError()

# @tf.function
def compute_loss_critic(critic_net, critic_net_stable, actor_net_stable, batch_reps):
    # compute predict Q
    tensor_obs = tf.convert_to_tensor(batch_reps[0])
    # print("tensor_obs: {}".format(tensor_obs))
    tensor_acts = tf.cast(tf.convert_to_tensor(batch_reps[1]), tf.float32)
    # print("tensor_acts: {}".format(tensor_acts))
    q_preds = critic_net(tf.concat([tensor_obs, tensor_acts], axis=1))
    # print("q_preds: {}".format(q_preds))
    # compute target Q
    tensor_rews = tf.convert_to_tensor(batch_reps[2])
    # print("tensor_rews: {}".format(tensor_rews))
    tensor_dones = tf.cast(tf.convert_to_tensor(batch_reps[3]), tf.float32)
    tensor_obs_next = tf.convert_to_tensor(batch_reps[4])
    q_next = tf.stop_gradient(critic_net_stable(tf.concat([tensor_obs_next, actor_net_stable(tensor_obs_next)], axis=1)))
    # print("q_next: {}".format(q_next))
    q_targs = tf.reshape(tensor_rews, shape=[-1,1]) + gamma*(1.-tf.reshape(tensor_dones, shape=[-1,1]))*q_next
    # print("q_targs: {}".format(q_targs))
    # loss_Q = tf.math.reduce_mean((q_preds - q_targs)**2)
    loss_Q = loss_object(q_targs, q_preds)
    print("loss_Q: {}".format(loss_Q))

    return loss_Q

def compute_grads_critic(critic_net, critic_net_stable, actor_net_stable, batch_reps, optimizer_critic):
    with tf.GradientTape() as tape:
        loss_Q = compute_loss_critic(critic_net, critic_net_stable, actor_net_stable, batch_reps)
    grads_critic = tape.gradient(loss_Q, critic_net.trainable_variables)
    optimizer_critic.apply_gradients(zip(grads_critic, critic_net.trainable_variables))

def compute_loss_actor(actor_net, critic_net_stable, batch_reps):
    tensor_obs = tf.convert_to_tensor(batch_reps[0])
    a_preds = actor_net(tensor_obs)
    q_mu = critic_net_stable(tf.concat([tensor_obs, a_preds], axis=1))
    # print("q_mu: {}".format(q_mu))
    loss_mu = -tf.math.reduce_mean(q_mu)
    print("loss_mu: {}".format(loss_mu))

    return loss_mu

def compute_grads_actor(actor_net, critic_net_stable, batch_reps, optimizer_critic):
    with tf.GradientTape() as tape:
        loss_mu = compute_loss_actor(actor_net, critic_net_stable, batch_reps)
    grads_actor = tape.gradient(loss_mu, actor_net.trainable_variables)
    optimizer_actor.apply_gradients(zip(grads_actor, actor_net.trainable_variables))

# func store expeirience: (state, reward, done, next_state)
def store(replay_buffer, experience, buffer_size):
    if len(replay_buffer) >= buffer_size:
        replay_buffer.pop(0)
    replay_buffer.append(experience)
    logging.debug("experience: {} stored in replay buffer".format(experience))

# func sample batch experience
def sample_batch(replay_buffer, batch_size):
    if len(replay_buffer) < batch_size:
        minibatch = replay_buffer
    else:
        minibatch = random.sample(replay_buffer, batch_size)
    batch_reps = list(zip(*minibatch))

    return batch_reps

# params
num_episodes = 1000
num_steps = env.spec.max_episode_steps
gamma = 0.99
polyak = 0.997
warmup = 100
buffer_size = 100000
batch_size = 4096
act_noise = 0.1
replay_buffer = []
# train_loss = []
step_counter = 0
# Create Optimizer
optimizer_critic = tf.keras.optimizers.Adam(learning_rate=1e-3)
optimizer_actor = tf.keras.optimizers.Adam(learning_rate=1e-4)
model_dir = './training_models/ddpg/'+datetime.now().strftime("%Y-%m-%d-%H-%M")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
ep_rets, ave_rets = [], []
start_time = time.time()
if __name__ == '__main__':
    for ep in range(num_episodes):
        obs, done, rewards = env.reset(), False, []
        for st in range(num_steps):
            # env.render()
            # exploit vs explore
            if ep >= warmup:
                act = tf.stop_gradient(tf.squeeze(actor_net(obs.reshape(1,-1)))).numpy() + act_noise*np.random.randn(env.action_space.shape[0])
            else:
                act = env.action_space.sample()
            next_obs, rew, done, info = env.step(act)
            step_counter += 1
            rewards.append(rew)
            store(replay_buffer=replay_buffer, experience=(obs.copy(), act, np.float32(rew), done, next_obs.copy()), buffer_size=buffer_size)
            batch_reps = sample_batch(replay_buffer, batch_size)
            if ep >= warmup:
                # train_one_batch
                # loss_Q_old = compute_loss_critic(critic_net, critic_net_stable, actor_net_stable, batch_reps)
                # loss_mu_old = compute_loss_actor(actor_net, critic_net_stable, batch_reps)
                # print("loss_Q: {}, loss_mu: {}".format(loss_Q, loss_mu))
                compute_grads_critic(critic_net, critic_net_stable, actor_net_stable, batch_reps, optimizer_critic)
                compute_grads_actor(actor_net, critic_net_stable, batch_reps, optimizer_actor)
                # loss_Q = compute_loss_critic(critic_net, critic_net_stable, actor_net_stable, batch_reps)
                # loss_mu = compute_loss_actor(actor_net, critic_net_stable, batch_reps)
                # polyak averaging
                weights_critic_update = []
                for w_q, w_q_targ in zip(critic_net.get_weights(), critic_net_stable.get_weights()):
                    w_q_upd = polyak*w_q_targ
                    w_q_upd = w_q_upd + (1 - polyak)*w_q
                    weights_critic_update.append(w_q_upd)
                critic_net_stable.set_weights(weights_critic_update)
                weights_actor_update = []
                for w_mu, w_mu_targ in zip(actor_net.get_weights(), actor_net_stable.get_weights()):
                    w_mu_upd = polyak*w_mu_targ
                    w_mu_upd = w_mu_upd + (1 - polyak)*w_mu
                    weights_actor_update.append(w_mu_upd)
                actor_net_stable.set_weights(weights_actor_update)
            logging.debug("\n-\nepisode: {}, step: {} \naction: {} \nobs: {}, \nreward: {}".format(ep+1, st+1, act, obs, rew))
            obs = next_obs.copy()
            if done:
                ep_rets.append(sum(rewards))
                ave_rets.append(sum(ep_rets)/len(ep_rets))
                logging.info(
                    "\n================================================================\nEpisode: {}, Step: {} \nEpReturns: {} \nAveEpReturn: {} \nTime: {} \n================================================================\n".format(
                        ep+1,
                        st+1,
                        ep_rets[-1],
                        ave_rets[-1],
                        time.time()-start_time
                    )
                )
                break

# Save final ckpt
# save_path = ckpt_manager.save()
# save model
actor_net.save(os.path.join(model_dir, str(step_counter)))
critic_net.save(os.path.join(model_dir, str(step_counter)))

# plot returns
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle('Averaged Returns')
ax.plot(ave_rets)
plt.show()
