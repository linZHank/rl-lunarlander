import sys
import os
import numpy as np
import random
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
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)


# Create LunarLander env
env = gym.make('LunarLanderContinuous-v2')

# Create policy network
actor_net = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=env.observation_space.shape),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(env.action_space.shape[0])
    ]
)
actor_net_stable = tf.keras.models.clone_model(actor_net)
# Create Q Network
critic_net = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(env.observation_space.shape[0]+env.action_space.shape[0])),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ]
)
critic_net.summary()
critic_net_stable = tf.keras.models.clone_model(critic_net)

# Create Loss Object
loss_object = tf.keras.losses.MeanSquaredError()

# @tf.function
def compute_loss_actor(critic_net, batch_reps):
    # compute predict Q
    tensor_obs = tf.convert_to_tensor(batch_reps[0])
    print("tensor_obs: {}".format(tensor_obs))
    tensor_acts = tf.cast(tf.convert_to_tensor(batch_reps[1]), tf.float32)
    print("tensor_acts: {}".format(tensor_acts))
    q_preds = critic_net(tf.concat([tensor_obs, tensor_acts], axis=1))
    print("q_preds: {}".format(q_preds))

    # return loss_object(batch_reduce_qvals_target, batch_reduce_qvals_pred)


def grad_critic(critic_net, batch_reps):
    with tf.GradientTape() as tape:
        loss_Q = compute_loss_critic(qnet_active, qnet_stable, batch_reps)

    return loss_Q, tape.gradient(q_loss, qnet_active.trainable_variables)

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
num_episodes = 3
num_steps = env.spec.max_episode_steps
gamma = 0.99
decay_rate = 0.99
warmup = 0
buffer_size = 100000
batch_size = 81 #92
act_noise = 0.1
replay_buffer = []
# metrics_mse = tf.keras.metrics.MeanSquaredError()
# train_loss = []
step_counter = 0
update_steps = 81 #92
# Create Optimizer
optimizer_actor = tf.keras.optimizers.Adam(learning_rate=1e-3)
optimizer_critic = tf.keras.optimizers.Adam(learning_rate=1e-3)
# Create Checkpoint
# checkpoint_dir = './training_checkpoints/dqn'
# if not os.path.exists(checkpoint_dir):
#     os.makedirs(checkpoint_dir)
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=qnet_active)
# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=200)
# ckpt.restore(ckpt_manager.latest_checkpoint)
model_dir = './training_models/dqn/'+datetime.now().strftime("%Y-%m-%d-%H-%M")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
ep_rets, ave_rets = [], []
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
            store(replay_buffer=replay_buffer, experience=(obs.copy(), act, np.float32(rew), done, next_obs.copy()), buffer_size=buffer_size)
            batch_reps = sample_batch(replay_buffer, batch_size)
            if ep >= warmup:
                # train_one_batch
                compute_loss_actor(critic_net, batch_reps)
            step_counter += 1
            rewards.append(rew)
            logging.info("\n-\nepisode: {}, step: {} \naction: {} \nobs: {}, \nreward: {}".format(ep+1, st+1, act, obs, rew))
            obs = next_obs.copy()
            if done:
                ep_rets.append(sum(rewards))
                ave_rets.append(sum(ep_rets)/len(ep_rets))
                logging.info("\n---\nepisode: {} \nepisode return: {}, averaged return: {} \n---\n".format(ep+1, ep_rets[-1], ave_rets[-1]))
                break

# Save final ckpt
# save_path = ckpt_manager.save()
# save model
# actor_net.save(os.path.join(model_dir, str(step_counter)))
# critic_net.save(os.path.join(model_dir, str(step_counter)))

# Plot returns and loss
# fig, axes = plt.subplots(2, figsize=(12, 8))
# fig.suptitle('Metrics')
# axes[0].set_xlabel("Episode")
# axes[0].set_ylabel("Averaged Return")
# axes[0].plot(ave_rets)
# # axes[1].set_xlabel("Steps")
# # axes[1].set_ylabel("Loss")
# # axes[1].plot(train_loss)
# plt.show()
