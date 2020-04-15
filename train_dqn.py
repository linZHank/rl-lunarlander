import sys
import os
import numpy as np
import random
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
qnet_active = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=env.observation_space.shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(env.action_space.n)
    ]
)
qnet_active.summary()
qnet_stable = tf.keras.models.clone_model(qnet_active)

# Create Loss Object
loss_object = tf.keras.losses.MeanSquaredError()

def compute_loss(qnet_active, qnet_stable, batch_reps):
    # compute predict Q
    batch_obs = tf.convert_to_tensor((batch_reps[0]))
    batch_qvals = qnet_active(batch_obs)
    batch_acts = tf.convert_to_tensor(batch_reps[1])
    batch_acts_onehot = tf.one_hot(batch_acts, depth=env.action_space.n)
    batch_qvals_pred = tf.math.multiply(batch_qvals, batch_acts_onehot)
    batch_reduce_qvals_pred = tf.math.reduce_sum(batch_qvals_pred, axis=1)
    # compute target Q
    batch_next_obs = tf.convert_to_tensor(batch_reps[-1])
    batch_next_qvals = qnet_stable(batch_next_obs)
    batch_next_ids = tf.math.argmax(qnet_active(batch_next_obs), axis=1)
    batch_next_ids_onehot = tf.one_hot(batch_next_ids, depth=env.action_space.n)
    batch_qvals_update = tf.math.multiply(batch_next_qvals, batch_next_ids_onehot)
    batch_reduce_qvals_update = tf.math.reduce_sum(batch_qvals_update, axis=1)
    batch_rews = tf.convert_to_tensor(batch_reps[2])
    batch_dones = np.array(batch_reps[3])
    batch_reduce_qvals_target = batch_rews + (1.-batch_dones)*gamma*batch_reduce_qvals_update # Double DQN

    return loss_object(batch_reduce_qvals_target, batch_reduce_qvals_pred)


def grad(qnet_active, qnet_stable, batch_reps):
    with tf.GradientTape() as tape:
        q_loss = compute_loss(qnet_active, qnet_stable, batch_reps)

    return q_loss, tape.gradient(q_loss, qnet_active.trainable_variables)

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

def epsilon_greedy(qvals, epsilon):
    if np.random.uniform() > epsilon:
        act_id = np.argmax(qvals)
    else:
        act_id = np.random.randint(env.action_space.n)

    return act_id

################################### Test Loss ###################################
# epsilon = 0.5
# replay_buffer = []
# buffer_size = 10000
# batch_size = 100
# gamma = 0.99
# obs = env.reset()
# for _ in range(100):
#     env.render()
#     qvals = np.squeeze(qnet_active(obs.reshape(1,-1)).numpy())
#     # print(qvals)
#     action = epsilon_greedy(qvals, epsilon)
#     # print(action)
#     next_obs, rew, done, info = env.step(action)
#     store(replay_buffer=replay_buffer, experience=(obs, action, np.float32(rew), done, next_obs), buffer_size=buffer_size)
#     batch_reps = sample_batch(replay_buffer, batch_size)
#     # compute predict Q
#     batch_obs = tf.convert_to_tensor((batch_reps[0]))
#     batch_qvals = qnet_active(batch_obs)
#     batch_acts = tf.convert_to_tensor(batch_reps[1])
#     batch_acts_onehot = tf.one_hot(batch_acts, depth=env.action_space.n)
#     batch_qvals_pred = tf.math.multiply(batch_qvals, batch_acts_onehot)
#     batch_reduce_qvals_pred = tf.math.reduce_sum(batch_qvals_pred, axis=1)
#     # compute target Q
#     batch_next_obs = tf.convert_to_tensor(batch_reps[-1])
#     batch_next_qvals = qnet_stable(batch_next_obs)
#     batch_next_ids = tf.math.argmax(qnet_active(batch_next_obs), axis=1)
#     batch_next_ids_onehot = tf.one_hot(batch_next_ids, depth=env.action_space.n)
#     batch_qvals_update = tf.math.multiply(batch_next_qvals, batch_next_ids_onehot)
#     batch_reduce_qvals_update = tf.math.reduce_sum(batch_qvals_update, axis=1)
#     batch_rews = tf.convert_to_tensor(batch_reps[2])
#     batch_dones = np.array(batch_reps[3])
#     batch_reduce_qvals_target = batch_rews + (1.-batch_dones)*gamma*batch_reduce_qvals_update # Double DQN
#     q_loss = loss_object(batch_reduce_qvals_target, batch_reduce_qvals_pred)
#     print(q_loss)
##################################################################################


# params
num_episodes = 10000
num_steps = 10000
gamma = 0.99
decay_rate = 0.99
warmup = 128
epsilon = 1.
buffer_size = 100000
batch_size = 8192
replay_buffer = []
# metrics_mse = tf.keras.metrics.MeanSquaredError()
# train_loss = []
step_counter = 0
update_steps = 8192
# Create Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
# Create Checkpoint
checkpoint_dir = './training_checkpoints/dqn'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=qnet_active)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=200)
ckpt.restore(manager.latest_checkpoint)
ep_rets, ave_rets = [], []
if __name__ == '__main__':
    for ep in range(num_episodes):
        obs, done, rewards = env.reset(), False, []
        if ep >= warmup:
            epsilon *= decay_rate
            epsilon  = np.clip(epsilon, 0.1, 1)
        for st in range(num_steps):
            # env.render()
            qvals = np.squeeze(qnet_active(obs.reshape(1,-1)).numpy())
            action = epsilon_greedy(qvals, epsilon)
            next_obs, rew, done, info = env.step(action)
            store(replay_buffer=replay_buffer, experience=(obs.copy(), action, np.float32(rew), done, next_obs.copy()), buffer_size=buffer_size)
            batch_reps = sample_batch(replay_buffer, batch_size)
            if ep >= warmup:
                # train_one_batch
                q_loss, grads = grad(qnet_active, qnet_stable, batch_reps)
                logging.debug("Initial Loss: {}".format(q_loss))
                optimizer.apply_gradients(zip(grads, qnet_active.trainable_variables))
                logging.debug("After gradient Loss: {}".format(compute_loss(qnet_active, qnet_stable, batch_reps)))
                # train_loss.append(q_loss)
                ckpt.step.assign_add(1)
                if not int(ckpt.step) % 20000:
                    save_path = ckpt_manager.save()
            # update qnet_stable every update_steps
            step_counter += 1
            if not step_counter % update_steps:
                qnet_stable.set_weights(qnet_active.get_weights())
            rewards.append(rew)
            logging.info("\n-\nepisode: {}, step: {}, epsilon: {} \naction: {} \nobs: {}, \nreward: {}".format(ep+1, st+1, epsilon, action, obs, rew))
            obs = next_obs.copy()
            if done:
                ep_rets.append(sum(rewards))
                ave_rets.append(sum(ep_rets)/len(ep_rets))
                logging.info("\n---\nepisode: {} \nepisode return: {}, averaged return: {} \n---\n".format(ep+1, ep_rets[-1], ave_rets[-1]))
                break

# Plot returns and loss
fig, axes = plt.subplots(2, figsize=(12, 8))
fig.suptitle('Metrics')
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Averaged Return")
axes[0].plot(ave_rets)
axes[1].set_xlabel("Steps")
axes[1].set_ylabel("Loss")
axes[1].plot(train_loss)
plt.show()

# Save final ckpt
save_path = ckpt_manager.save()
