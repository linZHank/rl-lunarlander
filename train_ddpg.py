import sys
import os
from copy import deepcopy
import numpy as np
import scipy.signal
import random
import time
import gym
import matplotlib.pyplot as plt
from datetime import datetime
import logging

import tensorflow as tf
print(tf.__version__)
import tensorflow_probability as tfp
tfd = tfp.distributions
################################################################
"""
Unnecessary initial settings
"""
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
# set log level
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)
################################################################


################################################################
"""
instantiate env
"""
env = gym.make('LunarLanderContinuous-v2')
# env = gym.make('Pendulum-v0')
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high
################################################################


################################################################
"""
Build actor_net, critic_net
"""
def mlp(sizes, activation, output_activation=None):
    inputs = tf.keras.Input(shape=(sizes[0],))
    x = tf.keras.layers.Dense(sizes[1], activation=activation)(inputs)
    for i in range(2,len(sizes)-1):
        x = tf.keras.layers.Dense(sizes[i], activation=activation)(x)
    outputs = tf.keras.layers.Dense(sizes[-1], activation=output_activation)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

class MLPActor(tf.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.actor_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation='tanh')
        self.act_limit = act_limit

    def __call__(self, obs):
        return self.act_limit*self.actor_net(obs)

class MLPCritic(tf.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.critic_net = mlp([obs_dim+act_dim] + list(hidden_sizes) + [1], activation)

    # @tf.function
    def __call__(self, obs, act):
        q_val = self.critic_net(tf.concat([obs, act], axis=-1))
        return tf.squeeze(q_val, axis=-1)

class MLPActorCritic(tf.Module):
    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=(256,256), activation='tanh'):
        super().__init__()
        # obs_dim = observation_space.shape[0]
        # act_dim = action_space.shape[0]
        # act_limit = action_space.high
        self.pi = MLPActor(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=hidden_sizes, activation=activation, act_limit=act_limit)
        self.q = MLPCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=hidden_sizes, activation=activation)

    def act(self, obs):
        return self.pi(obs).numpy()[0]
################################################################


################################################################
"""
Compute losses and gradients
"""
def compute_actor_gradients(ac, actor_optimizer, data):
    with tf.GradientTape() as tape:
        tape.watch(ac.pi.trainable_variables)
        obs_batch = data['obs']
        q_pi = ac.q(obs_batch, ac.pi(obs_batch))
        loss_actor = -tf.math.reduce_mean(q_pi)
    grads_actor = tape.gradient(loss_actor, ac.pi.trainable_variables)
    actor_optimizer.apply_gradients(zip(grads_actor, ac.pi.trainable_variables))

    return loss_actor

def compute_critic_gradients(ac, ac_targ, critic_optimizer, data):
    with tf.GradientTape() as tape:
        tape.watch(ac.q.trainable_variables)
        obs_batch, act_batch, rew_batch, done_batch, next_obs_batch = data['obs'], data['act'], data['rew'], data['done'], data['next_obs']
        q_pred = ac.q(obs_batch, act_batch)
        # q_pi_targ = ac_targ.q(next_obs_batch, ac_targ.pi(obs_batch))
        q_targ = rew_batch + gamma*(1 - done_batch)*(ac_targ.q(next_obs_batch, ac_targ.pi(obs_batch)))
        loss_critic = tf.keras.losses.MSE(q_targ, q_pred)
    grads_critic = tape.gradient(loss_critic, ac.q.trainable_variables)
    critic_optimizer.apply_gradients(zip(grads_critic, ac.q.trainable_variables))

    return loss_critic

# def update(ac, actor_optimizer, critic_optimizer, date):
#     loss_actor = compute_actor_gradients(ac=ac, actor_optimizer=actor_optimizer, data=data)
#     print("loss_actor: {}".format(loss_actor))

    # if not i%policy_delay:
    #     compute_pi_grads(act_limit, batch)
    #     # polyak averaging
    #     weights_critic1_update = []
    #     for w_q1, w_q1_targ in zip(critic_net_1.get_weights(), critic_net_1_targ.get_weights()):
    #         w_q1_upd = polyak*w_q1_targ
    #         w_q1_upd = w_q1_upd + (1 - polyak)*w_q1
    #         weights_critic1_update.append(w_q1_upd)
    #     critic_net_1_targ.set_weights(weights_critic1_update)
    #     weights_critic2_update = []
    #     for w_q2, w_q2_targ in zip(critic_net_2.get_weights(), critic_net_2_targ.get_weights()):
    #         w_q2_upd = polyak*w_q2_targ
    #         w_q2_upd = w_q2_upd + (1 - polyak)*w_q2
    #         weights_critic2_update.append(w_q2_upd)
    #     critic_net_2_targ.set_weights(weights_critic2_update)
    #     weights_actor_update = []
    #     for w_pi, w_pi_targ in zip(actor_net.get_weights(), actor_net_targ.get_weights()):
    #         w_pi_upd = polyak*w_pi_targ
    #         w_pi_upd = w_pi_upd + (1 - polyak)*w_pi
    #         weights_actor_update.append(w_pi_upd)
    #     actor_net_targ.set_weights(weights_actor_update)
# def compute_critic_gradients(ac, critic_optimizer, data):
#     obs_batch, act_batch, rew_batch, done_batch, next_obs_batch = data['obs'], data['act'], data['rew'], data['done'], data['next_obs']
#     with tf.GradientTape() as tape:
#         tape.watch(ac.q.trainable_variables)
#
#
#     return loss_pi, pi_info
# #
# def compute_critic_gradients(data):
#     obs, ret = data['obs'], data['ret']
#     with tf.GradientTape() as tape:
#         tape.watch(ac.critic.trainable_variables)
#         loss_v = tf.keras.losses.MSE(ret, ac.critic(obs))
#     critic_grads = tape.gradient(loss_v, ac.critic.trainable_variables)
#     critic_optimizer.apply_gradients(zip(critic_grads, ac.critic.trainable_variables))
#
#     return loss_v
#
# def update(buffer):
#     data = buffer.get()
#     for i in range(train_pi_iters):
#         loss_pi, pi_info = compute_actor_gradients(data)
#         kl = pi_info['kl']
#         if kl > 1.5 * target_kl:
#                 print('Early stopping at step %d due to reaching max kl.'%i)
#                 break
#     for j in range(train_v_iters):
#         loss_v = compute_critic_gradients(data)
#
#     return loss_pi, pi_info, loss_v
################################################################


################################################################
"""
On-policy Replay Buffer for PPO
"""
class DDPGReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     next_obs=self.next_obs_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: tf.convert_to_tensor(v, dtype=tf.float32) for k,v in batch.items()}
################################################################


################################################################
"""
paramas
"""
steps_per_epoch = 400
epochs = 10
replay_size = int(1e6)
gamma = 0.99
polyak = 0.995
pi_lr = 1e-3
q_lr = 1e-3
batch_size = 100
start_step = 100
update_after = 100
update_freq = 50
act_noise = 0.1
max_ep_len=env.spec.max_episode_steps
save_freq=10
# instantiate actor-critic and replay buffer
ac = MLPActorCritic(obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit)
ac_targ = MLPActorCritic(obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit)
ac_targ.pi.actor_net = tf.keras.models.clone_model(ac.pi.actor_net)
ac_targ.q.critic_net = tf.keras.models.clone_model(ac.q.critic_net)
buffer = DDPGReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
# create optimizer
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=pi_lr)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=q_lr)
model_dir = './models/ddpg'
# get action function
def get_action(obs, noise_scale, act_limit):
    action = ac.act(obs)
    action += noise_scale * np.random.randn(act_dim)
    return np.clip(action, -act_limit, act_limit)
################################################################


################################################################
"""
Main
"""
# Prepare for interaction with environment
start_time = time.time()
total_steps = steps_per_epoch*epochs
obs, ep_ret, ep_len = env.reset(), 0, 0
total_episodes = 0
stepwise_rewards, episodic_returns, averaged_returns = [], [], []
# main loop
for t in range(total_steps):
    if t > start_step:
        act = get_action(obs=obs.reshape(1,-1), noise_scale=act_noise, act_limit=act_limit)
        # act = ac.act(obs)
        # act += act_noise*np.random.randn(act_dim)
        # act = np.clip(act, -act_limit, act_limit)
    else:
        act = env.action_space.sample()
    next_obs, rew, done, _ = env.step(act)
    ep_ret += rew
    ep_len += 1
    print("Step: {} \nobs: {} \nact: {} \nrew: {}".format(t, obs, act, rew))
    # ignore artificial terminal
    done = False if ep_len==max_ep_len else done
    buffer.store(obs, act, rew, next_obs, done)
    obs = next_obs.copy() # CRITICAL!!!
    # handle episode termination
    if done or (ep_len==max_ep_len):
        obs, ep_ret, ep_len = env.reset(), 0, 0
    # update
    if t >=update_after and not t%update_freq:
        for _ in range(update_freq):
            batch = buffer.sample_batch(batch_size)
            loss_actor = compute_actor_gradients(ac, actor_optimizer, batch)
            print("loss_actor: {}".format(loss_actor))
            loss_critic = compute_critic_gradients(ac, ac_targ, critic_optimizer, batch)
            print("loss_critic: {}".format(loss_critic))
            # update(ac, actor_optimizer, critic_optimizer, date=batch)
            # print()
    # handle end of epoch
    if not (t+1)%steps_per_epoch:
        epoch = (t+1)//steps_per_epoch
        # save model
        if not(epoch%save_freq) or (epoch==epochs):
            model_path = os.path.join(model_dir, env.spec.id, 'models', str(epoch))
            if not os.path.exists(os.path.dirname(model_path)):
                os.makedirs(os.path.dirname(model_path))
            tf.saved_model.save(ac, model_path)
# for ep in range(epochs):
#     for st in range(steps_per_epoch):
#         act, val, logp = ac.step(obs.reshape(1,-1))
#         next_obs, rew, done, _ = env.step(act.numpy())
#         ep_ret += rew
#         ep_len += 1
#         stepwise_rewards.append(rew)
#         total_steps += 1
#         buffer.store(obs, act.numpy(), rew, val.numpy(), logp.numpy())
#         obs = next_obs # SUPER CRITICAL!!!
#         # handle episode termination
#         timeout = (ep_len==env.spec.max_episode_steps)
#         terminal = done or timeout
#         epoch_ended = (st==steps_per_epoch-1)
#         if terminal or epoch_ended:
#             if epoch_ended and not(terminal):
#                 print('Warning: trajectory cut off by epoch at {} steps.'.format(ep_len), flush=True)
#             if timeout or epoch_ended:
#                 _, val, _ = ac.step(obs.reshape(1,-1))
#             else:
#                 val = 0
#             buffer.finish_path(val)
#             if terminal:
#                 episodes += 1
#                 episodic_returns.append(ep_ret)
#                 averaged_returns.append(sum(episodic_returns)/episodes)
#                 print("\nTotalSteps: {} \nEpisode: {}, Step: {}, EpReturn: {}, EpLength: {}".format(total_steps, episodes, st+1, ep_ret, ep_len))
#             obs, ep_ret, ep_len = env.reset(), 0, 0
#     # Save model
#     if not ep%save_freq or (ep==epochs-1):
#         model_path = os.path.join(model_dir, env.spec.id, 'models', str(ep))
#         if not os.path.exists(os.path.dirname(model_path)):
#             os.makedirs(os.path.dirname(model_path))
#         tf.saved_model.save(ac, model_path)
#
#     # update actor-critic
#     loss_pi, pi_info, loss_v = update(buffer)
#     print("\n================================================================\nEpoch: {} \nStep: {} \nAveReturn: {} \nLossPi: {} \nLossV: {} \nKLDivergence: {} \n Entropy: {} \nTimeElapsed: {}\n================================================================\n".format(ep+1, st+1, averaged_returns[-1], loss_pi, loss_v, pi_info['kl'], pi_info['ent'], time.time()-start_time))
# ################################################################
#
#
# # plot returns
# fig, ax = plt.subplots(figsize=(8, 6))
# fig.suptitle('Averaged Returns')
# ax.plot(averaged_returns)
# plt.show()
#
# # Test trained model
# input("Press ENTER to test lander...")
# num_episodes = 10
# num_steps = env.spec.max_episode_steps
# ep_rets, ave_rets = [], []
# for ep in range(num_episodes):
#     obs, done, rewards = env.reset(), False, []
#     for st in range(num_steps):
#         env.render()
#         act, _, _ = ac.step(obs.reshape(1,-1))
#         next_obs, rew, done, info = env.step(act.numpy())
#         rewards.append(rew)
#         # print("\n-\nepisode: {}, step: {} \naction: {} \nobs: {}, \nreward: {}".format(ep+1, st+1, act, obs, rew))
#         obs = next_obs.copy()
#         if done:
#             ep_rets.append(sum(rewards))
#             ave_rets.append(sum(ep_rets)/len(ep_rets))
#             print("\n---\nepisode: {} \nepisode return: {}, averaged return: {} \n---\n".format(ep+1, ep_rets[-1], ave_rets[-1]))
#             break
