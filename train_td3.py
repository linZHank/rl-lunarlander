import sys
import os
from copy import deepcopy
import numpy as np
import random
import time
import gym
import matplotlib.pyplot as plt
from datetime import datetime
import logging

import tensorflow as tf
print(tf.__version__)
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
################################################################


################################################################
"""
Build actor_net, critic_net
"""
# def mlp(sizes, activation, output_activation=None):
#     inputs = tf.keras.Input(shape=(sizes[0],))
#     x = tf.keras.layers.Dense(sizes[1], activation=activation)(inputs)
#     for i in range(2,len(sizes)-1):
#         x = tf.keras.layers.Dense(sizes[i], activation=activation)(x)
#     outputs = tf.keras.layers.Dense(sizes[-1], activation=output_activation)(x)
#
#     return tf.keras.Model(inputs=inputs, outputs=outputs)
#
# class MLPActor(tf.Module):
#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
#         super().__init__()
#         pi_sizes = [obs_dim] +list(hidden_sizes) + [act_dim]
#         assert len(pi_sizes)>=3
#         self.pi = mlp(sizes=pi_sizes, activation=activation, output_activation='tanh')
#         # self.pi_targ = tf.keras.models.clone_model(self.pi)
#         self.act_limit = act_limit
#
#     @tf.Module.with_name_scope
#     def __call__(self, obs):
#         return self.act_limit * tf.squeeze(self.pi(obs))
#
# class MLPQFunction(tf.Module):
#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
#         super().__init__()
#         q_sizes = [obs_dim+act_dim] +list(hidden_sizes) + [1]
#         self.q = mlp(sizes=q_sizes, activation=activation)
#
#     @tf.Module.with_name_scope
#     def __call__(self, obs, act):
#         q = self.q(tf.concat([obs, act], axis=-1))
#         return tf.squeeze(q) # Critical to ensure q has right shape.
#
# class MLPActorCritic(tf.Module):
#     def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=(256,256),
#                  activation='relu'):
#         super().__init__()
#         # obs_dim = observation_space.shape[0]
#         # act_dim = action_space.shape[0]
#         # act_limit = action_space.high[0]
#         # build policy and value functions
#         self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
#         self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
#         self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
#
#     def act(self, obs):
#         a = tf.stop_gradient(self.pi(obs).numpy())
#         return np.squeeze(a)
actor_net = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=env.observation_space.shape),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(env.action_space.shape[0], activation='tanh')
    ]
)
actor_net_targ = tf.keras.models.clone_model(actor_net)

critic_net_1 = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(env.observation_space.shape[0]+env.action_space.shape[0])),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1)
    ]
)
critic_net_1_targ = tf.keras.models.clone_model(critic_net_1)

critic_net_2 = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(env.observation_space.shape[0]+env.action_space.shape[0])),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1)
    ]
)
critic_net_2_targ = tf.keras.models.clone_model(critic_net_2)

################################################################


################################################################
"""
Build replay_buffer
"""
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
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
Compute loss and gradients
"""
# def compute_q_loss(act_limit, target_noise, noise_clip, batch):
#     obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = batch['obs'], batch['act'], batch['rew'], batch['next_obs'], batch['done']
#     q1 = critic_net_1(tf.concat([obs_batch, act_batch], axis=1))
#     q2 = critic_net_2(tf.concat([obs_batch, act_batch], axis=1))
#     pi_targ = tf.stop_gradient(tf.math.multiply(act_limit, actor_net_targ(obs_batch)))
#     # print("pi_targ: {}".format(pi_targ))
#     # target policy smoothing
#     epsilon = tf.stop_gradient(tf.random.normal(pi_targ.shape)*target_noise)
#     epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
#     next_act = pi_targ + epsilon
#     next_act = tf.clip_by_value(next_act, -act_limit, act_limit)
#     # targe Q-values
#     next_q1 = tf.stop_gradient(critic_net_1_targ(tf.concat([next_obs_batch, next_act], axis=1)))
#     next_q2 = tf.stop_gradient(critic_net_2_targ(tf.concat([next_obs_batch, next_act], axis=1)))
#     pessimistic_next_q = tf.math.minimum(next_q1, next_q2)
#     q_targ = rew_batch + gamma*(1 - done_batch)*pessimistic_next_q
#     # MSE loss
#     loss_q1 = tf.keras.losses.MSE(q_targ, q1)
#     loss_q2 = tf.keras.losses.MSE(q_targ, q2)
#     loss_Q = loss_q1 + loss_q2
#
#     return loss_Q

# def compute_q1_grads(act_limit, target_noise, noise_clip, batch):
#     with tf.GradientTape() as tape:
#         loss_Q = compute_q_loss(act_limit, target_noise, noise_clip, batch)
#     grads_q1 = tape.gradient(loss_Q, critic_net_1.trainable_variables)
#     q1_optimizer.apply_gradients(zip(grads_q1, critic_net_1.trainable_variables))
#
# def compute_q2_grads(act_limit, target_noise, noise_clip, batch):
#     with tf.GradientTape() as tape:
#         loss_Q = compute_q_loss(act_limit, target_noise, noise_clip, batch)
#     grads_q2 = tape.gradient(loss_Q, critic_net_2.trainable_variables)
#     q2_optimizer.apply_gradients(zip(grads_q2, critic_net_2.trainable_variables))

def compute_q_grads(act_limit, target_noise, noise_clip, batch):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        # specify gradients of interested variables
        tape.watch(critic_net_1.trainable_variables+critic_net_2.trainable_variables)
        # read in data from sampled batch
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = batch['obs'], batch['act'], batch['rew'], batch['next_obs'], batch['done']
        q1 = critic_net_1(tf.concat([obs_batch, act_batch], axis=1))
        q2 = critic_net_2(tf.concat([obs_batch, act_batch], axis=1))
        # compute target q's without recording gradients
        with tape.stop_recording():
            pi_targ = tf.math.multiply(act_limit, actor_net_targ(obs_batch))
            # print("pi_targ: {}".format(pi_targ))
            # target policy smoothing
            epsilon = tf.random.normal(pi_targ.shape)*target_noise
            epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
            next_act = pi_targ + epsilon
            next_act = tf.clip_by_value(next_act, -act_limit, act_limit)
            # targe Q-values
            next_q1 = critic_net_1_targ(tf.concat([next_obs_batch, next_act], axis=1))
            next_q2 = critic_net_2_targ(tf.concat([next_obs_batch, next_act], axis=1))
            pessimistic_next_q = tf.math.minimum(next_q1, next_q2)
            q_targ = rew_batch + gamma*(1 - done_batch)*pessimistic_next_q
        # MSE loss
        loss_q1 = tf.keras.losses.MSE(q_targ, q1)
        loss_q2 = tf.keras.losses.MSE(q_targ, q2)
        loss_Q = loss_q1 + loss_q2
        # loss_Q = compute_q_loss(act_limit, target_noise, noise_clip, batch)
    grads_q = tape.gradient(loss_Q, critic_net_1.trainable_variables+critic_net_2.trainable_variables)
    q_optimizer.apply_gradients(zip(grads_q, critic_net_1.trainable_variables+critic_net_2.trainable_variables))

# def compute_pi_loss(act_limit, batch):
#     obs_batch = batch['obs']
#     pi = tf.math.multiply(act_limit, actor_net(obs_batch))
#     objective_pi = critic_net_1(tf.concat([obs_batch, pi], axis=1))
#     loss_pi = -tf.math.reduce_mean(objective_pi)
#
#     return loss_pi

def compute_pi_grads(act_limit, batch):
    with tf.GradientTape() as tape:
        tape.watch(actor_net.trainable_variables)
        obs_batch = batch['obs']
        pi = tf.math.multiply(act_limit, actor_net(obs_batch))
        objective_pi = critic_net_1(tf.concat([obs_batch, pi], axis=1))
        loss_pi = -tf.math.reduce_mean(objective_pi)
        # loss_pi = compute_pi_loss(act_limit, batch)
    grads_pi = tape.gradient(loss_pi, actor_net.trainable_variables)
    actor_optimizer.apply_gradients(zip(grads_pi, actor_net.trainable_variables))

# # @tf.function
# def compute_loss_critic(critic_net, critic_net_stable, actor_net_stable, batch_reps):
#     # compute predict Q
#     tensor_obs = tf.convert_to_tensor(batch_reps[0])
#     # print("tensor_obs: {}".format(tensor_obs))
#     tensor_acts = tf.cast(tf.convert_to_tensor(batch_reps[1]), tf.float32)
#     # print("tensor_acts: {}".format(tensor_acts))
#     q_preds = critic_net(tf.concat([tensor_obs, tensor_acts], axis=1))
#     # print("q_preds: {}".format(q_preds))
#     # compute target Q
#     tensor_rews = tf.convert_to_tensor(batch_reps[2])
#     # print("tensor_rews: {}".format(tensor_rews))
#     tensor_dones = tf.cast(tf.convert_to_tensor(batch_reps[3]), tf.float32)
#     tensor_next_obs = tf.convert_to_tensor(batch_reps[4])
#     q_next = tf.stop_gradient(critic_net_stable(tf.concat([tensor_next_obs, actor_net_stable(tensor_next_obs)], axis=1)))
#     # print("q_next: {}".format(q_next))
#     q_targs = tf.reshape(tensor_rews, shape=[-1,1]) + gamma*(1.-tf.reshape(tensor_dones, shape=[-1,1]))*q_next
#     # print("q_targs: {}".format(q_targs))
#     # loss_Q = tf.math.reduce_mean((q_preds - q_targs)**2)
#     loss_Q = loss_object(q_targs, q_preds)
#     print("loss_Q: {}".format(loss_Q))
#
#     return loss_Q
#
# def compute_grads_critic(critic_net, critic_net_stable, actor_net_stable, batch_reps, optimizer_critic):
#     with tf.GradientTape() as tape:
#         loss_Q = compute_loss_critic(critic_net, critic_net_stable, actor_net_stable, batch_reps)
#     grads_critic = tape.gradient(loss_Q, critic_net.trainable_variables)
#     optimizer_critic.apply_gradients(zip(grads_critic, critic_net.trainable_variables))
#
# def compute_loss_actor(actor_net, critic_net_stable, batch_reps):
#     tensor_obs = tf.convert_to_tensor(batch_reps[0])
#     q_mu = critic_net_stable(tf.concat([tensor_obs, actor_net(tensor_obs)], axis=1))
#     # print("q_mu: {}".format(q_mu))
#     loss_mu = -tf.math.reduce_mean(q_mu)
#     print("loss_mu: {}".format(loss_mu))
#
#     return loss_mu
#
# def compute_grads_actor(actor_net, critic_net_stable, batch_reps, optimizer_critic):
#     with tf.GradientTape() as tape:
#         loss_mu = compute_loss_actor(actor_net, critic_net_stable, batch_reps)
#     grads_actor = tape.gradient(loss_mu, actor_net.trainable_variables)
#     optimizer_actor.apply_gradients(zip(grads_actor, actor_net.trainable_variables))
#
# # func store expeirience: (state, reward, done, next_state)
# def store(replay_buffer, experience, buffer_size):
#     if len(replay_buffer) >= buffer_size:
#         replay_buffer.pop(0)
#     replay_buffer.append(experience)
#     logging.debug("experience: {} stored in replay buffer".format(experience))
#
# # func sample batch experience
# def sample_batch(replay_buffer, batch_size):
#     if len(replay_buffer) < batch_size:
#         minibatch = replay_buffer
#     else:
#         minibatch = random.sample(replay_buffer, batch_size)
#     batch_reps = list(zip(*minibatch))
#
#     return batch_reps
################################################################


################################################################
"""
Main
"""
# params
steps_per_epoch=4000
epochs=100
replay_size=int(1e6)
gamma=0.99
polyak=0.995
pi_lr=3e-4
q_lr=1e-3
batch_size=100
warmup_steps=10000
update_after=1000
update_freq=50
act_noise=0.1
target_noise=0.2
noise_clip=0.5
policy_delay=2
num_test_episodes=10
# max_ep_len=1000
# instantiate env, actor-critic net and replay_buffer
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high
replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
# Create Optimizer
# q1_optimizer = tf.keras.optimizers.Adam(learning_rate=q_lr)
# q2_optimizer = tf.keras.optimizers.Adam(learning_rate=q_lr)
q_optimizer = tf.keras.optimizers.Adam(learning_rate=q_lr)
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=pi_lr)
model_dir = './training_models/ddpg/'+datetime.now().strftime("%Y-%m-%d-%H-%M")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
episodic_returns, episodic_lengths = [], []
total_steps = steps_per_epoch * epochs
start_time = time.time()
obs, ep_ret, ep_len = env.reset(), 0, 0
episode = 0
# start
for t in range(total_steps):
    # take random actions until warmup steps elapsed; add noise to policy generated action after that
    if t > warmup_steps:
        a = tf.stop_gradient(tf.squeeze(actor_net(obs.reshape(1,-1))))
        a += act_noise * np.random.randn(act_dim)
        act = np.clip(a, -act_limit, act_limit)
    else:
        act = env.action_space.sample()
    # step
    next_obs, rew, done, _ = env.step(act)
    ep_ret += rew
    ep_len += 1
    # ignore artificial done due to max steps reached
    done = False if ep_len==env.spec.max_episode_steps else done
    replay_buffer.store(obs, act, rew, next_obs, done)
    # NEXT LINE IS SUPER CRITICAL!!!: update current state
    obs = next_obs.copy()
    # end of trajectory handling
    if done or (ep_len>=env.spec.max_episode_steps):
        episode += 1
        episodic_returns.append(ep_ret)
        episodic_lengths.append(ep_len)
        logging.info(
            "\n================================================================\nEpisode: {}, EpisodeLength: {} \nTotalSteps: {} \nEpReturns: {} \nAveEpReturn: {} \nTime: {} \n================================================================\n".format(
                episode,
                ep_len,
                t+1,
                ep_ret,
                ep_ret/ep_len,
                time.time()-start_time
            )
        )
        obs, ep_ret, ep_len = env.reset(), 0, 0
    # let's update neural nets
    if t>=update_after and not t%update_freq:
        for i in range(update_freq):
            batch = replay_buffer.sample_batch(batch_size)
            # loss_Q = compute_q_loss(act_limit, target_noise, noise_clip, batch)
            # print("loss_Q: {}".format(loss_Q))
            # loss_pi = compute_pi_loss(act_limit, batch)
            # print("loss_pi: {}".format(loss_pi))
            # compute_q1_grads(act_limit, target_noise, noise_clip, batch)
            # compute_q2_grads(act_limit, target_noise, noise_clip, batch)
            compute_q_grads(act_limit, target_noise, noise_clip, batch)
            if not i%policy_delay:
                compute_pi_grads(act_limit, batch)
                # polyak averaging
                weights_critic1_update = []
                for w_q1, w_q1_targ in zip(critic_net_1.get_weights(), critic_net_1_targ.get_weights()):
                    w_q1_upd = polyak*w_q1_targ
                    w_q1_upd = w_q1_upd + (1 - polyak)*w_q1
                    weights_critic1_update.append(w_q1_upd)
                critic_net_1_targ.set_weights(weights_critic1_update)
                weights_critic2_update = []
                for w_q2, w_q2_targ in zip(critic_net_2.get_weights(), critic_net_2_targ.get_weights()):
                    w_q2_upd = polyak*w_q2_targ
                    w_q2_upd = w_q2_upd + (1 - polyak)*w_q2
                    weights_critic2_update.append(w_q2_upd)
                critic_net_2_targ.set_weights(weights_critic2_update)
                weights_actor_update = []
                for w_pi, w_pi_targ in zip(actor_net.get_weights(), actor_net_targ.get_weights()):
                    w_pi_upd = polyak*w_pi_targ
                    w_pi_upd = w_pi_upd + (1 - polyak)*w_pi
                    weights_actor_update.append(w_pi_upd)
                actor_net_targ.set_weights(weights_actor_update)
#     # end of epoch handling
#     if not (t+1)%steps_per_epoch:
#         epoch = (t+1)//steps_per_epoch
#         # save model
#         pass
#
# Test trained model
input("Press ENTER to test lander...")
num_episodes = 10
num_steps = env.spec.max_episode_steps
ep_rets, ave_rets = [], []
for ep in range(num_episodes):
    obs, done, rewards = env.reset(), False, []
    for st in range(num_steps):
        env.render()
        a = tf.stop_gradient(tf.squeeze(actor_net(obs.reshape(1,-1))))
        a += act_noise * np.random.randn(act_dim)
        act = np.clip(a, -act_limit, act_limit)
        next_obs, rew, done, info = env.step(act)
        rewards.append(rew)
        print("\n-\nepisode: {}, step: {} \naction: {} \nobs: {}, \nreward: {}".format(ep+1, st+1, act, obs, rew))
        obs = next_obs.copy()
        if done:
            ep_rets.append(sum(rewards))
            ave_rets.append(sum(ep_rets)/len(ep_rets))
            print("\n---\nepisode: {} \nepisode return: {}, averaged return: {} \n---\n".format(ep+1, ep_rets[-1], ave_rets[-1]))
            break





# if __name__ == '__main__':
#         obs, done, rewards = env.reset(), False, []
#         for st in range(num_steps):
#             # env.render()
#             # exploit vs explore
#             if ep >= warmup:
#                 act = tf.stop_gradient(tf.squeeze(actor_net(obs.reshape(1,-1)))).numpy() + act_noise*np.random.randn(env.action_space.shape[0])
#             else:
#                 act = env.action_space.sample()
#             next_obs, rew, done, info = env.step(act)
#             step_counter += 1
#             rewards.append(rew)
#             store(replay_buffer=replay_buffer, experience=(obs.copy(), act, np.float32(rew), done, next_obs.copy()), buffer_size=buffer_size)
#             batch_reps = sample_batch(replay_buffer, batch_size)
#             if ep >= warmup:
#                 # train_one_batch
#                 # loss_Q_old = compute_loss_critic(critic_net, critic_net_stable, actor_net_stable, batch_reps)
#                 # loss_mu_old = compute_loss_actor(actor_net, critic_net_stable, batch_reps)
#                 # print("loss_Q: {}, loss_mu: {}".format(loss_Q, loss_mu))
#                 compute_grads_critic(critic_net, critic_net_stable, actor_net_stable, batch_reps, optimizer_critic)
#                 compute_grads_actor(actor_net, critic_net_stable, batch_reps, optimizer_actor)
#                 # loss_Q = compute_loss_critic(critic_net, critic_net_stable, actor_net_stable, batch_reps)
#                 # loss_mu = compute_loss_actor(actor_net, critic_net_stable, batch_reps)
#                 # polyak averaging
#                 weights_critic_update = []
#                 for w_q, w_q_targ in zip(critic_net.get_weights(), critic_net_stable.get_weights()):
#                     w_q_upd = polyak*w_q_targ
#                     w_q_upd = w_q_upd + (1 - polyak)*w_q
#                     weights_critic_update.append(w_q_upd)
#                 critic_net_stable.set_weights(weights_critic_update)
#                 weights_actor_update = []
#                 for w_mu, w_mu_targ in zip(actor_net.get_weights(), actor_net_stable.get_weights()):
#                     w_mu_upd = polyak*w_mu_targ
#                     w_mu_upd = w_mu_upd + (1 - polyak)*w_mu
#                     weights_actor_update.append(w_mu_upd)
#                 actor_net_stable.set_weights(weights_actor_update)
#             logging.debug("\n-\nepisode: {}, step: {} \naction: {} \nobs: {}, \nreward: {}".format(ep+1, st+1, act, obs, rew))
#             obs = next_obs.copy()
#             if done:
#                 ep_rets.append(sum(rewards))
#                 ave_rets.append(sum(ep_rets)/len(ep_rets))
#                 logging.info(
#                     "\n================================================================\nEpisode: {}, Step: {} \nEpReturns: {} \nAveEpReturn: {} \nTime: {} \n================================================================\n".format(
#                         ep+1,
#                         st+1,
#                         ep_rets[-1],
#                         ave_rets[-1],
#                         time.time()-start_time
#                     )
#                 )
#                 break
#
# # Save final ckpt
# # save_path = ckpt_manager.save()
# # save model
# actor_net.save(os.path.join(model_dir, str(step_counter)))
# critic_net.save(os.path.join(model_dir, str(step_counter)))
#
# # plot returns
# fig, ax = plt.subplots(figsize=(8, 6))
# fig.suptitle('Averaged Returns')
# ax.plot(ave_rets)
# plt.show()
