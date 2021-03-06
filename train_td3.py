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
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
# set log level
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)
################################################################


################################################################
class Critic(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, **kwargs):
        super(Critic, self).__init__(name='critic', **kwargs)
        inputs = tf.keras.Input(shape=(obs_dim+act_dim,))
        x = tf.keras.layers.Dense(hidden_sizes[0], activation=activation)(inputs)
        for i in range(1, len(hidden_sizes)):
            x = tf.keras.layers.Dense(hidden_sizes[i], activation=activation)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        self.q_net = tf.keras.Model(inputs=inputs, outputs=outputs)
        
    @tf.function
    def call(self, obs, act):
        qval = self.q_net(tf.concat([obs, act], axis=-1))
        return tf.squeeze(qval, axis=-1)
    
class Actor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, **kwargs):
        super(Actor, self).__init__(name='actor', **kwargs)
        inputs = tf.keras.Input(shape=(obs_dim,))
        x = tf.keras.layers.Dense(hidden_sizes[0], activation=activation)(inputs)
        for i in range(1, len(hidden_sizes)):
            x = tf.keras.layers.Dense(hidden_sizes[i], activation=activation)(x)
        outputs = tf.keras.layers.Dense(act_dim)(x) 
        self.policy_net = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.act_limit = act_limit

    def call(self, obs):
        return self.act_limit*self.policy_net(obs)
        
class TwinDelayedDDPG(tf.Module):
    def __init__(self, obs_dim, act_dim, act_limit=1, hidden_sizes=(256,256), activation='relu', gamma = 0.99,
                 critic_lr=1e-4, actor_lr=1e-4, noise_clip=0.5, target_noise=0.2, policy_delay=2, polyak=0.995,
                 act_noise=0.1, **kwargs):
        super(TwinDelayedDDPG, self).__init__(name='td3', **kwargs)
        # params
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.gamma = gamma # discount rate
        self.polyak = polyak
        self.act_noise = act_noise
        # networks
        self.pi = Actor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.targ_pi = Actor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q0 = Critic(obs_dim, act_dim, hidden_sizes, activation) 
        self.q1 = Critic(obs_dim, act_dim, hidden_sizes, activation) 
        self.targ_q0 = Critic(obs_dim, act_dim, hidden_sizes, activation)
        self.targ_q1 = Critic(obs_dim, act_dim, hidden_sizes, activation)
        self.critic_optimizer = tf.keras.optimizers.Adam(lr=critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(lr=actor_lr)

    def act(self, obs):
        a = self.pi(obs).numpy()
        a += self.act_noise*np.random.randn(self.act_dim)
        
        return np.clip(a, -self.act_limit, self.act_limit)

    def train_one_batch(self, data, timer):
        # update critic
        with tf.GradientTape() as tape:
            tape.watch(self.q0.trainable_weights + self.q1.trainable_weights)
            pred_qval0 = self.q0(data['obs'], data['act'])
            pred_qval1 = self.q1(data['obs'], data['act'])
            targ_a = self.targ_pi(data['nobs'])
            # next 4 lines implement target smoothing
            eps = tf.random.normal(shape=targ_a.shape)*self.target_noise
            eps = tf.clip_by_value(eps, -self.noise_clip, self.noise_clip)
            na = targ_a + eps
            na = tf.clip_by_value(na, -self.act_limit, self.act_limit)
            nqval0 = self.targ_q0(data['nobs'], na)
            nqval1 = self.targ_q1(data['nobs'], na)
            pessi_nqval = tf.math.minimum(nqval0, nqval1) # pessimistic value
            targ_qval = data['rew'] + self.gamma*(1-data['done'])*pessi_nqval
            loss_q0 = tf.keras.losses.MSE(y_true=targ_qval, y_pred=pred_qval0)
            loss_q1 = tf.keras.losses.MSE(y_true=targ_qval, y_pred=pred_qval1)
            loss_q = loss_q0 + loss_q1
        grads_critic = tape.gradient(loss_q, self.q0.trainable_weights + self.q1.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(grads_critic, self.q0.trainable_weights + self.q1.trainable_weights))
        # update actor
        if not timer%self.policy_delay:
            with tf.GradientTape() as tape:
                tape.watch(self.pi.trainable_weights)
                loss_pi = -tf.math.reduce_mean(self.q0(data['obs'], self.pi(data['obs'])))
            grads_actor = tape.gradient(loss_pi, self.pi.trainable_weights)
            self.actor_optimizer.apply_gradients(zip(grads_actor, self.pi.trainable_weights))
            # Polyak average update target Q-nets
            q0_weights_update = []
            for w_q0, w_targ_q0 in zip(self.q0.get_weights(), self.targ_q0.get_weights()):
                w_q0_upd = self.polyak*w_targ_q0
                w_q0_upd = w_q0_upd + (1 - self.polyak)*w_q0
                q0_weights_update.append(w_q0_upd)
            self.targ_q0.set_weights(q0_weights_update)
            q1_weights_update = []
            for w_q1, w_targ_q1 in zip(self.q1.get_weights(), self.targ_q1.get_weights()):
                w_q1_upd = self.polyak*w_targ_q1
                w_q1_upd = w_q1_upd + (1 - self.polyak)*w_q1
                q1_weights_update.append(w_q1_upd)
            self.targ_q1.set_weights(q1_weights_update)
            # Polyak average update target policy-nets
            pi_weights_update = []
            for w_pi, w_targ_pi in zip(self.pi.get_weights(), self.targ_pi.get_weights()):
                w_pi_upd = self.polyak*w_targ_pi
                w_pi_upd = w_pi_upd + (1 - self.polyak)*w_pi
                pi_weights_update.append(w_pi_upd)
            self.targ_pi.set_weights(pi_weights_update)
        else:
            loss_pi = 0.

        return loss_q, loss_pi
################################################################


################################################################
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.nobs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, done, nobs):
        self.obs_buf[self.ptr] = obs
        self.nobs_buf[self.ptr] = nobs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=tf.convert_to_tensor(self.obs_buf[idxs]),
                     nobs=tf.convert_to_tensor(self.nobs_buf[idxs]),
                     act=tf.convert_to_tensor(self.act_buf[idxs]),
                     rew=tf.convert_to_tensor(self.rew_buf[idxs]),
                     done=tf.convert_to_tensor(self.done_buf[idxs]))
        return batch
################################################################


################################################################
RANDOM_SEED = 0
if __name__=='__main__':
    env = gym.make('LunarLanderContinuous-v2')
    td3 = TwinDelayedDDPG(obs_dim=8, act_dim=2)
    # set seed
    tf.random.set_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)
    # params
    max_episode_steps = env.spec.max_episode_steps
    batch_size = 100
    update_freq = 50
    update_after = 1000
    warmup_steps = 5000
    replay_buffer = ReplayBuffer(obs_dim=8, act_dim=2, size=int(1e6)) 
    total_steps = int(1e6)
    episodic_returns = []
    sedimentary_returns = []
    episodic_steps = []
    save_freq = 100
    episode_counter = 0
    model_dir = './models/td3/'+env.spec.id
    obs, done, ep_ret, ep_len = env.reset(), False, 0, 0
    for t in range(total_steps):
        # env.render()
        if t < warmup_steps:
            act = env.action_space.sample()
        else:
            act = np.squeeze(td3.act(obs.reshape(1,-1)))
        nobs, rew, done, _ = env.step(act)
        ep_ret += rew
        ep_len += 1
        done = False if ep_len == max_episode_steps else done
        replay_buffer.store(obs, act, rew, done, nobs)
        obs = nobs
        if done or (ep_len==max_episode_steps):
            episode_counter += 1
            episodic_returns.append(ep_ret)
            sedimentary_returns.append(sum(episodic_returns)/episode_counter)
            episodic_steps.append(t+1)
            print("\n====\nEpisode: {} \nEpisodeLength: {} \nTotalSteps: {} \nEpisodeReturn: {} \nSedimentaryReturn: {}\n====\n".format(episode_counter, ep_len, t+1, ep_ret, sedimentary_returns[-1]))
            # save model
            if not episode_counter%save_freq:
                model_path = os.path.join(model_dir, str(episode_counter))
                if not os.path.exists(os.path.dirname(model_path)):
                    os.makedirs(os.path.dirname(model_path))
                td3.pi.policy_net.save(model_path)
            # reset env
            obs, done, ep_ret, ep_len = env.reset(), False, 0, 0
        if not t%update_freq and t>=update_after:
            for uf in range(update_freq):
                minibatch = replay_buffer.sample_batch(batch_size=batch_size)
                loss_q, loss_pi = td3.train_one_batch(data=minibatch, timer=uf)
                logging.debug("\nloss_q: {} \nloss_pi: {}".format(loss_q, loss_pi))

    # Save returns 
    np.save(os.path.join(model_dir, 'episodic_returns.npy'), episodic_returns)
    np.save(os.path.join(model_dir, 'sedimentary_returns.npy'), sedimentary_returns)
    np.save(os.path.join(model_dir, 'episodic_steps.npy'), episodic_steps)
    # Save final model
    model_path = os.path.join(model_dir, str(episode_counter))
    td3.pi.policy_net.save(model_path)
    # plot returns
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Averaged Returns')
    ax.plot(sedimentary_returns)
    plt.show()

# Test
input("Press ENTER to test lander...")
for ep in range(10):
    o, d, ep_ret = env.reset(), False, 0
    for st in range(max_episode_steps):
        env.render()
        a = np.squeeze(td3.act(o.reshape(1,-1)))
        o2,r,d,_ = env.step(a)
        ep_ret += r
        o = o2
        if d:
            print("EpReturn: {}".format(ep_ret))
            break 
