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

class Critic(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, **kwargs):
        super(Critic, self).__init__(name='critic', **kwargs)
        inputs = tf.keras.Input(shape=(obs_dim,))
        x = tf.keras.layers.Dense(hidden_sizes[0], activation=activation)(inputs)
        for i in range(1, len(hidden_sizes)):
            x = tf.keras.layers.Dense(hidden_sizes[i], activation=activation)(x)
        outputs = tf.keras.layers.Dense(act_dim)(x)
        self.q_net = tf.keras.Model(inputs=inputs, outputs=outputs)
        
    def call(self, obs):
        return self.q_net(obs)
        # return tf.squeeze(qval, axis=-1)

class DeepQNet(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256,256), activation='relu', gamma = 0.99, alpha=0.,
                 lr=3e-4, polyak=0.995, **kwargs):
        super(DeepQNet, self).__init__(name='dqn', **kwargs)
        # params
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.alpha = alpha # entropy temperature
        self.gamma = gamma # discount rate
        self.polyak = polyak
        self.init_eps = 1.
        self.final_eps = .1
        # model
        self.q = Critic(obs_dim, act_dim, hidden_sizes, activation) 
        self.targ_q = Critic(obs_dim, act_dim, hidden_sizes, activation)
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)
        # variable
        self.epsilon = self.init_eps

    def linear_epsilon_decay(self, episode, decay_period, warmup_episodes):
        episodes_left = decay_period + warmup_episodes - episode
        bonus = (self.init_eps - self.final_eps) * episodes_left / decay_period
        bonus = np.clip(bonus, 0., self.init_eps-self.final_eps)
        self.epsilon = self.final_eps + bonus

    def act(self, obs):
        if np.random.rand() > self.epsilon:
            a = np.argmax(self.q(obs))
        else:
            a = np.random.randint(self.act_dim)
        return a

    def train_one_batch(self, data):
        # update critic
        with tf.GradientTape() as tape:
            tape.watch(self.q.trainable_weights)
            pred_qval = tf.math.reduce_sum(self.q(data['obs']) * tf.one_hot(data['act'], self.act_dim), axis=-1)
            targ_qval = data['rew'] + self.gamma*(1 - data['done'])*tf.math.reduce_sum(self.targ_q(data['nobs'])*tf.one_hot(tf.math.argmax(self.q(data['nobs']),axis=1), self.act_dim),axis=1) # double DQN trick
            loss_q = tf.keras.losses.MSE(y_true=targ_qval, y_pred=pred_qval)
        grads = tape.gradient(loss_q, self.q.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.q.trainable_weights))
        # Polyak average update target Q-nets
        q_weights_update = []
        for w_q, w_targ_q in zip(self.q.get_weights(), self.targ_q.get_weights()):
            w_q_upd = self.polyak*w_targ_q
            w_q_upd = w_q_upd + (1 - self.polyak)*w_q
            q_weights_update.append(w_q_upd)
        self.targ_q.set_weights(q_weights_update)

        return loss_q

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.nobs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.int32)
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


RANDOM_SEED = 0
if __name__=='__main__':
    env = gym.make('LunarLander-v2')
    max_episode_steps = env.spec.max_episode_steps
    # set seed
    tf.random.set_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)
    batch_size = 100
    update_freq = 50
    update_after = 1000
    decay_period = 500
    warmup_episodes = 50
    dqn = DeepQNet(obs_dim=8, act_dim=4)
    replay_buffer = ReplayBuffer(obs_dim=8, size=int(1e5)) 
    total_steps = int(1e6)
    episodic_returns = []
    sedimentary_returns = []
    episodic_steps = []
    save_freq = 100
    episode_counter = 0
    model_dir = './models/dqn_small_buffer/'+env.spec.id
    start_time = time.time()
    obs, done, ep_ret, ep_len = env.reset(), False, 0, 0
    for t in range(total_steps):
        # env.render()
        act = np.squeeze(dqn.act(obs.reshape(1,-1)))
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
            print("\n====\nEpisode: {} \nEpisodeLength: {} \nTotalSteps: {} \nEpsilon: {} \nEpisodeReturn: {} \nSedimentaryReturn: {} \nTimeElapsed: {} \n====\n".format(episode_counter, ep_len, t+1, dqn.epsilon, ep_ret, sedimentary_returns[-1], time.time()-start_time))
            # save model
            if not episode_counter%save_freq:
                model_path = os.path.join(model_dir, str(episode_counter))
                if not os.path.exists(os.path.dirname(model_path)):
                    os.makedirs(os.path.dirname(model_path))
                dqn.q.q_net.save(model_path)
            # reset env
            obs, done, ep_ret, ep_len = env.reset(), False, 0, 0
            dqn.linear_epsilon_decay(episode=episode_counter, decay_period=decay_period, warmup_episodes=warmup_episodes)
        if not t%update_freq and t>=update_after:
            for _ in range(update_freq):
                minibatch = replay_buffer.sample_batch(batch_size=batch_size)
                loss_q = dqn.train_one_batch(data=minibatch)
                logging.debug("\nloss_q: {}".format(loss_q))

    # Save returns 
    np.save(os.path.join(model_dir, 'episodic_returns.npy'), episodic_returns)
    np.save(os.path.join(model_dir, 'sedimentary_returns.npy'), sedimentary_returns)
    np.save(os.path.join(model_dir, 'episodic_steps.npy'), episodic_steps)
    with open(os.path.join(model_dir, 'training_time.txt'), 'w') as f:
        f.write("{}".format(time.time()-start_time))
    # Save final model
    model_path = os.path.join(model_dir, str(episode_counter))
    dqn.q.q_net.save(model_path)
    # plot returns
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Averaged Returns')
    ax.plot(sedimentary_returns)
    plt.show()

# Test
input("Press ENTER to test lander...")
dqn.epsilon = 0.
for ep in range(10):
    o, d, ep_ret = env.reset(), False, 0
    for st in range(max_episode_steps):
        env.render()
        a = np.squeeze(dqn.act(o.reshape(1,-1)))
        o2,r,d,_ = env.step(a)
        ep_ret += r
        o = o2
        if d:
            print("EpReturn: {}".format(ep_ret))
            break 
