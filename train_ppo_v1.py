"""
Train a PPO agent for LL using agent.ppo
"""
import sys
import os
import gym
import numpy as np
import scipy.signal
import tensorflow as tf
from agents.ppo import PPOAgent
import logging
# set log level
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)
import pdb

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPOBuffer:
    def __init__(self, size, gamma=.99, lam=.95, batch_size=128):
        self.obs_buf = []
        self.act_buf = []
        self.logp_buf = []
        self.rew_buf = []
        self.ret_buf = []
        self.adv_buf = []
        self.val_buf = []
        self.gamma = gamma
        self.lam = lam
        self.actor_data = dict()
        self.critic_data = dict()
        self.ptr, self.episode_start_idx, self.max_size = 0, 0, size
        self.batch_size = batch_size

    def store(self, obs, act, logp, rew, val):
        assert self.ptr <= self.max_size
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.logp_buf.append(logp)
        self.rew_buf.append(rew)
        self.val_buf.append(val)
        self.ptr += 1

    def finish_episode(self, last_val=0):
        ep_slice = slice(self.episode_start_idx, self.ptr)
        rews = np.array(self.rew_buf[ep_slice])
        vals = np.append(np.array(self.val_buf[ep_slice]), last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews + self.gamma*vals[1:] - vals[:-1]
        self.adv_buf += list(discount_cumsum(deltas, self.gamma*self.lam))
        # next line implement reward-to-go
        self.ret_buf += list(discount_cumsum(np.append(rews, last_val), self.gamma)[:-1])
        self.episode_start_idx = self.ptr

    def get(self):
        """
        Get a data dicts from replay buffer
        """
        # convert list to array
        obs_buf = np.array(self.obs_buf)
        act_buf = np.array(self.act_buf) 
        logp_buf = np.array(self.logp_buf)
        rew_buf = np.array(self.rew_buf) 
        ret_buf = np.array(self.ret_buf) 
        adv_buf = np.array(self.adv_buf) 
        # next three lines implement advantage normalization
        adv_mean = np.mean(adv_buf)
        adv_std = np.std(adv_buf)
        adv_buf = (adv_buf - adv_mean) / adv_std
        # create data dict for training actor
        actor_data = dict(
            obs = tf.convert_to_tensor(obs_buf, dtype=tf.float32),
            act = tf.convert_to_tensor(act_buf, dtype=tf.float32),
            logp = tf.convert_to_tensor(logp_buf, dtype=tf.float32),
            adv = tf.convert_to_tensor(adv_buf, dtype=tf.float32)
        )
        self.actor_data = actor_data
        actor_dataset = tf.data.Dataset.from_tensor_slices(actor_data)
        batched_actor_dataset = actor_dataset.shuffle(1024).batch(self.batch_size)
        # create data dict for training critic
        critic_data = dict(
            obs = tf.convert_to_tensor(obs_buf, dtype=tf.float32),
            ret = tf.convert_to_tensor(ret_buf, dtype=tf.float32)
        )
        self.critic_data = critic_data
        critic_dataset = tf.data.Dataset.from_tensor_slices(critic_data)
        batched_critic_dataset = critic_dataset.shuffle(1024).batch(self.batch_size)

        return batched_actor_dataset, batched_critic_dataset

if __name__=='__main__':
    env = gym.make('LunarLanderContinuous-v2')
    agent = PPOAgent()
    # parameters
    num_episodes = 11
    num_steps = env.spec.max_episode_steps
    buffer_size = int(1e4)
    update_every = 10 # perform training every update_every episodes
    # variables
    step_counter = 0
    episode_counter = 0
    episodic_returns = []
    sedimentary_returns = []
    buf = PPOBuffer(size=buffer_size)
    for ep in range(num_episodes):
        obs, ep_rew = env.reset(), 0
        for st in range(num_steps):
            env.render()
            act, val, logp = agent.pi_given_state(np.expand_dims(obs, axis=0))
            n_obs, rew, done, info = env.step(act)
            logging.debug("\nepisodes: {}, step: {} \nobs: {} \nact: {} \nrew: {} \ndone: {} \ninfo: {}".format(
                ep, 
                st, 
                obs,
                act, 
                rew, 
                done,
                info
            ))
            ep_rew += rew
            step_counter += 1
            buf.store(obs, act, rew, val, logp)
            obs = n_obs.copy()
            if done or (st==num_steps-1):
                episode_counter += 1
                if st==num_steps - 1:
                    _, val, _ = agent.pi_given_state(np.expand_dims(obs, axis=0))
                else:
                    val = 0
                buf.finish_episode(last_val=val)
                episodic_returns.append(ep_rew)
                sedimentary_returns.append(sum(episodic_returns)/episode_counter)
                # Update actor critic
                if not episode_counter%update_every:
                    # pdb.set_trace()
                    batched_actor_dataset, batched_critic_dataset = buf.get()
#                     agent.train(batched_actor_dataset, batched_critic_dataset, num_epochs=80)
#                     buf = PPOBuffer(size=buffer_size)
#                 logging.info("\n======== \nEpisode: {} \nEpLength: {} \nTotalReward: {} \nSedimentaryReturn: {}".format(
#                     ep+1,
#                     st+1,
#                     ep_rew,
#                     sedimentary_returns[-1]
#                 ))
                break

