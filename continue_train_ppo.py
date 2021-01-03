import sys
import os
import numpy as np
import random
import time
import gym
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
print(tf.__version__)
import logging
# set log level
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)

from train_ppo import PPOAgent, PPOBuffer



"""
Main
"""
env = gym.make('LunarLanderContinuous-v2')
# paramas
steps_per_epoch=5000
epochs=20
gamma=0.99
train_iters=80
lam=0.97
max_ep_len=1000
save_freq=10
# instantiate actor-critic and replay buffer
dim_obs=env.observation_space.shape[0]
dim_act=env.action_space.shape[0]
agent = PPOAgent(dim_obs=dim_obs, dim_act=dim_act, beta=0.)
replay_buffer = PPOBuffer(dim_obs, dim_act, steps_per_epoch, gamma, lam)
# load models
model_dir = './models/ppo/LunarLanderContinuous-v2/'
mean_path = os.path.join(model_dir, 'actor', '199')
logstd_path = os.path.join(model_dir, 'actor', 'log_std.npy')
logstd = np.load(logstd_path)
val_path = os.path.join(model_dir, 'critic', '199')
agent.actor.mean_net = tf.keras.models.load_model(mean_path)
agent.actor.log_std = tf.Variable(initial_value=logstd)
agent.critic.val_net = tf.keras.models.load_model(val_path)

# Prepare for interaction with environment
model_dir = './models/continue_ppo/'+env.spec.id
obs, ep_ret, ep_len = env.reset(), 0, 0
episodes, total_steps = 0, 0
stepwise_rewards, episodic_returns, sedimentary_returns = [], [], []
episodic_steps = []
start_time = time.time()
# main loop
for ep in range(epochs):
    for st in range(steps_per_epoch):
        act, val, logp = agent.pi_of_a_given_s(np.expand_dims(obs, 0))
        next_obs, rew, done, _ = env.step(act)
        ep_ret += rew
        ep_len += 1
        stepwise_rewards.append(rew)
        total_steps += 1
        replay_buffer.store(obs, act, rew, np.squeeze(val), logp)
        obs = next_obs # SUPER CRITICAL!!!
        # handle episode termination
        timeout = (ep_len==env.spec.max_episode_steps)
        terminal = done or timeout
        epoch_ended = (st==steps_per_epoch-1)
        if terminal or epoch_ended:
            if epoch_ended and not(terminal):
                print('Warning: trajectory cut off by epoch at {} steps.'.format(ep_len), flush=True)
            if timeout or epoch_ended:
                _, val, _ = agent.pi_of_a_given_s(np.expand_dims(obs,0))
            else:
                val = [0]
            replay_buffer.finish_path(np.squeeze(val))
            if terminal:
                episodes += 1
                episodic_returns.append(ep_ret)
                sedimentary_returns.append(sum(episodic_returns)/episodes)
                episodic_steps.append(total_steps)
                print("\n====\nTotalSteps: {} \nEpisode: {}, Step: {}, EpReturn: {}, EpLength: {} \n====\n".format(total_steps, episodes, st+1, ep_ret, ep_len))
            obs, ep_ret, ep_len = env.reset(), 0, 0
    # Save model
    if not ep%save_freq or (ep==epochs-1):
        mean_path = os.path.join(model_dir, 'mean_net', str(ep))
        if not os.path.exists(os.path.dirname(mean_path)):
            os.makedirs(os.path.dirname(mean_path))
        val_path = os.path.join(model_dir, 'val_net', str(ep))
        if not os.path.exists(os.path.dirname(val_path)):
            os.makedirs(os.path.dirname(val_path))
        agent.actor.mean_net.save(mean_path)
        np.save(model_dir+'log_std.npy', agent.actor.log_std.numpy())
        agent.critic.val_net.save(val_path)

    # update actor-critic
    loss_pi, loss_v, loss_info = agent.train(replay_buffer.get(), train_iters)
    print("\n================================================================\nEpoch: {} \nStep: {} \nAveReturn: {} \nLossPi: {} \nLossV: {} \nKLDivergence: {} \nEntropy: {} \nTimeElapsed: {}\n================================================================\n".format(ep+1, st+1, sedimentary_returns[-1], loss_pi, loss_v, loss_info['kl'], loss_info['ent'], time.time()-start_time))
################################################################

# Save returns 
np.save(os.path.join(model_dir, 'episodic_returns.npy'), episodic_returns)
np.save(os.path.join(model_dir, 'sedimentary_returns.npy'), sedimentary_returns)
np.save(os.path.join(model_dir, 'episodic_steps.npy'), episodic_steps)
with open(os.path.join(model_dir, 'training_time.txt'), 'w') as f:
    f.write("{}".format(time.time()-start_time))
# plot returns
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle('Averaged Returns')
ax.plot(sedimentary_returns)
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
        act, _, _ = agent.pi_of_a_given_s(np.expand_dims(obs, 0))
        next_obs, rew, done, info = env.step(act)
        rewards.append(rew)
        # print("\n-\nepisode: {}, step: {} \naction: {} \nobs: {}, \nreward: {}".format(ep+1, st+1, act, obs, rew))
        obs = next_obs.copy()
        if done:
            ep_rets.append(sum(rewards))
            ave_rets.append(sum(ep_rets)/len(ep_rets))
            print("\n---\nepisode: {} \nepisode return: {}, averaged return: {} \n---\n".format(ep+1, ep_rets[-1], ave_rets[-1]))
            break
