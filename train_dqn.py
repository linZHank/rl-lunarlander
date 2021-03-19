import os
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)

import tensorflow as tf
from agents.dqn import DQNAgent, DQNBuffer


# instantiate env
env = gym.make('LunarLander-v2')
dim_obs = env.observation_space.shape
num_act = env.action_space.n
# instantiate agent and replay buffer
agent = DQNAgent()
replay_buffer = DQNBuffer(dim_obs=dim_obs[0], size=int(1e6))
save_dir = './saved_models/'+env.spec.id+'/dqn/'+datetime.now().strftime("%Y-%m-%d-%H-%M")+'/'
if not os.path.exists(save_dir):
    try:
        os.makedirs(save_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
value_net_path = os.path.join(save_dir, 'value_net')
RANDOM_SEED = 0
tf.random.set_seed(RANDOM_SEED)
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
env.action_space.seed(RANDOM_SEED)
# paramas
num_episodes = int(1500)
batch_size = 128
update_freq = 50
update_after = 1000
decay_period = 500
warmup_episodes = 20
save_freq = 50
# variables
ep_cntr, st_cntr = 0, 0
stepwise_rewards, episodic_returns, sedimentary_returns = [], [], []
episodic_steps = []
start_time = time.time()
# main loop
for e in range(num_episodes):
    obs, ep_ret, ep_len = env.reset(), 0, 0
    agent.linear_epsilon_decay(episode=e, decay_period=decay_period, warmup_episodes=warmup_episodes)
    for t in range(env.spec.max_episode_steps):
        act = agent.make_decision(np.expand_dims(obs, 0))
        nobs, rew, done, info = env.step(act.numpy())
        ep_ret += rew
        ep_len += 1
        st_cntr += 1
        replay_buffer.store(obs, act, rew, done, nobs)
        obs = nobs.copy() # SUPER CRITICAL!!!
        if not st_cntr%update_freq and st_cntr>=update_after:
            for _ in range(update_freq):
                minibatch = replay_buffer.sample_batch(batch_size=batch_size)
                loss_q = agent.train(data=minibatch)
                logging.debug("\nloss_q: {}".format(loss_q))
        if done or ep_len>=env.spec.max_episode_steps:
            ep_cntr += 1
            episodic_returns.append(ep_ret)
            sedimentary_returns.append(sum(episodic_returns)/ep_cntr)
            episodic_steps.append(st_cntr)
            logging.debug("\n----\nEpisode: {}, epsilon: {}, EpisodeLength: {}, TotalSteps: {}, StepsInLoop: {}, \nEpReturn: {}\n----\n".format(ep_cntr, agent.epsilon, ep_len, st_cntr, ep_len, ep_ret))
            if not ep_cntr%save_freq:
                agent.qnet.value_net.save(value_net_path) # save model
            break

# Save returns 
np.save(os.path.join(save_dir, 'episodic_returns.npy'), episodic_returns)
np.save(os.path.join(save_dir, 'sedimentary_returns.npy'), sedimentary_returns)
np.save(os.path.join(save_dir, 'episodic_steps.npy'), episodic_steps)
with open(os.path.join(save_dir, 'training_time.txt'), 'w') as f:
    f.write("{}".format(time.time()-start_time))
# plot returns
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle('Averaged Returns')
ax.plot(sedimentary_returns)
plt.show()

# Test trained model
input("Press ENTER to test lander...")
for ep in range(10):
    o, ep_ret = env.reset(), 0
    for st in range(env.spec.max_episode_steps):
        env.render()
        act = agent.make_decision(np.expand_dims(o, 0))
        o2, r, d, _ = env.step(act.numpy())
        ep_ret += r
        o = o2
        if d:
            print("\n---\nepisode: {}, episode return: {}\n---\n".format(ep+1, ep_ret))
            break

