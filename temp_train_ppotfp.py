import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from agents.ppo_tfp import PPOAgent, PPOBuffer

# instantiate env
env = gym.make('LunarLander-v2')
dim_obs = env.observation_space.shape
num_act = env.action_space.n
dim_act = 1
# instantiate actor-critic and replay buffer
agent = PPOAgent()
replay_buffer = PPOBuffer(max_size=6000)  # max_size is the upper-bound
RANDOM_SEED = 0
tf.random.set_seed(RANDOM_SEED)
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
env.action_space.seed(RANDOM_SEED)
# paramas
min_steps_per_train = replay_buffer.max_size - env.spec.max_episode_steps
assert min_steps_per_train>0
num_trains = 64
train_epochs = 80
# variables
ep_cntr, st_cntr = 0, 0
stepwise_rewards, episodic_returns, sedimentary_returns = [], [], []
episodic_steps = []
# main loop
obs, ep_ret, ep_len = env.reset(), 0, 0
for t in range(num_trains):
    for s in range(replay_buffer.max_size):
        act, val, lpa = agent.make_decision(np.expand_dims(obs,0)) 
        nobs, rew, done, _ = env.step(act.numpy())
        stepwise_rewards.append(rew)
        ep_ret += rew
        ep_len += 1
        st_cntr += 1
        replay_buffer.store(obs, act, rew, val, lpa)
        obs = nobs # SUPER CRITICAL!!!
        if done or ep_len>=env.spec.max_episode_steps:
            val = 0.
            if ep_len>=env.spec.max_episode_steps:
                _, val, _ = agent.make_decision(np.expand_dims(obs,0))
            replay_buffer.finish_path(val)
            # summarize episode
            ep_cntr += 1
            episodic_returns.append(ep_ret)
            sedimentary_returns.append(sum(episodic_returns)/ep_cntr)
            episodic_steps.append(st_cntr)
            # print("\n----\nEpisode: {}, EpisodeLength: {}, TotalSteps: {}, StepsInLoop: {}, \nEpReturn: {}\n----\n".format(ep_cntr, ep_len, st_cntr, s+1, ep_ret))
            obs, ep_ret, ep_len = env.reset(), 0, 0
            if s+1>=min_steps_per_train:
                break
    # update actor-critic
    tic = time.time()
    data = replay_buffer.get()
    loss_pi, loss_v, loss_info = agent.train(data, train_epochs)
    toc = time.time()
    print("\n====")
    print(f"Training: {t+1}")
    print(f"TotalSteps: {st_cntr}")
    print(f"DataSize: {data['ret'].shape[0]}")
    print(f"AveReturn: {sedimentary_returns[-1]}")
    print(f"LossPi: {loss_pi}")
    print(f"LossV: {loss_v}")
    print(f"KLDivergence: {loss_info['kld']}")
    print(f"Entropy: {loss_info['entropy']}")
    print(f"TrainingTime: {toc-tic}")
    print("====\n")
    # Save model

# Save returns 
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
        act, _, _ = agent.make_decision(np.expand_dims(obs, 0))
        next_obs, rew, done, info = env.step(act.numpy())
        rewards.append(rew)
        # print("\n-\nepisode: {}, step: {} \naction: {} \nobs: {}, \nreward: {}".format(ep+1, st+1, act, obs, rew))
        obs = next_obs
        if done:
            ep_rets.append(sum(rewards))
            ave_rets.append(sum(ep_rets)/len(ep_rets))
            print("\n---\nepisode: {} \nepisode return: {}, averaged return: {} \n---\n".format(ep+1, ep_rets[-1], ave_rets[-1]))
            break

