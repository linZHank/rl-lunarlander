import tensorflow as tf
from agents.dqn import DQNAgent, DQNBuffer
import time
import gym
import numpy as np
import logging

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)


# instantiate env
env = gym.make("LunarLander-v2")
dim_obs = env.observation_space.shape[0]
num_act = env.action_space.n
# instantiate agent and replay buffer
agent = DQNAgent(dim_obs=dim_obs, num_act=num_act)
buf = DQNBuffer(dim_obs=dim_obs, size=int(1e6))
RANDOM_SEED = 19
tf.random.set_seed(RANDOM_SEED)
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
env.action_space.seed(RANDOM_SEED)
# init env
pobs = env.reset()
done = False
ep, ep_return = 0, 0
# main loop
tic = time.time()
for st in range(int(3e5)):
    act = agent.make_decision(np.expand_dims(pobs, 0))
    nobs, rew, done, info = env.step(act.numpy())
    # accumulate experience
    buf.store(pobs, act, rew, done, nobs)
    ep_return += rew
    pobs = nobs
    # learn
    if st + 1 >= 1024:
        minibatch = buf.sample_batch(batch_size=1024)
        loss_q = agent.train_one_batch(data=minibatch)
        # print(f"\nloss_q: {loss_q}")
    if done:
        print(f"episode {ep+1} return: {ep_return}")
        print(f"total steps: {st+1}")
        pobs = env.reset()
        done = False
        ep_return = 0
        ep += 1
        agent.linear_epsilon_decay(episode=ep, decay_period=500, warmup_episodes=0)

# time
toc = time.time()
print(f"episode {ep+1} return: {ep_return}")
print(f"total steps: {st+1}")
print(f"Training time: {toc-tic}")
