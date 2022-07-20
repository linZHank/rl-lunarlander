import os
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

import jax
import haiku as hk
from agents.dqnax import DQNAgent, ReplayBuffer


# instantiate env
env = gym.make("LunarLander-v2")
agent = DQNAgent(
    observation_space=env.observation_space,
    action_space=env.action_space,
    target_period=100,
    learning_rate=3e-4,
)
RANDOM_SEED = 19
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
env.action_space.seed(RANDOM_SEED)
rng = hk.PRNGSequence(jax.random.PRNGKey(RANDOM_SEED))
params = agent.init_params(next(rng))
learner_state = agent.init_learner(params)
buf = ReplayBuffer(capacity=int(1e6))
# init env
pobs = env.reset()
done = False
ep, ep_return = 0, 0
# start learning
tic = time.time()
for st in range(int(3e5)):
    actor_output = agent.make_decision(
        params=params,
        obs=pobs,
        episode_count=ep,
        key=next(rng),
        eval_flag=False,
    )
    act = int(actor_output.action)
    nobs, rew, done, info = env.step(act)
    # accumulate experience
    buf.store(pobs, act, rew, done, nobs)
    ep_return += rew
    pobs = nobs
    # learn
    if buf.is_ready(batch_size=1024):
        params, learner_state = agent.learn_step(
            params, buf.sample(batch_size=1024, discount_factor=0.99), learner_state, next(rng)
        )
    if done:
        print(f"episode {ep+1} return: {ep_return}")
        print(f"total steps: {st+1}")
        pobs = env.reset()
        done = False
        ep_return = 0
        ep += 1

# time
toc = time.time()
print(f"episode {ep+1} return: {ep_return}")
print(f"total steps: {st+1}")
print(f"Training time: {toc-tic}")
