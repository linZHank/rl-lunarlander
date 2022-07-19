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
rng = hk.PRNGSequence(jax.random.PRNGKey(19))
params = agent.init_params(next(rng))
learner_state = agent.init_learner(params)
buf = ReplayBuffer(capacity=int(1e6))
total_steps = 0
for ep in range(20):
    # Prepare agent, environment and accumulator for a new episode.
    actor_state = agent.init_actor()
    pobs = env.reset()
    done = False
    rew, ep_return = 0, 0
    while not done:
        # env.render()
        actor_output = agent.make_decision(
            params=params,
            obs=pobs,
            episode_count=ep,
            key=next(rng),
            eval_flag=False,
        )
        act = int(actor_output.action)
        nobs, rew, done, info = env.step(act)
        total_steps += 1
        # accumulate experience
        buf.store(pobs, act, rew, done, nobs)
        ep_return += rew
        pobs = nobs
        # learn
        if buf.is_ready(batch_size=1024):
            params, learner_state = agent.learn_step(
                params, buf.sample(batch_size=1024, discount_factor=0.99), learner_state, next(rng)
            )
    print(f"episode {ep+1} return: {ep_return}")
    print(f"total steps: {total_steps}")
