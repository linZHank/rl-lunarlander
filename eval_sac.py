import sys
import os
import numpy as np
import random
import gym
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)
from train_sac import SoftActorCritic


# Create LunarLander env
env = gym.make('LunarLanderContinuous-v2')

# load model
# model_path = './training_models/ppo/LunarLanderContinuous-v2/models/199'
model_path = './models/sac_auto_ent/LunarLanderContinuous-v2/4256'
sac = SoftActorCritic(obs_dim=8, act_dim=2)
sac.pi.policy_net = tf.keras.models.load_model(model_path)
# params
num_episodes = 10
ep_rets, ave_rets = [], []
num_steps = env.spec.max_episode_steps

# Evaluate
for ep in range(10):
    o, d, ep_ret = env.reset(), False, 0
    for st in range(num_steps):
        env.render()
        a = np.squeeze(sac.act(o.reshape(1,-1), deterministic=True))
        o2,r,d,_ = env.step(a)
        ep_ret += r
        o = o2
        if d:
            print("EpReturn: {}".format(ep_ret))
            break 
