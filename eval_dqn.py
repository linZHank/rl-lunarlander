import sys
import os
import numpy as np
import random
import gym
import matplotlib.pyplot as plt
import tensorflow as tf

import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)
from train_dqn import DeepQNet


# Create LunarLander env
env = gym.make('LunarLander-v2')

# Evaluate
model_path = './models/dqn/'+env.spec.id+'/2927'
dqn = DeepQNet(obs_dim=8, act_dim=4)
dqn.q.q_net = tf.keras.models.load_model(model_path)
dqn.epsilon = 0.

for ep in range(10):
    o, d, ep_ret = env.reset(), False, 0
    for st in range(env.spec.max_episode_steps):
        env.render()
        a = np.squeeze(dqn.act(o.reshape(1,-1)))
        o2,r,d,_ = env.step(a)
        ep_ret += r
        o = o2
        if d:
            print("EpReturn: {}".format(ep_ret))
            break 
