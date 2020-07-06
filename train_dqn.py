import tensorflow as tf
import numpy as np

class DeepQNetwork(tf.keras.layers.Layer):
    def __init__(obs_dim=8, act_dim=4, layer_sizes=[128,128], activation='relu'):
        super(DeepQNetwork, self).__init__():
            self.dense_1 = tf.keras.layers.Dense(layer_sizes[0], activation=activation)
            self.dense_2 = tf.keras.layers.Dense(layer_sizes[1], activation=activation)
            self.dense_q = tf.keras.layers.Dense(act_dim, activation=None)

    def call(self, obs):
        x = self.dense_1(obs)
        x = self.dense_2(x)
        q_vals = self.dense_q(x)

        return q_vals

                
