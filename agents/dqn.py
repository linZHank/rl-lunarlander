""" 
A DQN type agent class for pe_env_discrete 
"""
import tensorflow as tf
import numpy as np
import logging

################################################################
"""
Can safely ignore this block
"""
# restrict GPU and memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
################################################################
class DQNBuffer:
    """
    An off-policy replay buffer for DQN agent
    """
    def __init__(self, dim_obs, max_size):
        # property
        self.max_size = max_size
        # buffers
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.nobs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        # variables
        self.ptr, self.size = 0, 0 

    def store(self, obs, act, rew, done, nobs):
        self.obs_buf[self.ptr] = obs
        self.nobs_buf[self.ptr] = nobs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size):
        sample_ids = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=tf.convert_to_tensor(self.obs_buf[sample_ids]),
                     nobs=tf.convert_to_tensor(self.nobs_buf[sample_ids]),
                     act=tf.convert_to_tensor(self.act_buf[sample_ids]),
                     rew=tf.convert_to_tensor(self.rew_buf[sample_ids]),
                     done=tf.convert_to_tensor(self.done_buf[sample_ids]))
        return batch

class Critic(tf.keras.Model):
    def __init__(self, dim_obs, num_act, **kwargs):
        super(Critic, self).__init__(name='critic', **kwargs)
        self.dim_obs = dim_obs
        self.num_act = num_act
        self.value_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=dim_obs, name='critic_inputs'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(num_act, activation=None, name='critic_outputs')
            ]
        )

    @tf.function
    def maxq(self, obs):
        return tf.math.reduce_max(self.value_net(obs), axis=-1)

    @tf.function
    def call(self, obs):
        return tf.squeeze(self.value_net(obs))

class DQNAgent(tf.keras.Model):
    """
    DQN agent class. epsilon decay, epsilon greedy, train, etc..
    """
    def __init__(self, dim_obs=(8,), num_act=4, lr=3e-4, gamma=0.99, polyak=.995, init_eps=1., final_eps=.1, **kwargs):
        super(DQNAgent, self).__init__(name='dqn', **kwargs)
        # hyper parameters
        self.dim_obs = dim_obs
        self.num_act = num_act
        self.init_eps= init_eps
        self.final_eps= final_eps
        self.gamma = gamma
        self.polyak = polyak
        # variables
        self.epsilon = init_eps
        # DQN module
        self.qnet = Critic(obs_dim, act_dim) 
        self.targ_qnet = Critic(obs_dim, act_dim)
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)

    def linear_epsilon_decay(self, episode, decay_period=1000, warmup_episodes=0):
        episodes_left = decay_period + warmup_episodes - episode
        bonus = (self.init_eps - self.final_eps) * episodes_left / decay_period
        bonus = np.clip(bonus, 0., self.init_eps-self.final_eps)
        self.epsilon = self.final_eps + bonus

    @tf.function
    def make_decision(self, obs):
        if tf.random.uniform(shape=())>self.epsilon:
            a = tf.math.argmax(self.qnet(obs), axis=-1)
        else:
            a = np.random.uniform(shape=[], minval=0, maxval=self.num_act, dtype=tf.int64)
        return a

    # @tf.function
    def train_one_step(self):
        minibatch = self.replay_buffer.sample_batch(batch_size=self.batch_size)
        with tf.GradientTape() as tape:
            # compute current Q
            vals = self.dqn_active([minibatch['img'], minibatch['odom']])
            oh_acts = tf.one_hot(minibatch['act'], depth=self.dim_act)
            pred_qvals = tf.math.reduce_sum(tf.math.multiply(vals, oh_acts), axis=-1)
            # compute target Q
            nxt_vals = self.dqn_stable([minibatch['nxt_img'], minibatch['nxt_odom']])
            nxt_acts = tf.math.argmax(self.dqn_active([minibatch['nxt_img'], minibatch['nxt_odom']]), axis=-1)
            oh_nxt_acts = tf.one_hot(nxt_acts, depth=self.dim_act)
            nxt_qvals = tf.math.reduce_sum(tf.math.multiply(nxt_vals, oh_nxt_acts), axis=-1)
            targ_qvals = minibatch['rew'] + (1. - minibatch['done'])*self.gamma*nxt_qvals
            # compute loss
            loss_q = self.loss_fn(y_true=targ_qvals, y_pred=pred_qvals)
            logging.info("loss_Q: {}".format(loss_q))
        # gradient decent
        grads = tape.gradient(loss_q, self.dqn_active.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.dqn_active.trainable_weights))
        self.fit_cntr += 1
        # update dqn_stable if C steps of q_val fitted
        if not self.fit_cntr%self.sync_step:
            self.dqn_stable.set_weights(self.dqn_active.get_weights())


if __name__=='__main__':
    agent = DQNAgent(name='test_dqn_agent')
    test_img = np.random.rand(4,150,150,3)
    test_odom = np.random.randn(4,4)
    qvals = agent.dqn_active([test_img, test_odom])
    print("qvals: {}".format(qvals))

