""" 
A DQN type agent class: Compute Q-value with both state and action as inputs. 
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
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, dim_obs, size):
        self.obs_buf = np.zeros((size, dim_obs), dtype=np.float32)
        self.act_buf = np.zeros((size, 1), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.nobs_buf = np.zeros((size, dim_obs), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, done, nobs):
        self.obs_buf[self.ptr] = obs
        self.nobs_buf[self.ptr] = nobs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        slices = np.random.randint(0, self.size, size=batch_size)
        data = dict(
            obs=tf.convert_to_tensor(self.obs_buf[slices]),
            act=tf.convert_to_tensor(self.act_buf[slices]),
            rew=tf.convert_to_tensor(self.rew_buf[slices]),
            done=tf.convert_to_tensor(self.done_buf[slices]),
            nobs=tf.convert_to_tensor(self.nobs_buf[slices])
        )
        return data

class Critic(tf.keras.Model):
    def __init__(self, dim_obs, **kwargs):
        super(Critic, self).__init__(name='critic', **kwargs)
        self.dim_obs = dim_obs
        inputs_s = tf.keras.Input(shape=dim_obs, name='critic_inputs_state')
        inputs_a = tf.keras.Input(shape=(1,), name='critic_inputs_action')
        x = tf.keras.layers.concatenate([inputs_s, inputs_a])
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation=None)(x)
        self.value_net = tf.keras.Model(inputs=[inputs_s, inputs_a], outputs=outputs)

    @tf.function
    def call(self, obs, act):
        val = tf.squeeze(self.value_net([obs, act]))
        return val

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
        self.final_eps = final_eps
        self.lr = lr
        self.gamma = gamma
        self.polyak = polyak
        # variables
        self.epsilon = init_eps
        # DQN module
        self.qnet = Critic(dim_obs) 
        self.targ_qnet = Critic(dim_obs)
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def linear_epsilon_decay(self, episode, decay_period=1000, warmup_episodes=0):
        episodes_left = decay_period + warmup_episodes - episode
        bonus = (self.init_eps - self.final_eps) * episodes_left / decay_period
        bonus = np.clip(bonus, 0., self.init_eps-self.final_eps)
        self.epsilon = self.final_eps + bonus

    @tf.function
    def make_decision(self, obs):
        if tf.random.uniform(shape=())>self.epsilon:
            act = tf.math.argmax([self.qnet(obs, tf.expand_dims([i],0)) for i in range(self.num_act)], axis=-1)
        else:
            act = tf.random.uniform(shape=(), minval=0, maxval=self.num_act, dtype=tf.int64)
        return act

    def train(self, data):
        """
        Train one batch
        """
        # update critic
        with tf.GradientTape() as tape:
            tape.watch(self.qnet.trainable_variables)
            pred_qval = self.qnet(data['obs'], data['act'])
            next_q = self.targ_qnet([data['nobs'], tf.ones(shape=(data['nobs'].shape[0], 1))])

            pred_qval = tf.math.reduce_sum(self.qnet(data['obs'])*tf.one_hot(data['act'], self.num_act), axis=-1)
            # next two lines implement Doubld DQN trick
            id_nexta = tf.argmax(self.qnet(data['nobs']), axis=-1)
            next_q = tf.math.reduce_sum(self.targ_qnet(data['nobs'])*tf.one_hot(id_nexta, self.num_act), axis=-1) 
            targ_qval = data['rew'] + self.gamma*(1 - data['done'])*next_q
            # targ_qval = data['rew'] + self.gamma*(1 - data['done'])*tf.math.reduce_sum(self.targ_qnet(data['nobs'])*tf.one_hot(tf.math.argmax(self.qnet(data['nobs']), axis=1), self.num_act), axis=-1) # double DQN trick
            # loss_q = tf.keras.losses.MSE(y_true=targ_qval, y_pred=pred_qval)
            loss_q = self.loss_fn(targ_qval, pred_qval)
        grads = tape.gradient(loss_q, self.qnet.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.qnet.trainable_weights))
        # Polyak average update target Q-nets
        q_weights_update = []
        for w_q, w_targ_q in zip(self.qnet.get_weights(), self.targ_qnet.get_weights()):
            w_q_upd = self.polyak*w_targ_q
            w_q_upd = w_q_upd + (1 - self.polyak)*w_q
            q_weights_update.append(w_q_upd)
        self.targ_qnet.set_weights(q_weights_update)

        return loss_q


#############################Test##############################
# Uncomment following for testing the PPO agent
import gym
env = gym.make('LunarLander-v2')
dim_obs = env.observation_space.shape
num_act = env.action_space.n
agent = DQNAgent()
rb = DQNBuffer(dim_obs=dim_obs[0], size=int(1e4))
o = env.reset()
for _ in range(100):
    a = agent.make_decision(np.expand_dims(o, 0))
    o2, r, d, i = env.step(a.numpy())
    rb.store(o,a,r,d,o2)
    o = o2
    if d:
        break
data = rb.sample_batch(1024)
# loss_q = agent.train(data)
#############################Test##############################
