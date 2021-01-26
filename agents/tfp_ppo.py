""" 
A PPO type agent class for LunarLander env 
"""
import numpy as np
import scipy.signal
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


#############################Setup##############################
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
# set log level
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)
#############################Setup##############################


#############################Buffer#############################
"""
On-policy Replay Buffer 
"""
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x, [x0, x1, x2]
    output:
        [x0+discount*x1+discount^2*x2, x1+discount*x2, x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPOBuffer: 
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, dim_obs, dim_act, max_size, gamma=.99, lam=0.97):
        self.obs_buf = np.zeros((max_size, dim_obs), dtype=np.float32)
        self.act_buf = np.zeros((max_size, dim_act), dtype=np.float32)
        self.adv_buf = np.zeros(max_size, dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.ret_buf = np.zeros(max_size, dtype=np.float32)
        self.val_buf = np.zeros(max_size, dtype=np.float32)
        self.lpa_buf = np.zeros(max_size, dtype=np.float32)
        # params
        self.dim_obs=dim_obs
        self.dim_act=dim_act
        self.gamma=gamma
        self.lam=lam
        self.max_size=max_size 
        # vars
        self.ptr, self.path_start_idx, = 0, 0

    def store(self, obs, act, rew, val, lpa):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr <= self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.lpa_buf[self.ptr] = lpa
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, to compute advantage estimates with GAE-Lambda, as well as compute the rewards-to-go.
        The "last_val" argument should be 0 if the trajectory ended, and otherwise should be V(s_T).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma*self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        # set new path start idx 
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from the buffer, with advantages normalized. Also, resets the buffer.
        """
        assert self.ptr<=self.max_size
        self.obs_buf = self.obs_buf[:self.ptr] 
        self.act_buf = self.act_buf[:self.ptr] 
        self.rew_buf = self.rew_buf[:self.ptr] 
        self.val_buf = self.val_buf[:self.ptr] 
        self.ret_buf = self.ret_buf[:self.ptr] 
        self.adv_buf = self.adv_buf[:self.ptr] 
        self.lpa_buf = self.lpa_buf[:self.ptr]
        # the next three lines implement the advantage normalization trick
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, lpa=self.lpa_buf)
        # reset buffer
        self.__init__(self.dim_obs, self.dim_act, self.max_size)
        return {k: tf.convert_to_tensor(v, dtype=tf.float32) for k,v in data.items()}
#############################Buffer#############################


#############################Agent##############################
class CategoricalActor(tf.keras.Model):

    def __init__(self, dim_obs, num_act, **kwargs):
        super(CategoricalActor, self).__init__(name='categorical_actor', **kwargs)
        self.dim_obs=dim_obs
        self.num_act=num_act
        self.policy_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=dim_obs, name='actor_inputs'),
                tf.keras.layers.Dense(256, activation='tanh'),
                tf.keras.layers.Dense(256, activation='tanh'),
                tf.keras.layers.Dense(num_act, activation=None, name='actor_outputs')
            ]
        )

    def _distribution(self, obs):
        logits = tf.squeeze(self.policy_net(obs)) # squeeze to deal with size 1
        d = tfd.Categorical(logits=logits)
        return d

    def _logprob(self, distribution, act):
        logp_a = distribution.log_prob(act)
        return logp_a
        
    def call(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._logprob(pi, act)
        return pi, logp_a

class Critic(tf.keras.Model):
    def __init__(self, dim_obs, **kwargs):
        super(Critic, self).__init__(name='critic', **kwargs)
        self.value_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=dim_obs, name='critic_inputs'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(1, activation=None, name='critic_outputs')
            ]
        )

    @tf.function
    def call(self, obs):
        return tf.squeeze(self.value_net(obs))

class PPOAgent:
    def __init__(self, name='ppo_agent', continuous=False, dim_obs=8, num_act=4, dim_act=1, clip_ratio=0.2, lr_actor=3e-4, lr_critic=1e-3, target_kld=0.2, beta=0.001):
        # params
        self.continuous=continuous
        self.clip_ratio = clip_ratio
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.beta = beta
        self.target_kld = target_kld
        # modules
        self.actor = CategoricalActor(dim_obs, num_act)
        self.critic = Critic(dim_obs)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_critic)
        self.mse = tf.keras.losses.MeanSquaredError()

    def make_decision(self, obs):

        pi = self.actor._distribution(obs)
        act = pi.sample()
        logp_a = self.actor._logprob(pi, act)
        val = self.critic(obs)

        return act.numpy(), val.numpy(), logp_a.numpy()

    def train(self, data, epochs):
    
        # update actor
        ep_loss_pi = []
        ep_kld = []
        ep_ent = []
        for ep in range(epochs):
            logging.debug("Staring actor training epoch: {}".format(ep+1))
            with tf.GradientTape() as tape:
                tape.watch(self.actor.trainable_variables)
                pi, logp = self.actor(data['obs'], data['act'])
                ratio = tf.math.exp(logp - data['lpa']) # pi/old_pi
                clip_adv = tf.math.multiply(tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio), data['adv'])
                approx_kld = data['lpa'] - logp
                ent = pi.entropy()
                obj = tf.math.minimum(ratio*data['adv'], clip_adv) + self.beta*ent
                loss_pi = -tf.math.reduce_mean(obj)
            # gradient descent actor weights
            grads_actor = tape.gradient(loss_pi, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
            ep_loss_pi.append(loss_pi)
            ep_kld.append(tf.math.reduce_mean(approx_kld))
            ep_ent.append(tf.math.reduce_mean(ent))
            # log epoch
            logging.info("\n----Actor Training----\nEpoch :{} \nLoss: {} \nKLDivergence: {} \nEntropy: {}".format(
                ep+1,
                loss_pi,
                ep_kld[-1],
                ep_ent[-1],
            ))
            # early cutoff due to large kl-divergence
            if ep_kld[-1] > self.target_kld:
                logging.warning("\nEarly stopping at epoch {} due to reaching max kl-divergence.\n".format(ep+1))
                break
        mean_loss_pi = tf.math.reduce_mean(ep_loss_pi)
        mean_ent = tf.math.reduce_mean(ep_ent)
        mean_kld = tf.math.reduce_mean(ep_kld)
        # update critic
        ep_loss_val = []
        for ep in range(epochs):
            logging.debug("Starting critic training epoch: {}".format(ep+1))
            with tf.GradientTape() as tape:
                tape.watch(self.critic.trainable_variables)
                loss_val = self.mse(data['ret'], self.critic(data['obs']))
            # gradient descent critic weights
            grads_critic = tape.gradient(loss_val, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_variables))
            ep_loss_val.append(loss_val)
            # log loss_v
            logging.info("\n----Critic Training----\nEpoch :{} \nLoss: {}".format(
                ep+1,
                loss_val
            ))
        mean_loss_val = tf.math.reduce_mean(ep_loss_val)

        return mean_loss_pi, mean_loss_val, dict(kld=mean_kld, entropy=mean_ent)
#############################Agent##############################
        

# Uncomment following for testing the PPO agent
# agent = PPOAgent(8,4)
# o = np.random.uniform(-5,5,(10,8))
# a = np.random.randint(0,4,(10,))
# r = np.random.uniform(-1,4,(10,))
# l = np.random.uniform(-2,0,(10,))
# ad = np.random.normal(4,1,(10,))
# d = dict(
#     obs=o,
#     act=a,
#     ret=r,
#     adv=ad,
#     logp=l
# )
# data = {k: tf.convert_to_tensor(v, dtype=tf.float32) for k,v in d.items()}
# 
# lp,lv,i = agent.train(data,10)
