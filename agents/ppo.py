""" 
A PPO type agent class for LunarLander env 
"""
import tensorflow as tf
import numpy as np
import logging
import pdb


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
# set log level
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)
################################################################


class CategoricalActor(tf.keras.Model):

    def __init__(self, dim_obs, num_act, **kwargs):
        super(CategoricalActor, self).__init__(name='categorical_actor', **kwargs)
        self.dim_obs=dim_obs
        self.num_act=num_act
        self.policy_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=dim_obs, name='actor_inputs'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(num_act, activation='softmax', name='actor_outputs')
            ]
        )


    @tf.function
    def logprob(self, obs, act):
        pmf = self.call(obs) # PMF
        act_oh = tf.squeeze(tf.one_hot(tf.cast(act,tf.int32), self.num_act))
        p_a = tf.math.reduce_sum(tf.math.multiply(pmf, act_oh), axis=-1)
        logp_a = tf.squeeze(tf.math.log(p_a))
        return pmf, logp_a

    @tf.function
    def call(self, obs):
        pmf = self.policy_net(obs)
        return pmf

# class GaussianActor(tf.Module):
#     def __init__(self, dim_obs, dim_act):
#         super().__init__()
#         self.log_std = tf.Variable(initial_value=-0.5*np.ones(dim_act, dtype=np.float32))
#         self.mu_net = mlp(dim_inputs=dim_obs, dim_outputs=dim_act, activation='relu')
# 
#     def _distribution(self, obs):
#         mu = tf.squeeze(self.mu_net(obs))
#         std = tf.math.exp(self.log_std)
# 
#         return tfd.Normal(loc=mu, scale=std)
# 
#     def _log_prob_from_distribution(self, pi, act):
#         return tf.math.reduce_sum(pi.log_prob(act), axis=-1)
# 
#     def call(self, obs, act=None):
#         # def log_normal_pdf(sample, mean, log_std, raxis=1):
#         #     log2pi = tf.math.log(2.*np.pi)
#         #     return tf.reduce_sum(-.5*(((sample-mean)*tf.math.exp(-log_std))**2 + 2*log_std + log2pi), axis=raxis)
#         pi = self._distribution(obs)
#         logp_a = None
#         if act is not None:
#             logp_a = self._log_prob_from_distribution(pi, act)
#             
# 
#         return pi, logp_a

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
        return self.value_net(obs)

class PPOAgent:
    def __init__(self, name='ppo_agent', continuous=False, dim_obs=8, num_act=4, dim_act=1, clip_ratio=0.2, lr_actor=1e-4, lr_critic=1e-3, target_kl=0.01, beta=0.001):
        # params
        self.continuous=continuous
        self.clip_ratio = clip_ratio
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.beta = beta
        self.target_kl = target_kl
        # modules
        self.actor = CategoricalActor(dim_obs, num_act)
        self.critic = Critic(dim_obs)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_critic)

    def make_decision(self, obs):
        prob = self.actor(obs)
        act = tf.squeeze(tf.random.categorical(logits=prob, num_samples=1))
        logp_a = tf.squeeze(self.actor.logprob(obs,act))
        val = tf.squeeze(self.critic(obs))

        return act.numpy(), val.numpy(), logp_a.numpy()

    def train(self, data, epochs):
    
        @tf.function
        def categorical_entropy(pmf):
            return tf.math.reduce_sum(-pmf*tf.math.log(pmf), axis=-1)

        @tf.function
        def normal_entropy(stddev):
            return .5*tf.math.log(2.*np.pi*np.e*stddev**2)

        # update actor
        for ep in range(epochs):
            logging.debug("Staring actor training epoch: {}".format(ep+1))
            ep_loss_pi = []
            ep_kld = tf.convert_to_tensor([]) # kl-divergence storage
            ep_ent = tf.convert_to_tensor([]) # entropy storage
            with tf.GradientTape() as tape:
                tape.watch(self.actor.trainable_variables)
                pi, logp = self.actor.logprob(data['obs'], data['act'])
                ratio = tf.math.exp(logp - data['logp']) # pi/old_pi
                clip_adv = tf.math.multiply(tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio), data['adv'])
                approx_kld = data['logp'] - logp
                ent = categorical_entropy(pi)
                obj = tf.math.minimum(ratio*data['adv'], clip_adv) + self.beta*ent
                loss_pi = -tf.math.reduce_mean(obj)
            # gradient descent actor weights
            grads_actor = tape.gradient(loss_pi, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
            ep_loss_pi.append(loss_pi)
            ep_kld = tf.concat([ep_kld, approx_kld], axis=0)
            ep_ent = tf.concat([ep_ent, ent], axis=0)
            # log epoch
            mean_loss_pi = tf.math.reduce_mean(ep_loss_pi)
            mean_ent = tf.math.reduce_mean(ep_ent)
            mean_kld = tf.math.reduce_mean(ep_kld)
            logging.info("\n----Actor Training----\nEpoch :{} \nLoss: {} \nEntropy: {} \nKLDivergence: {}".format(
                ep+1,
                mean_loss_pi,
                mean_ent,
                mean_kld
            ))
            # early cutoff due to large kl-divergence
            if mean_kld > self.target_kl:
                logging.warning("Early stopping at epoch {} due to reaching max kl-divergence.".format(ep+1))
                break
        # update critic
        for ep in range(epochs):
            logging.debug("Starting critic training epoch: {}".format(ep+1))
            ep_loss_val = []
            with tf.GradientTape() as tape:
                tape.watch(self.critic.trainable_variables)
                loss_val = tf.keras.losses.MSE(data['ret'], self.critic(data['obs']))
            # gradient descent critic weights
            grads_critic = tape.gradient(loss_val, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_variables))
            ep_loss_val.append(loss_val)
            # log loss_v
            mean_loss_val = tf.math.reduce_mean(ep_loss_val)
            logging.info("\n----Critic Training----\nEpoch :{} \nLoss: {}".format(
                ep+1,
                mean_loss_val
            ))

        return mean_loss_pi, mean_loss_val, dict(kld=mean_kld, entropy=mean_ent)
        

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
