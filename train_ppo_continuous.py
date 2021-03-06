import sys
import os
from copy import deepcopy
import numpy as np
import scipy.signal
import random
import time
import gym
import matplotlib.pyplot as plt
from datetime import datetime
import logging

import tensorflow as tf
print(tf.__version__)
import tensorflow_probability as tfp
tfd = tfp.distributions
################################################################
"""
Unnecessary initial settings
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
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
# set log level
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)
################################################################



################################################################
"""
Build actor_net, critic_net
"""
def mlp(sizes, activation, output_activation=None):
    inputs = tf.keras.Input(shape=(sizes[0],))
    x = tf.keras.layers.Dense(sizes[1], activation=activation)(inputs)
    for i in range(2,len(sizes)-1):
        x = tf.keras.layers.Dense(sizes[i], activation=activation)(x)
    outputs = tf.keras.layers.Dense(sizes[-1], activation=output_activation)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

class GaussianActor(tf.keras.Model):

    def __init__(self, dim_obs, dim_act, hidden_sizes, activation, **kwargs):
        super(GaussianActor, self).__init__(name='actor', **kwargs)
        self.log_std = tf.Variable(initial_value=-0.5*np.ones(dim_act, dtype=np.float32))
        self.mean_net = mlp([dim_obs] + list(hidden_sizes) + [dim_act], activation)

    @tf.function
    def _distribution(self, obs):
        mu = tf.squeeze(self.mean_net(obs))
        log_sigma = tf.squeeze(self.log_std)

        return mu, log_sigma

    @tf.function
    def call(self, obs, act=None):
        mu, log_sigma = self._distribution(obs)
        logp_a = None
        def log_normal_pdf(sample, mean, log_std):
            log2pi = tf.math.log(2.*np.pi)
            return tf.math.reduce_sum(-.5*(((sample-mean)*tf.math.exp(-log_std))**2 + 2*log_std + log2pi), axis=-1)
        if act is not None:
            #logp_a = self._log_prob_from_distribution(pi, act)
            logp_a = log_normal_pdf(sample=act, mean=mu, log_std=log_sigma)

        return logp_a

class Critic(tf.keras.Model):
    def __init__(self, dim_obs, hidden_sizes, activation, **kwargs):
        super(Critic, self).__init__(name='critic', **kwargs)
        self.val_net = mlp([dim_obs] + list(hidden_sizes) + [1], activation)

    @tf.function
    def call(self, obs):
        return tf.squeeze(self.val_net(obs), axis=-1)

class PPOAgent(tf.Module):
    def __init__(self, dim_obs, dim_act, hidden_sizes=(256,256), activation='relu', clip_ratio=0.2, lr_actor=3e-4,
                 lr_critic=1e-3, beta=0., target_kl=0.01, **kwargs):
        super(PPOAgent, self).__init__(name='ppo', **kwargs)
        # params
        self.clip_ratio = clip_ratio
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.beta = beta
        self.target_kl = target_kl
        # networks
        self.actor = GaussianActor(dim_obs=dim_obs, dim_act=dim_act, hidden_sizes=hidden_sizes, activation=activation)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_critic)
        self.critic = Critic(dim_obs=dim_obs, hidden_sizes=hidden_sizes, activation=activation)

    def pi_of_a_given_s(self, obs):
        with tf.GradientTape() as t:
            with t.stop_recording():
                mu, log_sigma = self.actor._distribution(obs)
                act = tf.squeeze(tf.random.normal(shape=mu.shape, mean=mu, stddev=tf.math.exp(log_sigma)))
                logp_a = self.actor(obs, act) 
                val = self.critic(obs)

        return act.numpy(), val.numpy(), logp_a.numpy()

    def train(self, data, num_epochs):

        def normal_entropy(log_std):
            return .5*tf.math.log(2.*np.pi*np.e*tf.math.exp(log_std)**2)

        # update actor
        for epch in range(num_epochs):
            logging.debug("Staring actor epoch: {}".format(epch+1))
            ep_kl = tf.convert_to_tensor([]) 
            ep_ent = tf.convert_to_tensor([]) 
            with tf.GradientTape() as tape:
                tape.watch(self.actor.trainable_variables)
                logp = self.actor(data['obs'], data['act']) 
                ratio = tf.math.exp(logp - data['logp']) # pi/old_pi
                clip_adv = tf.math.multiply(tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio), data['adv'])
                approx_kl = tf.reshape(data['logp'] - logp, shape=[-1])
                ent = tf.reshape(tf.math.reduce_sum(normal_entropy(self.actor.log_std), axis=-1), shape=[-1])
                obj = tf.math.minimum(tf.math.multiply(ratio, data['adv']), clip_adv) + self.beta*ent
                loss_pi = -tf.math.reduce_mean(obj)
            # gradient descent actor weights
            grads_actor = tape.gradient(loss_pi, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
            # record kl-divergence and entropy
            ep_kl = tf.concat([ep_kl, approx_kl], axis=0)
            ep_ent = tf.concat([ep_ent, ent], axis=0)
            # log epoch
            kl = tf.math.reduce_mean(ep_kl)
            entropy = tf.math.reduce_mean(ep_ent)
            logging.info("Epoch :{} \nLoss: {} \nEntropy: {} \nKLDivergence: {}".format(
                epch+1,
                loss_pi,
                entropy,
                kl
            ))
            # early cutoff due to large kl-divergence
            # if kl > 1.5*self.target_kl:
            #     logging.warning("Early stopping at epoch {} due to reaching max kl-divergence.".format(epch+1))
            #     break
        # update critic
        for epch in range(num_epochs):
            logging.debug("Starting critic epoch: {}".format(epch))
            with tf.GradientTape() as tape:
                tape.watch(self.critic.trainable_variables)
                loss_v = tf.keras.losses.MSE(data['ret'], self.critic(data['obs']))
            # gradient descent critic weights
            grads_critic = tape.gradient(loss_v, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_variables))
            # log epoch
            logging.info("Epoch :{} \nLoss: {}".format(
                epch+1,
                loss_v
            ))

        return loss_pi, loss_v, dict(kl=kl, ent=entropy) 
################################################################


################################################################
"""
On-policy Replay Buffer for PPO
"""
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, dim_obs, dim_act, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, dim_obs), dtype=np.float32)
        self.act_buf = np.zeros((size, dim_act), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr <= self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr
        # self.ptr, self.path_start_idx = 0, 0

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: tf.convert_to_tensor(v, dtype=tf.float32) for k,v in data.items()}
################################################################


################################################################
"""
Main
"""
RANDOM_SEED = 0
if __name__=='__main__':
    # instantiate env
    env = gym.make('LunarLanderContinuous-v2')
    # set seed
    tf.random.set_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)
    # paramas
    steps_per_epoch=5000
    epochs=200
    gamma=0.99
    train_iters=80
    lam=0.97
    max_ep_len=1000
    save_freq=10
    # instantiate actor-critic and replay buffer
    dim_obs=env.observation_space.shape[0]
    dim_act=env.action_space.shape[0]
    agent = PPOAgent(dim_obs=dim_obs, dim_act=dim_act, beta=0.)
    replay_buffer = PPOBuffer(dim_obs, dim_act, steps_per_epoch, gamma, lam)
    # Prepare for interaction with environment
    model_dir = './models/ppo/'+env.spec.id
    obs, ep_ret, ep_len = env.reset(), 0, 0
    episodes, total_steps = 0, 0
    stepwise_rewards, episodic_returns, sedimentary_returns = [], [], []
    episodic_steps = []
    start_time = time.time()
    # main loop
    for ep in range(epochs):
        for st in range(steps_per_epoch):
            act, val, logp = agent.pi_of_a_given_s(np.expand_dims(obs, 0))
            next_obs, rew, done, _ = env.step(act)
            ep_ret += rew
            ep_len += 1
            stepwise_rewards.append(rew)
            total_steps += 1
            replay_buffer.store(obs, act, rew, np.squeeze(val), logp)
            obs = next_obs # SUPER CRITICAL!!!
            # handle episode termination
            timeout = (ep_len==env.spec.max_episode_steps)
            terminal = done or timeout
            epoch_ended = (st==steps_per_epoch-1)
            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at {} steps.'.format(ep_len), flush=True)
                if timeout or epoch_ended:
                    _, val, _ = agent.pi_of_a_given_s(np.expand_dims(obs,0))
                else:
                    val = [0]
                replay_buffer.finish_path(np.squeeze(val))
                if terminal:
                    episodes += 1
                    episodic_returns.append(ep_ret)
                    sedimentary_returns.append(sum(episodic_returns)/episodes)
                    episodic_steps.append(total_steps)
                    print("\n====\nTotalSteps: {} \nEpisode: {}, Step: {}, EpReturn: {}, EpLength: {} \n====\n".format(total_steps, episodes, st+1, ep_ret, ep_len))
                obs, ep_ret, ep_len = env.reset(), 0, 0
        # Save model
        if not ep%save_freq or (ep==epochs-1):
            mean_path = os.path.join(model_dir, 'actor', str(ep))
            if not os.path.exists(os.path.dirname(mean_path)):
                os.makedirs(os.path.dirname(mean_path))
            val_path = os.path.join(model_dir, 'critic', str(ep))
            if not os.path.exists(os.path.dirname(val_path)):
                os.makedirs(os.path.dirname(val_path))
            agent.actor.mean_net.save(mean_path)
            np.save(model_dir+'/actor/'+'log_std.npy', agent.actor.log_std.numpy())
            agent.critic.val_net.save(val_path)

        # update actor-critic
        loss_pi, loss_v, loss_info = agent.train(replay_buffer.get(), train_iters)
        print("\n================================================================\nEpoch: {} \nStep: {} \nAveReturn: {} \nLossPi: {} \nLossV: {} \nKLDivergence: {} \nEntropy: {} \nTimeElapsed: {}\n================================================================\n".format(ep+1, st+1, sedimentary_returns[-1], loss_pi, loss_v, loss_info['kl'], loss_info['ent'], time.time()-start_time))
################################################################

    # Save returns 
    np.save(os.path.join(model_dir, 'episodic_returns.npy'), episodic_returns)
    np.save(os.path.join(model_dir, 'sedimentary_returns.npy'), sedimentary_returns)
    np.save(os.path.join(model_dir, 'episodic_steps.npy'), episodic_steps)
    with open(os.path.join(model_dir, 'training_time.txt'), 'w') as f:
        f.write("{}".format(time.time()-start_time))
    # plot returns
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Averaged Returns')
    ax.plot(sedimentary_returns)
    plt.show()


    # Test trained model
    input("Press ENTER to test lander...")
    num_episodes = 10
    num_steps = env.spec.max_episode_steps
    ep_rets, ave_rets = [], []
    for ep in range(num_episodes):
        obs, done, rewards = env.reset(), False, []
        for st in range(num_steps):
            env.render()
            act, _, _ = agent.pi_of_a_given_s(np.expand_dims(obs, 0))
            next_obs, rew, done, info = env.step(act)
            rewards.append(rew)
            # print("\n-\nepisode: {}, step: {} \naction: {} \nobs: {}, \nreward: {}".format(ep+1, st+1, act, obs, rew))
            obs = next_obs.copy()
            if done:
                ep_rets.append(sum(rewards))
                ave_rets.append(sum(ep_rets)/len(ep_rets))
                print("\n---\nepisode: {} \nepisode return: {}, averaged return: {} \n---\n".format(ep+1, ep_rets[-1], ave_rets[-1]))
                break



# Debug
# env = gym.make('LunarLanderContinuous-v2')
# o, d = env.reset(), False
# agent = PPOAgent(dim_obs=8, dim_act=2)
# while not d:
#     # env.render()
#     a, v, l = agent.pi_of_a_given_s(obs=np.expand_dims(o,0)) 
#     print(a, v, l)
#     o,r,d,i = env.step(np.squeeze(a))
    
