import sys
import os
import numpy as np
import scipy.signal
import gym
import time
import matplotlib.pyplot as plt
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

class Actor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, **kwargs):
        super(Actor, self).__init__(name='actor', **kwargs)
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.log_std = tf.Variable(initial_value=-0.5*np.ones(act_dim, dtype=np.float32))

    def _distribution(self, obs):
        mu = tf.squeeze(self.mu_net(obs))
        std = tf.math.exp(self.log_std)

        return tfd.Normal(loc=mu, scale=std)

    def _log_prob_from_distribution(self, pi, act):
        return tf.math.reduce_sum(pi.log_prob(act), axis=-1)

    def call(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)

        return pi, logp_a

class Critic(tf.keras.Model):
    def __init__(self, obs_dim, hidden_sizes, activation, **kwargs):
        super(Critic, self).__init__(name='critic', **kwargs)
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def call(self, obs):
        return tf.squeeze(self.v_net(obs), axis=-1)

class ActorCritic(tf.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256,256), activation='relu', lr_actor=3e-4,
                 lr_critic=1e-3, **kwargs):
        super(ActorCritic, self).__init__(name='ac', **kwargs)
        # networks
        self.actor = Actor(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=hidden_sizes, activation=activation)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_critic)
        self.critic = Critic(obs_dim=obs_dim, hidden_sizes=hidden_sizes, activation=activation)

    def step(self, obs):
        pi_dist = self.actor._distribution(obs)
        a = pi_dist.sample()
        logp_a = self.actor._log_prob_from_distribution(pi_dist, a)
        v = self.critic(obs)

        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

    def train(self, data, iters_v):
        # update actor
        logging.debug("Staring actor update")
        with tf.GradientTape() as tape:
            tape.watch(self.actor.trainable_variables)
            pi, logp = self.actor(data['obs'], data['act']) 
            loss_pi = -tf.math.reduce_mean(logp*data['adv'])
            kl = tf.math.reduce_mean(data['logp'] - logp) # approximate kl-divergence
            ent = tf.math.reduce_mean(tf.math.reduce_sum(pi.entropy(), axis=-1))
        # gradient descent actor weights
        grads_actor = tape.gradient(loss_pi, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
        logging.info("\nLoss: {} \nEntropy: {} \nKLDivergence: {}".format(
            loss_pi,
            ent,
            kl
        ))
        # update critic
        for i in range(iters_v):
            logging.debug("Starting critic epoch: {}".format(i))
            with tf.GradientTape() as tape:
                tape.watch(self.critic.trainable_variables)
                loss_v = tf.keras.losses.MSE(data['ret'], self.critic(data['obs']))
            # gradient descent critic weights
            grads_critic = tape.gradient(loss_v, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_variables))
            # log epoch
            logging.info("Critic epoch :{} \nLoss: {}".format(
                i+1,
                loss_v
            ))

        return loss_pi, loss_v, dict(kl=kl, ent=ent) 

################################################################
"""
On-policy Replay Buffer
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

class ACBuffer:
    """
    A buffer for storing trajectories experienced by an on-policy agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
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
    save_freq=20
    # instantiate actor-critic and replay buffer
    obs_dim=env.observation_space.shape[0]
    act_dim=env.action_space.shape[0]
    ac = ActorCritic(obs_dim=obs_dim, act_dim=act_dim)
    replay_buffer = ACBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)
    # Prepare for interaction with environment
    model_dir = './models/ac/'+env.spec.id
    obs, ep_ret, ep_len = env.reset(), 0, 0
    episodes, total_steps = 0, 0
    stepwise_rewards, episodic_returns, sedimentary_returns = [], [], []
    episodic_steps = []
    start_time = time.time()
    # main loop
    for ep in range(epochs):
        for st in range(steps_per_epoch):
            act, val, logp = ac.step(obs.reshape(1,-1))
            next_obs, rew, done, _ = env.step(act)
            ep_ret += rew
            ep_len += 1
            stepwise_rewards.append(rew)
            total_steps += 1
            replay_buffer.store(obs, act, rew, val, logp)
            obs = next_obs # SUPER CRITICAL!!!
            # handle episode termination
            timeout = (ep_len==env.spec.max_episode_steps)
            terminal = done or timeout
            epoch_ended = (st==steps_per_epoch-1)
            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at {} steps.'.format(ep_len), flush=True)
                if timeout or epoch_ended:
                    _, val, _ = ac.step(obs.reshape(1,-1))
                else:
                    val = 0
                replay_buffer.finish_path(val)
                if terminal:
                    episodes += 1
                    episodic_returns.append(ep_ret)
                    sedimentary_returns.append(sum(episodic_returns)/episodes)
                    episodic_steps.append(total_steps)
                    print("\n====\nTotalSteps: {} \nEpisode: {}, Step: {}, EpReturn: {}, EpLength: {} \n====\n".format(total_steps, episodes, st+1, ep_ret, ep_len))
                obs, ep_ret, ep_len = env.reset(), 0, 0
        # update actor-critic
        loss_pi, loss_v, loss_info = ac.train(replay_buffer.get(), train_iters)
        print("\n================================================================\nEpoch: {} \nStep: {} \nAveReturn: {} \nLossPi: {} \nLossV: {} \nKLDivergence: {} \nEntropy: {} \nTimeElapsed: {}\n================================================================\n".format(ep+1, st+1, sedimentary_returns[-1], loss_pi, loss_v, loss_info['kl'], loss_info['ent'], time.time()-start_time))
        # Save model
        if not ep%save_freq or (ep==epochs-1):
            model_path = os.path.join(model_dir, str(ep))
            if not os.path.exists(os.path.dirname(model_path)):
                os.makedirs(os.path.dirname(model_path))
            ac.actor.mu_net.save(model_path)

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
            act = ac.act(obs.reshape(1,-1))
            next_obs, rew, done, info = env.step(act)
            rewards.append(rew)
            # print("\n-\nepisode: {}, step: {} \naction: {} \nobs: {}, \nreward: {}".format(ep+1, st+1, act, obs, rew))
            obs = next_obs.copy()
            if done:
                ep_rets.append(sum(rewards))
                ave_rets.append(sum(ep_rets)/len(ep_rets))
                print("\n---\nepisode: {} \nepisode return: {}, averaged return: {} \n---\n".format(ep+1, ep_rets[-1], ave_rets[-1]))
                break
