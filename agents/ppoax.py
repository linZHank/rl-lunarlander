"""
A simple PPO agent
"""
import numpy as np
import collections
import scipy.signal
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import rlax
import distrax


Weights = collections.namedtuple("Weights", "actor, critic")
def build_network(num_outputs: int) -> hk.Transformed:
    """Factory for a simple MLP network (for approximating Q-values)."""

    def net(inputs):
        mlp = hk.nets.MLP([128, 128, num_outputs])

        return mlp(inputs)

    return hk.without_apply_rng(hk.transform(net))


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x, [x0, x1, x2]
    output:
        [x0+discount*x1+discount^2*x2, x1+discount*x2, x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class OnPolicyReplayBuffer(object):
    """A simple off-policy replay buffer."""

    def __init__(
        self, 
        dim_obs: int, 
        dim_act: int, 
        capacity: int,
        gamma=0.99,
        lmbd=0.97,
    ):

        # params
        self.dim_obs=dim_obs
        self.dim_act=dim_act
        self.gamma=gamma
        self.lmbd=lmbd
        self.capacity=capacity
        # buffers
        self.obs_buf = np.zeros((capacity, dim_obs), dtype=np.float32)
        self.act_buf = np.squeeze(np.zeros((capacity, dim_act), dtype=np.float32)) # squeeze in case dim_act=1
        self.adv_buf = np.zeros(capacity, dtype=np.float32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.ret_buf = np.zeros(capacity, dtype=np.float32)
        self.val_buf = np.zeros(capacity, dtype=np.float32)
        self.lpa_buf = np.zeros(capacity, dtype=np.float32)
        # vars
        self.ptr, self.traj_start_id = 0, 0

    def store(self, observation, action, reward, value, log_prob):
        assert self.ptr <= self.capacity
        self.obs_buf[self.ptr] = observation
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.val_buf[self.ptr] = value
        self.lpa_buf[self.ptr] = log_prob
        self.ptr += 1

    def finish_traj(self, last_val=0):
        """
        Call this function at the end of a trajectory.
        Compute advantage estimates with GAE-Lambda, and the rewards-to-go.
        "last_val" argument should be 0 if episode done, otherwise, V(s_T).
        """
        
        traj_slice = slice(self.traj_start_id, self.ptr)
        rew_traj = jnp.append(self.rew_buf[traj_slice], last_val)
        val_traj = jnp.append(self.val_buf[traj_slice], last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        adv_traj = rew_traj[:-1] + self.gamma * val_traj[1:] - val_traj[:-1]
        self.adv_buf[traj_slice] = discount_cumsum(
            adv_traj,
            self.gamma*self.lmbd,
        )
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[traj_slice] = discount_cumsum(rew_traj, self.gamma)[:-1]
        # set new trajectory starting point 
        self.traj_start_id = self.ptr

    def get(self):
        """
        Call this function to get all the buffered experience with normalize advantage.
        """
        assert self.ptr <= self.capacity
        self.obs_buf = self.obs_buf[:self.ptr]
        self.act_buf = self.act_buf[:self.ptr]
        self.rew_buf = self.rew_buf[:self.ptr]
        self.val_buf = self.val_buf[:self.ptr]
        self.ret_buf = self.ret_buf[:self.ptr]
        self.adv_buf = self.adv_buf[:self.ptr]
        self.lpa_buf = self.lpa_buf[:self.ptr]
        # the next three lines implement the advantage normalization trick
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.clip(np.std(self.adv_buf), 1e-10, None)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            lpa=self.lpa_buf,
        )
        # reset buffer
        self.__init__(self.dim_obs, self.dim_act, self.capacity) # On-Policy
        return {k: jnp.asarray(v) for k,v in data.items()}


class CategoricalActor(object):

    def __init__(self, dim_obs: int, num_act: int, key: jnp.DeviceArray):
        self.dim_obs = dim_obs
        self.num_act = num_act
        self.policy_net = build_network(num_outputs=num_act)
        self.params = None
        self.policy_distribution = None
        self.init_policy_net(key)

    def init_policy_net(self, key: jnp.DeviceArray):
        sample_input = jax.random.normal(key, shape=(self.dim_obs, ))
        # sample_input = jnp.expand_dims(sample_input, 0)
        self.params = self.policy_net.init(key, sample_input)

    def gen_policy(self, obs):
        """
        pi(a|s)
        """
        logits = jnp.squeeze(self.policy_net.apply(self.params, obs))
        self.policy_distribution = distrax.Categorical(logits=logits)

        # return self.policy

    def log_prob(self, action):
        logp_a = self.policy_distribution.log_prob(action)

        return logp_a

    def __call__(self, obs, act=None):
        self.gen_policy(obs)
        logp_a = None
        if act is not None:
            logp_a = self.log_prob(act)

        return self.policy_distribution, logp_a


class Critic(object):
    def __init__(self, dim_obs, key):
        self.dim_obs = dim_obs
        self.value_net = build_network(num_outputs=1)
        self.params = None
        self.init_value_net(key)

    def init_value_net(self, key):
        sample_input = jax.random.normal(key, shape=(self.dim_obs, ))
        # sample_input = jnp.expand_dims(sample_input, 0)
        self.params = self.value_net.init(key, sample_input)

    def __call__(self, obs):

        return jnp.squeeze(self.value_net.apply(self.params, obs))


class PPOAgent(object):
    """A simple PPO agent. Compatible with discrete gym envs"""

    def __init__(self, key, observation_space, action_space, learning_rate):
        self.observation_space = observation_space
        self.action_space = action_space
        # Neural net and optimiser.
        self.actor = CategoricalActor(
            dim_obs=observation_space.shape[0],
            num_act=action_space.n,
            key=key,
        )
        self.critic = Critic(
            dim_obs=observation_space.shape[0],
            key=key,
        )
        self.actor_optimizer = optax.adam(learning_rate)
        self.critic_optimizer = optax.adam(learning_rate)
        self.actropt_state = self.actor_optimizer.init(self.actor.params)
        self.critopt_state = self.critic_optimizer.init(self.critic.params)
        # Jitting for speed.
        self.make_decision = jax.jit(self.make_decision)
        self.update_actor = jax.jit(self.update_actor)

    def make_decision(self, key, obs):
        self.actor.gen_policy(obs)
        act = self.actor.policy_distribution.sample(seed=key)
        logp_a = self.actor.log_prob(act)
        val = self.critic(obs)

        return act, val, logp_a

    def update_actor(self, data):
        dloss, grads = jax.grad(self.actor_loss_fn)(data)
        updates, self.actropt_state = self.actor_optimizer.update(
            dloss, self.actropt_state
        )
        self.actor.params = optax.apply_updates(self.actor.params, updates)
    
    def actor_loss_fn(self, data):
        pi, lpa = self.actor(data['obs'], data['act'])
        ratio = jnp.exp(lpa - data['lpa'])
        batched_adv = data['adv']
        # batched_loss = jax.vmap(rlax.clipped_surrogate_pg_loss)
        neg_obj = rlax.clipped_surrogate_pg_loss(
            prob_ratios_t=ratio,
            adv_t=batched_adv,
            epsilon=0.2,
        )

        return jnp.mean(rlax.l2_loss(neg_obj))


# test
import gym
env = gym.make("LunarLander-v2")
rng = hk.PRNGSequence(jax.random.PRNGKey(123))
agent = PPOAgent(
    observation_space=env.observation_space,
    action_space=env.action_space,
    learning_rate=3e-4,
    key=next(rng),
)
buf = OnPolicyReplayBuffer(dim_obs=8, dim_act=1, capacity=int(1e3))
pobs = env.reset()
done = False
ep, ep_return = 0, 0
for st in range(int(3e3)):
    # env.render()
    act, val, lpa = agent.make_decision(
        key=next(rng),
        obs=pobs,
    )
    act = int(act)
    nobs, rew, done, info = env.step(act)
    # accumulate experience
    buf.store(pobs, act, rew, val, lpa)
    ep_return += rew
    pobs = nobs
    # learn
    # if buf.is_ready(batch_size=1024):
    #     params, learner_state = agent.learn_step(
    #         params, buf.sample(batch_size=1024, discount_factor=0.99), learner_state, next(rng)
    #     )
    if done:
        buf.finish_traj()
        if buf.ptr > 500:
            data = buf.get()
        print(f"episode {ep+1} return: {ep_return}")
        print(f"total steps: {st+1}")
        pobs = env.reset()
        done = False
        ep_return = 0
        ep += 1


