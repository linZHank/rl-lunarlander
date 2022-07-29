"""
A simple PPO agent
"""
import collections
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


class CategoricalActor:

    def __init__(self, dim_obs: int, num_act: int, key: jnp.DeviceArray):
        self.dim_obs = dim_obs
        self.num_act = num_act
        self.policy_net = build_network(num_outputs=num_act)
        self.weights = None
        self.policy_distribution = None
        self.init_policy_net(key)

    def init_policy_net(self, key: jnp.DeviceArray):
        sample_input = jax.random.normal(key, shape=(self.dim_obs, ))
        # sample_input = jnp.expand_dims(sample_input, 0)
        self.weights = self.policy_net.init(key, sample_input)

    def gen_policy(self, obs):
        """
        pi(a|s)
        """
        logits = jnp.squeeze(self.policy_net.apply(self.weights, obs))
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

class Critic:
    def __init__(self, dim_obs, key):
        self.dim_obs = dim_obs
        self.value_net = build_network(num_outputs=1)
        self.weights = None
        self.init_value_net(key)

    def init_value_net(self, key):
        sample_input = jax.random.normal(key, shape=(self.dim_obs, ))
        # sample_input = jnp.expand_dims(sample_input, 0)
        self.weights = self.value_net.init(key, sample_input)

    def __call__(self, obs):

        return self.value_net.apply(self.weights, obs)

class PPOAgent:
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
        # self._epsilon_by_frame = optax.polynomial_schedule(**epsilon_cfg)
        # Jitting for speed.
        self.make_decision = jax.jit(self.make_decision)
        self.learn_step = jax.jit(self.learn_step)

    def make_decision(self, key, obs):
        self.actor.gen_policy(obs)
        act = self.actor.policy_distribution.sample(seed=key)
        logp_a = self.actor.log_prob(act)
        val = self.critic(obs)

        return act, val, logp_a

    def learn_step(self, key):
        pass

# test
import gym
env = gym.make("LunarLander-v2")
rng = hk.PRNGSequence(jax.random.PRNGKey(123))
agent = PPOAgent(
    observation_space=env.observation_space,
    action_space=env.action_space,
    learning_rate=3e-4,
    key = next(rng),
)
# buf = ReplayBuffer(capacity=int(1e6))
pobs = env.reset()
done = False
ep, ep_return = 0, 0
for st in range(int(3e3)):
    env.render()
    act, _, _ = agent.make_decision(
        key=next(rng),
        obs=pobs,
    )
    act = int(act)
    nobs, rew, done, info = env.step(act)
    # accumulate experience
    # buf.store(pobs, act, rew, done, nobs)
    ep_return += rew
    pobs = nobs
    # learn
    # if buf.is_ready(batch_size=1024):
    #     params, learner_state = agent.learn_step(
    #         params, buf.sample(batch_size=1024, discount_factor=0.99), learner_state, next(rng)
    #     )
    if done:
        print(f"episode {ep+1} return: {ep_return}")
        print(f"total steps: {st+1}")
        pobs = env.reset()
        done = False
        ep_return = 0
        ep += 1


