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

    def __init__(self, dim_obs, num_act, key):
        self.dim_obs = dim_obs
        self.num_act = num_act
        self.policy_net = build_network(num_outputs=num_act)
        self.init_policy_net(key)

    def init_policy_net(self, key):
        sample_input = jax.random.normal(key, shape=(self.dim_obs, ))
        sample_input = jnp.expand_dims(sample_input, 0)
        self.weights = self.policy_net.init(key, sample_input)

    def gen_policy(self, obs):
        """
        pi(a|s)
        """
        logits = jnp.squeeze(self.policy_net.apply(self.weights, obs))
        policy = distrax.Categorical(logits=logits)

        return policy

    def log_prob(self, policy, action):
        logp_a = policy.log_prob(action)

        return logp_a

    def __call__(self, obs, act=None):
        policy = self.gen_policy(obs)
        logp_a = None
        if act is not None:
            logp_a = self.log_prob(policy, act)

        return policy, logp_a

# class PPOAgent:
#     """A simple PPO agent. Compatible with discrete gym envs"""
#
#     def __init__(self, observation_space, action_space, learning_rate):
#         self.observation_space = observation_space
#         self.action_space = action_space
#         # Neural net and optimiser.
#         self.actor = CategoricalActor(
#             dim_obs=observation_space.shape[0],
#             num_act=action_space.n,
#         )
#         self.critic = build_network(num_outputs=1)
#         self.actor_optimizer = optax.adam(learning_rate)
#         self.critic = optax.adam(learning_rate)
#         # self._epsilon_by_frame = optax.polynomial_schedule(**epsilon_cfg)
#         # Jitting for speed.
#         self.make_decision = jax.jit(self.make_decision)
#         self.learn_step = jax.jit(self.learn_step)
#
#     def make_decision(self, obs):
#         pass
#

