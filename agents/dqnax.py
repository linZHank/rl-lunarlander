"""
A simple double-DQN agent. Supported by JAX.
"""

import collections
import random
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import rlax

Params = collections.namedtuple("Params", "online, target")
ActorState = collections.namedtuple("ActorState", "count")
ActorOutput = collections.namedtuple("ActorOutput", "action, q_value")
LearnerState = collections.namedtuple("LearnerState", "count, opt_state")


def build_network(num_outputs: int) -> hk.Transformed:
    """Factory for a simple MLP network (for approximating Q-values)."""

    def net(inputs):
        mlp = hk.nets.MLP([128, 128, num_outputs])

        return mlp(inputs)

    return hk.without_apply_rng(hk.transform(net))


class ReplayBuffer(object):
    """A simple off-policy replay buffer."""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def store(self, prev_obs, action, reward, done_flag, next_obs):
        if action is not None:
            self.buffer.append(
                (
                    prev_obs,
                    action,
                    reward,
                    done_flag,
                    next_obs,
                )
            )

    def sample(self, batch_size, discount_factor):
        pobs, acts, rews, dnfs, nobs = zip(*random.sample(self.buffer, batch_size))
        return (
            np.stack(pobs),
            np.asarray(acts),
            np.asarray(rews),
            (1 - np.asarray(dnfs)) * discount_factor,
            np.stack(nobs),
        )

    def is_ready(self, batch_size):
        return batch_size <= len(self.buffer)


class DQNAgent:
    """A simple DQN agent. Compatible with gym"""

    def __init__(self, observation_space, action_space, target_period, learning_rate):
        self._observation_space = observation_space
        self._action_space = action_space
        self._target_period = target_period
        # Neural net and optimiser. 
        self._critic_net = build_network(action_space.n)
        self._optimizer = optax.adam(learning_rate)
        # self._epsilon_by_frame = optax.polynomial_schedule(**epsilon_cfg)
        self._epsilon_by_frame = optax.polynomial_schedule(
            init_value=1.0,
            end_value=0.01,
            power=1,
            transition_steps=500,
        )
        # Jitting for speed.
        self.make_decision = jax.jit(self.make_decision)
        self.learn_step = jax.jit(self.learn_step)

    def init_params(self, key):
        sample_input = self._observation_space.sample()
        sample_input = jnp.expand_dims(sample_input, 0)
        online_params = self._critic_net.init(key, sample_input)
        return Params(online_params, online_params)

    def init_actor(self):
        actor_count = jnp.zeros((), dtype=jnp.float32)
        return ActorState(actor_count)

    def init_learner(self, params):
        learner_count = jnp.zeros((), dtype=jnp.float32)
        opt_state = self._optimizer.init(params.online)
        return LearnerState(learner_count, opt_state)

    def make_decision(self, params, obs, episode_count, key, eval_flag):
        obs = jnp.expand_dims(obs, 0)  # add dummy batch
        qval = jnp.squeeze(self._critic_net.apply(params.online, obs))
        epsilon = self._epsilon_by_frame(episode_count)
        act_epsgreedy = rlax.epsilon_greedy(epsilon).sample(key, qval)
        act_greedy = rlax.greedy().sample(key, qval)
        action = jax.lax.select(eval_flag, act_greedy, act_epsgreedy)
        return ActorOutput(action=action, q_value=qval)

    def learn_step(self, params, data, learner_state, unused_key):
        target_params = optax.periodic_update(
            params.online, params.target, learner_state.count, self._target_period
        )
        dloss_dtheta = jax.grad(self._loss)(params.online, target_params, *data)
        updates, opt_state = self._optimizer.update(
            dloss_dtheta, learner_state.opt_state
        )
        online_params = optax.apply_updates(params.online, updates)
        return (
            Params(online_params, target_params),
            LearnerState(learner_state.count + 1, opt_state),
        )

    def _loss(
        self,
        online_params,
        target_params,
        pobs_batch,
        acts_batch,
        rews_batch,
        disc_batch,
        nobs_batch,
    ):
        pred_qval = self._critic_net.apply(online_params, pobs_batch)
        next_qval = self._critic_net.apply(target_params, nobs_batch)
        deul_qval = self._critic_net.apply(online_params, nobs_batch)
        batched_loss = jax.vmap(rlax.double_q_learning)
        td_error = batched_loss(
            pred_qval, acts_batch, rews_batch, disc_batch, next_qval, deul_qval
        )
        return jnp.mean(rlax.l2_loss(td_error))
