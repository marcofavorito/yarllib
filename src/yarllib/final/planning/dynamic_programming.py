from typing import Any, Iterator, cast

import gym
import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv

from yarllib.final.base import Agent
from yarllib.final.gym import iter_space, space_size


class QValueIteration(Agent):
    """Q-value iteration."""

    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma

    def done(self):
        """
        Return false, as this type of agent is never "done".

        :return: always False.
        """
        return False

    def choose_best_action(self, state):
        return self.Q[state, :].argmax()

    def setup(self, env: gym.Env):
        super().setup(env)
        nb_states = space_size(env.observation_space)
        nb_actions = space_size(env.action_space)
        self.Q = np.zeros((nb_states, nb_actions))

    def explore(self) -> Iterator:
        for s in iter_space(self.env.observation_space):
            for a in iter_space(self.env.action_space):
                yield s, a

    def make_trial(self, state, action):
        env = cast(DiscreteEnv, self.env)
        distribution = env.P[state][action]
        return state, action, distribution

    def backup(self, trial):
        state, action, distribution = trial
        env = cast(DiscreteEnv, self.env)
        result = 0.0
        for (p, sp, r, done) in env.P[state][action]:
            result += p * (r + 0.99 * self.Q[sp, :].max())
        return result

    def loss(self, state, action, estimate: float):
        """Compute the loss."""
        return estimate

    def update(self, state, action, loss):
        """Update the loss."""
        self.Q[state, action] = loss
