# -*- coding: utf-8 -*-
#
# Copyright 2020 Marco Favorito
#
# ------------------------------
#
# This file is part of yarllib.
#
# yarllib is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# yarllib is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with yarllib.  If not, see <https://www.gnu.org/licenses/>.
#

"""This module implements Dynamic Programming algorithms (e.g. Value Iteration, Policy Iteration etc.)."""
import logging
from abc import abstractmethod
from typing import List, Optional

import gym
import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv
from gym.spaces import Discrete

from yarllib.base import AbstractAgent
from yarllib.core import Policy
from yarllib.helpers.base import get_machine_epsilon, set_env_seed, set_seed
from yarllib.helpers.history import AgentObs, History
from yarllib.policies import GreedyPolicy

logger = logging.getLogger(__name__)


class GPIAgent(AbstractAgent):
    """A Generalized Policy Iteration agent."""

    def __init__(
        self, observation_space: Discrete, action_space: Discrete, gamma: float = 0.9
    ):
        """Initialize a GPI agent."""
        self.observation_space = observation_space
        self.action_space = action_space
        self.nS = self.observation_space.n
        self.nA = self.action_space.n
        self.gamma = gamma

    def train(self, env: DiscreteEnv, *args, max_nb_iterations: int = 50, **kwargs):
        """
        Train the agent.

        :param env: the environment to train on.
        :param max_nb_iterations: the number of iterations.
        :return:
        """
        _i = 0
        for _i in range(max_nb_iterations):
            if self.can_stop():
                break
            self.evaluation(env)
            self.improvement(env)
        logger.debug("Training number of iterations: %s", _i)

    def test(
        self,
        env: gym.Env,
        policy: Optional[Policy] = None,
        nb_episodes: int = 10,
        seed: Optional[int] = None,
        experiment_name: str = "",
        **_kwargs
    ) -> History:
        """Test the agent."""
        if policy is None:
            policy = GreedyPolicy()
        policy.action_space = env.action_space
        policy.model = self

        set_seed(seed)
        set_env_seed(seed, env)

        history: List[List[AgentObs]] = []
        current_episode: List[AgentObs] = []
        for _ in range(nb_episodes):
            done = False
            s = env.reset()
            while not done:
                a = policy.get_action(s)
                sp, r, done, info = env.step(a)
                current_episode.append((s, a, r, sp))
                s = sp
            history.append(current_episode)
            current_episode = []
        return History(history, is_training=False, seed=seed, name=experiment_name)

    @abstractmethod
    def evaluation(self, env: DiscreteEnv):
        """Do the evaluation step."""

    @abstractmethod
    def improvement(self, env: DiscreteEnv):
        """Do the improvement step."""

    def can_stop(self) -> bool:
        """Stop the GPI loop."""
        return False


class PolicyIterationAgent(GPIAgent):
    """The Policy Iteration algorithm."""

    def __init__(self, *args, eps: float = 1e-10, **kwargs):
        """
        Initialize the algorithm.

        :param eps: the accuracy of the estimation.
        """
        super().__init__(*args, **kwargs)

        self.eps = eps
        self.v = np.random.rand(self.nS) * get_machine_epsilon()
        self.pi = np.random.randint(0, self.nA, self.nS)
        self.policy_stable = False

    def evaluation(self, env: DiscreteEnv):
        """Evaluate current policy."""
        delta = np.inf
        while not delta < self.eps:
            delta = 0
            for s in range(len(self.v)):
                v = self.v[s]
                a = self.pi[s]
                new_v = self._get_next_value(env, s, a)
                self.v[s] = new_v
                delta = max(delta, abs(v - new_v))

    def _get_next_value(self, env, state, action):
        """Get the next value, given state and action."""
        return sum(
            [
                p * (r + self.gamma * self.v[sp])
                for (p, sp, r, _done) in env.P[state][action]
            ]
        )

    def improvement(self, env: DiscreteEnv):
        """Improve current policy."""
        self.policy_stable = True
        for s in range(len(self.v)):
            old_action = self.pi[s]
            action_values = [self._get_next_value(env, s, a) for a in range(self.nA)]
            new_action = np.argmax(action_values)
            self.pi[s] = new_action
            if old_action != new_action:
                self.policy_stable = False

    def can_stop(self) -> bool:
        """Decide if we can stop the main GPI loop."""
        return self.policy_stable

    def get_best_action(self, state):
        """Get the best action from a state."""
        return self.pi[state]


class ValueIterationAgent(GPIAgent):
    """
    The Value Iteration algorithm.

    differently from the abstract class, the max_nb_iterations
    parameter to the 'train' method is ignored.
    """

    def __init__(self, *args, eps: float = 1e-10, **kwargs):
        """
        Initialize the algorithm.

        :param eps: the accuracy of the estimation.
        """
        super().__init__(*args, **kwargs)

        self.eps = eps
        self.v = np.zeros(self.nS)
        self.pi = np.random.randint(0, self.nA, self.nS)
        self.policy_stable = False

    def evaluation(self, env: DiscreteEnv):
        """Evaluate current policy."""
        delta = np.inf
        while not delta < self.eps:
            delta = 0
            for s in range(len(self.v)):
                v = self.v[s]
                new_v = np.max(self._get_next_values(env, s))
                self.v[s] = new_v
                delta = max(delta, abs(v - new_v))

    def _get_next_values(self, env, state):
        """Get the next value, given state and action."""
        return [
            sum(
                [
                    p * (r + self.gamma * self.v[sp])
                    for (p, sp, r, _done) in env.P[state][action]
                ]
            )
            for action in range(self.nA)
        ]

    def improvement(self, env: DiscreteEnv):
        """Improve current policy."""
        self.policy_stable = True
        for s in range(len(self.v)):
            action_values = self._get_next_values(env, s)
            new_action = np.argmax(action_values)
            self.pi[s] = new_action

    def can_stop(self) -> bool:
        """Decide if we can stop the main GPI loop."""
        return self.policy_stable

    def get_best_action(self, state):
        """Get the best action from a state."""
        if self.observation_space.contains(state):
            return self.pi[state]
        else:
            return self.action_space.sample()
