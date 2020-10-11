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

"""Test the multi-agent model of yarllib."""
from typing import Any, Optional, Tuple, Type
from unittest import mock
from unittest.mock import MagicMock

import gym
import pytest
from gym.spaces import Discrete
from gym.spaces import Tuple as GymTuple

from yarllib.marl import IndependentLearners, MultiAgentPolicy, SharedStateLearners
from yarllib.policies import GreedyPolicy

_HIGH_NB_AGENTS = 10


class MultiAgentTestEnv(gym.Env):
    """A multi-agent OpenAI Gym environment for testing purposes."""

    _DEFAULT_STATE = 0

    def __init__(self, nb_agents: int):
        """Instantiate a simple Gym environment."""
        super().__init__()

        self.nb_agents = nb_agents
        self.observation_space = GymTuple((Discrete(1),) * nb_agents)
        self.action_space = GymTuple((Discrete(2),) * nb_agents)

    def step(self, action: Tuple):
        """Do a step."""
        win = all(a == 1 for a in action)
        return (
            tuple([self._DEFAULT_STATE] * self.nb_agents),
            1.0 if win else 0.0,
            True,
            {},
        )

    def reset(self) -> Tuple[int, ...]:
        """Reset the state."""
        return tuple([self._DEFAULT_STATE] * self.nb_agents)

    def render(self, mode="human"):
        """Do nothing."""


class BaseTestMultiAgentModel:
    """Base class for testing multi-agent models."""

    nb_agents: int
    multiagent_model_class: Optional[Type] = None
    expected_state: Any = None

    @classmethod
    def setup_class(cls):
        """Set up the class."""
        if cls == BaseTestMultiAgentModel:
            pytest.skip("Base test class - skip.")

    def setup(self):
        """Set up the test."""
        self.models = [self.make_model() for _ in range(self.nb_agents)]
        self.policies = [self.make_policy() for _ in range(self.nb_agents)]
        multiagent_model = self.multiagent_model_class(self.models)
        agent = multiagent_model.agent()
        env = MultiAgentTestEnv(self.nb_agents)
        policy = MultiAgentPolicy(self.policies)
        agent.train(env, policy=policy, nb_steps=10)
        agent.test(env, policy=GreedyPolicy(), nb_steps=10)

    def make_model(self) -> MagicMock:
        """Make the mock model."""
        dummy_model = mock.MagicMock()
        dummy_model.get_best_action = mock.MagicMock(return_value=0)
        return dummy_model

    def make_policy(self) -> MagicMock:
        """Make the mock model."""
        dummy_policy = mock.MagicMock()
        dummy_policy.get_action = mock.MagicMock(return_value=0)
        return dummy_policy

    def test_policy_get_action_called(self):
        """Test policy get action is called for each subpolicy."""
        for p in self.policies:
            p.get_action.assert_called_with(self.expected_state)

    def test_model_get_best_action_called(self):
        """Test model get best action is called for each submodel."""
        for m in self.models:
            m.get_best_action.assert_called_with(self.expected_state)


class TestSharedStateLearnersOneAgent(BaseTestMultiAgentModel):
    """Test shared state learners, one agent."""

    nb_agents = 1
    multiagent_model_class = SharedStateLearners
    expected_state = (MultiAgentTestEnv._DEFAULT_STATE,)


class TestSharedStateLearnersManyAgent(BaseTestMultiAgentModel):
    """Test shared state learners, many agents."""

    nb_agents = _HIGH_NB_AGENTS
    multiagent_model_class = SharedStateLearners
    expected_state = (MultiAgentTestEnv._DEFAULT_STATE,) * _HIGH_NB_AGENTS


class TestIndependentLearnersOneAgent(BaseTestMultiAgentModel):
    """Test independent learners, one agent."""

    nb_agents = 1
    multiagent_model_class = IndependentLearners
    expected_state = MultiAgentTestEnv._DEFAULT_STATE


class TestIndependentLearnersManyAgents(BaseTestMultiAgentModel):
    """Test independent learners, many agents."""

    nb_agents = _HIGH_NB_AGENTS
    multiagent_model_class = IndependentLearners
    expected_state = MultiAgentTestEnv._DEFAULT_STATE
