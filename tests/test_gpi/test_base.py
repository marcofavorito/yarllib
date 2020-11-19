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
# You should h  ave received a copy of the GNU Lesser General Public License
# along with yarllib.  If not, see <https://www.gnu.org/licenses/>.
#

"""Test GPI algorithms implementations."""
import numpy as np
import pytest
from gym.envs.toy_text import FrozenLakeEnv, TaxiEnv
from gym.envs.toy_text.discrete import DiscreteEnv

from tests.helpers import make_cliff
from yarllib.planning.gpi import PolicyIterationAgent, ValueIterationAgent


class TwoCells(DiscreteEnv):
    """Environment with two cells."""

    def __init__(self):
        """Initialize the environment."""
        P = {
            0: {
                0: [(1.0, 0, -1, True)],
                1: [(1.0, 1, 1, False)],
            },
            1: {
                0: [(0.0, 0, 0, False)],
                1: [(0.0, 1, 0, True)],
            },
        }

        super().__init__(2, 2, P, [1, 0])


parametrize_discrete_env = pytest.mark.parametrize(
    "env,optimal_reward,nb_episodes",
    [
        (make_cliff(), -13.0, 1),
        (FrozenLakeEnv(is_slippery=False), 1.0, 1),
        (FrozenLakeEnv(is_slippery=True), 0.82, 10000),
        (TaxiEnv(), 7.9, 10000),
    ],
)


@parametrize_discrete_env
@pytest.mark.parametrize("agent_type", [ValueIterationAgent, PolicyIterationAgent])
def test_gpi(agent_type, env, optimal_reward, nb_episodes):
    """Test GPI algorithms.."""
    agent = agent_type(env.observation_space, env.action_space, discount=0.99)
    agent.train(env)
    history = agent.test(env, nb_episodes=nb_episodes)
    actual_optimal_reward = np.mean(history.total_rewards)
    assert np.isclose(actual_optimal_reward, optimal_reward, atol=0.07)
