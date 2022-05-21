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

"""Test GPI algorithms implementations."""
import logging
from typing import Dict, Optional, Tuple, Union

import gym
import numpy as np
import pytest
from gym.core import ActType, ObsType
from gym.envs.toy_text import FrozenLakeEnv, TaxiEnv
from gym.envs.toy_text.utils import categorical_sample
from gym.spaces import Discrete
from gym.wrappers import TimeLimit

from tests.helpers import make_cliff, parametrize_seed
from yarllib.planning.gpi import PolicyIterationAgent, ValueIterationAgent

logger = logging.getLogger(__name__)


RELATIVE_TOLERANCE = 0.05


class TwoCells(gym.Env):
    """Environment with two cells."""

    observation_space = Discrete(2)
    action_space = Discrete(2)

    def __init__(self):
        """Initialize the environment."""
        self.P = {
            0: {
                0: [(1.0, 0, -1, True)],
                1: [(1.0, 1, 1, False)],
            },
            1: {
                0: [(1.0, 0, 0, False)],
                1: [(1.0, 1, 0, True)],
            },
        }
        self.current_state = 0

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        """Do a step."""
        transitions = self.P[self.current_state][action]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.current_state = s
        self.lastaction = action
        return (int(s), r, d, {"prob": p})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, Dict]]:
        """Reset the environment."""
        super().reset(seed=seed)
        self.current_state = 0
        return self.current_state


parametrize_discrete_env = pytest.mark.parametrize(
    "env,optimal_reward,nb_episodes,gamma",
    [
        (TimeLimit(TwoCells(), max_episode_steps=100), 50.0, 1, 0.99),
        (make_cliff(), -13.0, 1, 0.99),
        (FrozenLakeEnv(is_slippery=False), 1.0, 1, 0.99),
        (FrozenLakeEnv(is_slippery=True), 0.82, 1000, 0.99),
        (TaxiEnv(), 7.9, 5000, 0.99),
    ],
)


@parametrize_seed(nb_seeds=2)
@parametrize_discrete_env
@pytest.mark.parametrize("agent_type", [ValueIterationAgent, PolicyIterationAgent])
def test_gpi(seed, agent_type, env, optimal_reward, nb_episodes, gamma):
    """Test GPI algorithms."""
    env.reset(seed=seed)
    agent = agent_type(env.observation_space, env.action_space, gamma=gamma)
    agent.train(env)
    history = agent.test(env, nb_episodes=nb_episodes)
    actual_optimal_reward = np.mean(history.total_rewards)
    logger.debug(
        f"env={env.unwrapped.__class__.__name__}, "
        f"expected_optimal_reward={optimal_reward}, "
        f"actual_optimal_reward={actual_optimal_reward}, "
        f"relative tolerance={RELATIVE_TOLERANCE}"
    )
    assert np.isclose(actual_optimal_reward, optimal_reward, rtol=RELATIVE_TOLERANCE)
