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

"""Test the Q-Learning and Sarsa implementation using FrozenLake."""

import numpy as np
import pytest
from gym.envs.toy_text import FrozenLakeEnv
from gym.wrappers import TimeLimit

from tests.helpers import parametrize_seed
from yarllib.core import HistoryCallback
from yarllib.learning.tabular import TabularQLearning, TabularSarsa
from yarllib.policies import EpsGreedyPolicy, GreedyPolicy


@parametrize_seed(nb_seeds=2)
@pytest.mark.parametrize("model_class", [TabularQLearning, TabularSarsa])
@pytest.mark.parametrize("sparse", [True, False])
def test_frozenlake(seed, model_class, sparse):
    """Test Q-Learning implementation on N-Chain environment."""
    env = FrozenLakeEnv(is_slippery=False)
    env = TimeLimit(env, max_episode_steps=200)
    history_callback = HistoryCallback()
    agent = model_class(
        env.observation_space, env.action_space, gamma=0.99, sparse=sparse
    ).agent()
    agent.train(env, policy=EpsGreedyPolicy(epsilon=1.0), nb_steps=20000, seed=seed)
    agent.test(env, policy=GreedyPolicy(), nb_episodes=10, callbacks=[history_callback])
    evaluation = history_callback.get_history()
    actual_total_rewards_mean = evaluation.total_rewards.mean()
    assert np.isclose(actual_total_rewards_mean, 1.0)
