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

"""Test the Q-Learning and Sarsa implementation using Cliff."""

import gym
import numpy as np
import pandas as pd
from gym.envs.toy_text import CliffWalkingEnv

from yarllib.helpers.experiment_utils import run_experiments
from yarllib.models.tabular import TabularQLearning, TabularSarsa
from yarllib.policies import EpsGreedyPolicy


class CliffWalkingEnvWrapper(gym.Wrapper):
    """A wrapper to the CliffWalking environment."""

    def step(self, action):
        """Stop the episode when the cliff is reached."""
        s, r, d, i = super().step(action)
        if r == -100:
            d = True
        return s, r, d, i


def test_cliff():
    """Test that Sarsa > QLearning in the Cliff Environment."""
    env = CliffWalkingEnvWrapper(CliffWalkingEnv())

    def make_sarsa():
        return TabularSarsa(env.observation_space, env.action_space).agent()

    def make_qlearning():
        return TabularQLearning(env.observation_space, env.action_space).agent()

    nb_episodes = 500
    nb_runs = 50
    policy = EpsGreedyPolicy(0.1)

    sarsa_histories = run_experiments(
        make_sarsa, env, policy, nb_runs=nb_runs, nb_episodes=nb_episodes
    )
    qlearning_histories = run_experiments(
        make_qlearning, env, policy, nb_runs=nb_runs, nb_episodes=nb_episodes
    )

    sarsa_total_rewards = pd.DataFrame(
        np.asarray([h.total_rewards for h in sarsa_histories])
    )
    qlearning_total_rewards = pd.DataFrame(
        np.asarray([h.total_rewards for h in qlearning_histories])
    )

    sarsa_last_reward = sarsa_total_rewards.mean(axis=0).iloc[-1]
    qlearning_last_reward = qlearning_total_rewards.mean(axis=0).iloc[-1]

    # test that they learned
    assert sarsa_last_reward > -30
    assert qlearning_last_reward > -40

    # compare sarsa and q-learning on the averaged total reward in the last episode
    # sarsa is better than q-learning
    assert sarsa_last_reward > qlearning_last_reward
