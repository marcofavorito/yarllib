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

"""Test helpers."""
import gym
import numpy as np
from gym.envs.toy_text import CliffWalkingEnv
from gym.envs.toy_text.cliffwalking import DOWN, LEFT, RIGHT, UP
from gym.wrappers import TimeLimit


class CliffWalkingEnvWrapper(gym.Wrapper):
    """A wrapper to the CliffWalking environment."""

    def __init__(self, *args, **kwargs):
        """Init the env."""
        super().__init__(*args, **kwargs)

    def step(self, action):
        """Stop the episode when the cliff is reached."""
        s, r, d, i = super().step(action)
        if r == -100:
            d = True
        return s, r, d, i


def make_cliff(max_episode_steps: int = 50):
    """Make the Cliff environment for testing."""
    env = CliffWalkingEnv()

    cliffs = [np.ravel_multi_index((3, y), env.shape) for y in range(1, 11)]
    cliff_reward = -100
    cliff_done = True
    # make transitions to cliff as final...
    # ...from initial state
    env.P[env.start_state_index][RIGHT] = [(1.0, cliffs[0], cliff_reward, cliff_done)]
    # ...from states above the cliff
    for cliff_x in range(1, 11):
        current_state = np.ravel_multi_index((2, cliff_x), env.shape)
        env.P[current_state][DOWN] = [
            (1.0, cliffs[cliff_x - 1], cliff_reward, cliff_done)
        ]
    # ...from the final state
    terminal_state = (env.shape[0] - 1, env.shape[1] - 1)
    terminal_state_index = np.ravel_multi_index(terminal_state, env.shape)
    env.P[terminal_state_index][UP] = []
    env.P[terminal_state_index][RIGHT] = []
    env.P[terminal_state_index][DOWN] = []
    env.P[terminal_state_index][LEFT] = [(1.0, cliffs[-1], cliff_reward, cliff_done)]

    # make no transitions from cliff
    for cliff_state in cliffs:
        for action in [UP, RIGHT, DOWN, LEFT]:
            env.P[cliff_state][action] = []

    env = CliffWalkingEnvWrapper(env)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env
