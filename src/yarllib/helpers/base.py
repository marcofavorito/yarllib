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

"""Base helper module."""
from typing import Any

import gym
from gym.spaces import Discrete


def assert_(condition: bool, message: str = ""):
    """User-defined assert."""
    if not condition:
        raise AssertionError(message)


def ensure(value: Any, default: Any) -> Any:
    """Check that a value is not None. If so, return a default value."""
    return value if value is not None else default


def check_is_discrete(env: gym.Env) -> None:
    """
    Check that the environment has discrete state/action spaces.

    :param env: an OpenAI Gym environment.
    :return: None
    :raise ValueError: if the environment doesn't pass the check.
    """
    if type(env.observation_space) != Discrete:
        raise ValueError("Cannot handle non-discrete state space")
    if type(env.action_space) != Discrete:
        raise ValueError("Cannot handle non-discrete action space")


def check_has_time_limit(env: gym.Env) -> None:
    """
    Check the environment has the time limit.

    :param env: an OpenAI Gym environment.
    :return: None
    :raise ValueError: if the environment doesn't pass the check.
    """
    if not hasattr(env.spec, "max_episode_steps"):
        raise ValueError("The environment is not wrapped by a TimeLimit wrapper.")
