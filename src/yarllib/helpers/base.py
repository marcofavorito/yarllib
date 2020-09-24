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
from collections import defaultdict
from functools import singledispatch
from typing import Any

import gym
import numpy as np
from gym.spaces import Discrete, MultiBinary, MultiDiscrete, Tuple


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


@singledispatch
def get_gym_space_dimension(space: gym.spaces.Space) -> int:
    """
    Get the cardinality of a OpenAI Gym space, if possible.

    :param space: the cardinality of a Gym space.
    :return: an integer.
    :raises ValueError: if the Gym space is not supported.
    """
    raise ValueError(f"OpenAI Gym space {type(space)} not supported.")


@get_gym_space_dimension.register(Discrete)
def _(space: gym.spaces.Discrete) -> int:
    """Get the size of a Discrete space."""
    return space.n


@get_gym_space_dimension.register(MultiBinary)
def _multibinary(space: gym.spaces.MultiBinary) -> int:
    """Get the size of a Discrete space."""
    return 2 ** space.n


@get_gym_space_dimension.register(MultiDiscrete)
def _multidiscrete(space: gym.spaces.MultiDiscrete) -> int:
    """Get the size of a Discrete space."""
    return int(np.prod(space.nvec))


@get_gym_space_dimension.register(Tuple)
def _tuple(space: gym.spaces.Tuple) -> int:
    """Get the size of a Discrete space."""
    return int(np.prod(get_gym_space_dimension(s) for s in space.spaces))


class SparseTable:
    """A (naive) sparse table, implemented using defaultdict."""

    def __init__(self, *args):
        """Initialize the sparse table."""
        assert len(args) == 2, "Only two-dimensional matrices can be represented."
        self._rows, self._cols = args
        self._m = defaultdict(lambda: np.random.rand(self._cols) * 0.01)

    def __getitem__(self, key):
        """Get an item."""
        if isinstance(key, (int, np.int64)):
            assert_(0 <= key < self._rows, f"Row index {key} out of bound.")
            return self._m[key]
        if len(key) == 2:
            row, col = key
            assert_(0 <= row < self._rows, f"Row index {row} out of bound.")
            assert_(0 <= col < self._cols, f"Column index {col} out of bound.")
            return self._m[row][col]

    def __setitem__(self, key, item) -> None:
        """Set an item."""
        self._m[key] = item
