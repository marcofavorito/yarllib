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
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial, singledispatch
from typing import Any, Dict, Optional, cast

import gym
import numpy as np
from gym.spaces import Discrete, MultiDiscrete, Tuple

from yarllib.types import State


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


@get_gym_space_dimension.register(MultiDiscrete)
def _multidiscrete(space: gym.spaces.MultiDiscrete) -> int:
    """Get the size of a Discrete space."""
    return int(np.prod(space.nvec))


@get_gym_space_dimension.register(Tuple)
def _tuple(space: gym.spaces.Tuple) -> int:
    """Get the size of a Discrete space."""
    return int(np.prod(get_gym_space_dimension(s) for s in space.spaces))


class QFunction(ABC):
    """Abstract Q-value function."""

    @abstractmethod
    def get_q_values(self, s: State):
        """Get all q-values given a state."""


class QTable(QFunction):
    """A complete Q-value function."""

    def __init__(
        self, nb_states: int, nb_actions: int, rng: np.random.Generator
    ) -> None:
        """Initialize the Q-table."""
        self._m = rng.random((nb_states, nb_actions)) * np.finfo(float).eps

    def get_q_values(self, s: State) -> np.ndarray:
        """Get the Q-values of a state."""
        return self._m[s]


class SparseTable(QFunction):
    """A (naive) sparse table, implemented using defaultdict."""

    @staticmethod
    def _initialize_row(nb_cols: int, rng: np.random.Generator):
        return np.finfo(float).eps * rng.random(nb_cols)

    @staticmethod
    def _is_index_type(key) -> bool:
        return isinstance(key, (int, np.int64))

    def __init__(self, nb_actions: int, rng: np.random.Generator):
        """Initialize the sparse table."""
        self._nb_actions = nb_actions
        self._m: Dict[State, np.ndarray] = defaultdict(
            partial(self._initialize_row, self._nb_actions, rng)
        )

    def get_q_values(self, s: State) -> np.ndarray:
        """Get Q-values of a state."""
        return cast(np.ndarray, self._m[s])


def to_native_type(numpy_obj):
    """From NumPy type to Python type."""
    return getattr(numpy_obj, "tolist", lambda: numpy_obj)()


def array_to_list(array):
    """From NumPy array to list."""
    if isinstance(array, np.ndarray):
        return array_to_list(array.tolist())
    elif isinstance(array, list):
        return [array_to_list(item) for item in array]
    elif isinstance(array, tuple):
        return tuple(array_to_list(item) for item in array)
    else:
        return to_native_type(array)


def get_machine_epsilon() -> float:
    """Get machine epsilon."""
    return np.finfo(float).eps


def set_seed(seed: Optional[int]) -> None:
    """Set seed."""
    random.seed(seed)
    np.random.seed(seed)


def set_env_seed(seed: Optional[int], env: gym.Env) -> None:
    """Set seed to a OpenAI Gym environment."""
    env.reset(seed=seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
