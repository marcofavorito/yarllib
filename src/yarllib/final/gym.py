# -*- coding: utf-8 -*-
"""This module contains helpers related to OpenAI Gym."""
import itertools
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import singledispatch
from typing import Any, Callable, Dict, List, Tuple

import gym
import numpy as np
from graphviz import Digraph
from gym import Env, GoalEnv
from gym.envs.toy_text.discrete import DiscreteEnv as GymDiscreteEnv
from gym.spaces import Box, Discrete, MultiDiscrete

State = Any
Action = int
Probability = float
Reward = float
Done = bool
Transition = Tuple[Probability, State, Reward, Done]
Transitions = Dict[State, Dict[Action, List[Transition]]]


def from_discrete_env_to_graphviz(
    env: "DiscreteEnv",
    state2str: Callable[[int], str] = lambda s: str(s),
    action2str: Callable[[int], str] = lambda a: str(a),
) -> Digraph:
    """From discrete environment to graphviz."""
    g = Digraph()
    g.attr(rankdir="LR")
    for state in range(env.nS):
        state_str = state2str(state)
        g.node(state_str)
        for (action, transitions) in env.P.get(state, {}).items():
            action_str = action2str(action)
            for (prob, next_state, reward, done) in transitions:
                if np.isclose(prob, 0.0):
                    continue
                taken_transition = False
                if (
                    env.laststate == state
                    and env.lastaction == action
                    and env.s == next_state
                ):
                    taken_transition = True
                next_state_str = state2str(next_state)
                g.edge(
                    state_str,
                    next_state_str,
                    label=f"{action_str}, p={prob}, r={reward}, done={done}",
                    color="red" if taken_transition else None,
                )

    if env.laststate is not None:
        g.node(state2str(env.laststate), fillcolor="lightyellow", style="filled")
    g.node(state2str(env.s), fillcolor="lightsalmon", style="filled")
    return g


class DiscreteEnv(GymDiscreteEnv):
    """
    A custom version of DiscreteEnv.

    Like DiscreteEnv, but adds:

    - 'laststate' for rendering purposes
    - 'available_actions' to get the available action from a state.
    - 'raise ValueError if action is not available in the current state.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the environment."""
        super().__init__(*args, **kwargs)

        self.laststate = None
        self.rewards = self._compute_rewards()
        self.nb_rewards = len(self.rewards)

    def _compute_rewards(self):
        """Compute the number of rewards from the transition function."""
        P = self.P
        rewards = set(
            r
            for state in P
            for action in self.P[state]
            for _, _, r, _ in P[state][action]
        )
        return sorted(rewards)

    def reset(self):
        """Reset the enviornment."""
        self.laststate = None
        return super().reset()

    def step(self, a):
        """Do a step in the enviornment."""
        self.laststate = deepcopy(self.s)
        if a not in self.available_actions(self.s):
            raise ValueError(f"Cannot perform action {a} in state {self.s}.")
        return super().step(a)

    def _is_legal_state(self, state: State):
        """Check that it is a legal state."""
        assert 0 <= state < self.nS, f"{state} is not a legal state."

    def _is_legal_action(self, action: Action):
        """Check that it is a legal action."""
        assert 0 <= action < self.nA, f"{action} is not a legal action."

    def available_actions(self, state):
        """Get the available action from a state."""
        self._is_legal_state(state)
        actions = set()
        for action, _transitions in self.P.get(state, {}).items():
            actions.add(action)
        return actions


class GoalWrapper(gym.Wrapper):
    """
    An environment with a goal state.

    This wrapper uses the same observation space
    of GoalEnv. In particular, an observation
    of this wrapper is a dictionary with keys:
    - "observation": the current observation of the environment
    - "achieved_goal": ignored, used for compatibility with GoalEnv
    - "desired_goal": the goal to achieve.
    """

    def __init__(self, env: gym.Env, goal: Any):
        """
        Initialize the wrapper.

        :param env: the environment to wrap.
        :param goal: the goal to achieve.
        """
        super().__init__(env)

        self.observation_space = gym.spaces.Dict(
            dict(
                desired_goal=self.env.observation_space,
                achieved_goal=self.env.observation_space,
                observation=self.env.observation_space,
            )
        )
        self.goal = goal

    def _process_observation(self, state) -> dict:
        """Produce an observation for this wrapper."""
        return dict(desired_goal=self.goal, achieved_goal=None, observation=state)

    def reset(self, **kwargs):
        """Reset the environment."""
        return self._process_observation(super().reset(**kwargs))

    def step(self, a):
        """Do a step."""
        state, reward, done, info = super().step(a)
        new_state = self._process_observation(state)
        new_done = done or self.goal == state
        return new_state, reward, new_done, info


class SearchWrapper(Env, ABC):
    """The search wrapper is an environment with known transitions."""

    @abstractmethod
    def next_transitions(self, state: State) -> Dict[Action, List[Transition]]:
        """
        Compute the next transitions from a state.

        :param state: the state to compute the transition from.
        :return: the transitions, indexed by action.
        """


@singledispatch
def iter_space(_):
    """Iterate over a Gym space."""
    raise NotImplementedError


@iter_space.register(Discrete)
def _(space: Discrete):  # type: ignore
    """Iterate over a discrete state space."""
    for i in range(space.n):
        yield i


@iter_space.register(MultiDiscrete)  # type: ignore
def _(space: MultiDiscrete):
    """Iterate over a discrete environment."""
    for i in itertools.product(*map(range, space.nvec)):
        yield i


@singledispatch
def space_size(_) -> int:
    """Get the size of a space. Works only for discrete spaces."""
    raise NotImplementedError


@space_size.register(Discrete)  # type: ignore
def _(space: Discrete):
    """Return the size of a Discrete space."""
    return space.n


@space_size.register(MultiDiscrete)  # type: ignore
def _(space: MultiDiscrete):
    """Return the size of a MultiDiscrete space."""
    return np.prod(space.nvec)


def combine_boxes(*args: Box) -> Box:
    """Combine a list of gym.Box spaces into one."""
    assert all(list(space.shape) == [1] for space in args)
    lows = np.asarray([space.low[0] for space in args])
    highs = np.asarray([space.high[0] for space in args])
    return Box(lows, highs)
