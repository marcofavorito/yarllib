import itertools
from functools import partial
from typing import Dict, List

import numpy as np

from yarllib.final.gym import (
    Action,
    DiscreteEnv,
    GoalWrapper,
    SearchWrapper,
    State,
    Transition,
)
from yarllib.types import Dynamics

RIGHT = 0
LEFT = 1
SUCK = 2


class VacuumWorld(GoalWrapper, SearchWrapper):
    """The vacuum world (Stuart and Russel, 2016)."""

    def __init__(self, negative_reward: bool = True) -> None:
        """Initialize."""
        nb_states = 8
        nb_actions = 3
        self.negative_reward = negative_reward
        self.encode = partial(np.ravel_multi_index, dims=(2, 2, 2))
        P = self._compute_dynamics()
        isd = [0.0] * nb_states
        isd[0] = 1.0
        self._discrete_env = DiscreteEnv(nb_states, nb_actions, P, isd)
        goal = self.encode((1, 1, 1))
        super().__init__(self._discrete_env, goal)

    def _compute_dynamics(self) -> Dynamics:
        """Compute dynamics for the environment."""
        P: Dynamics = {}
        for is_left_clean, is_right_clean, current_position in list(
            itertools.product([0, 1], repeat=3)
        ):
            current_state = self.encode(
                (is_left_clean, is_right_clean, current_position)
            )
            P[current_state] = {}
            for action in {LEFT, RIGHT}:
                next_position = int(action == LEFT)
                next_state = self.encode((is_left_clean, is_right_clean, next_position))
                done = is_left_clean and is_right_clean
                reward = float(done) - bool(self.negative_reward)
                new_transition = (1.0, next_state, reward, bool(done))
                P[current_state][action] = [new_transition]

            # suck action
            next_left = is_left_clean or current_position == LEFT
            next_right = is_right_clean or current_position == RIGHT
            next_state = self.encode((next_left, next_right, current_position))
            done = next_left and next_right
            reward = float(done) - bool(self.negative_reward)
            new_transition = (1.0, next_state, reward, bool(done))
            P[current_state][SUCK] = [new_transition]

        return P

    def next_transitions(self, state: State) -> Dict[Action, List[Transition]]:
        """Compute the next transitions from the state."""
        return self._discrete_env.P[state]
