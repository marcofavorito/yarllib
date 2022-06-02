"""
N-puzzle environment.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

from gym.spaces import Discrete, MultiDiscrete

from yarllib.final.gym import GoalWrapper, SearchWrapper, State, Transition
from yarllib.types import Action

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

COST = -1.0


class _NPuzzle(SearchWrapper):
    """Implementation of the N-Puzzle."""

    @classmethod
    def _generate_initial_state(cls, n: int):
        """
        Sample initial state, randomly.

        This function makes sure the final state
        is reachable.
        """

        def count_swaps(sequence: List[int]):
            """Compute the number of swaps in the ordering."""
            result = 0
            for i in range(len(sequence) - 1):
                before, after = sequence[i], sequence[i + 1]
                result += int(before > after)
            return result

        result = list(range(n))
        random.shuffle(result)
        while count_swaps(result) % 2 == 1:
            random.shuffle(result)
        return tuple(result)

    def __init__(self, n: int = 3, initial_state: Optional[State] = None) -> None:
        """
        Initialize.

        :param n: the length of the square.
        """
        assert n >= 2, f"N must be at least 2, {n} given."
        self.n = n
        self.N = self.n * self.n
        assert set(initial_state) == set(range(self.N))
        self._initial_state = initial_state

        self.observation_space = MultiDiscrete([self.N] * self.N)
        self.action_space = Discrete(4)

    def reset(self) -> State:
        """Reset the environment."""
        initial_state = self._initial_state or self._generate_initial_state(self.N)
        self.initial_state = initial_state
        self.blank_index = initial_state.index(0)
        self.current_state = tuple(self.initial_state)
        return self.initial_state

    def step(self, action: Action):
        """Do an action."""
        assert 0 <= action <= 3
        new_blank_index = self._compute_new_blank_index(self.blank_index, action)
        self.current_state = self._swap(
            self.current_state, self.blank_index, new_blank_index
        )
        self.blank_index = new_blank_index
        return self.current_state, COST, False, dict()

    def _compute_new_blank_index(self, blank_index: int, action: Action):
        dy = int(action == DOWN) - int(action == UP)
        dx = int(action == RIGHT) - int(action == LEFT)
        y = blank_index // self.n
        x = blank_index % self.n
        new_y = max(0, min(y + dy, self.n - 1))
        new_x = max(0, min(x + dx, self.n - 1))
        new_blank_index = self.n * new_y + new_x
        return new_blank_index

    def _swap(self, sequence: Tuple, index_a, index_b):
        s = list(sequence)
        old, new = index_a, index_b
        s[old], s[new] = s[new], s[old]
        return tuple(s)

    def seed(self, seed=None):
        """Set the random seed."""
        random.seed(seed)

    def next_transitions(self, state: State) -> Dict[Action, List[Transition]]:
        """Compute the next transitions."""
        result = {}
        blank_index = state.index(0)
        for action in [LEFT, RIGHT, UP, DOWN]:
            new_blank_index = self._compute_new_blank_index(blank_index, action)
            new_state = self._swap(state, blank_index, new_blank_index)
            new_transition: Transition = (1.0, new_state, COST, False)
            result[action] = [new_transition]
        return result

    def render(self, mode="human", **kwargs):
        """Render the environment."""
        print("*" * 2 * self.n)
        for i in range(0, self.N, self.n):
            print(" ".join(map(str, self.current_state[i : i + self.n])))
        print("*" * 2 * self.n)


class NPuzzle(GoalWrapper):
    """Goal wrapper to the core NPuzzle environment."""

    def __init__(self, n: int = 3):
        """Initialize the goal wrapper for the N-puzzle."""
        env = _NPuzzle(n)
        goal = tuple(list(range(1, env.N)) + [0])
        super().__init__(env, goal)


def manhattan_distance(s1, s2):
    """Compute the Manhattan distance between two states."""
    assert len(s1) == len(s2)
    n = len(s1)
    l = int(math.sqrt(n))
    result = 0
    # compute indexes of both states.
    indexes_by_tile_1 = dict(map(reversed, enumerate(s1)))
    indexes_by_tile_2 = dict(map(reversed, enumerate(s2)))
    for i in range(n):
        source, target = indexes_by_tile_1[i], indexes_by_tile_2[i]
        x1, y1 = source // l, source % l
        x2, y2 = target // l, target % l
        result += abs(y1 - y2) + abs(x1 - x2)
    return result


def misplaced_tile_distance(s1, s2):
    """Compute the misplaced tile distance."""
    assert len(s1) == len(s2)
    result = 0
    for a, b in zip(s1, s2):
        result += int(a == b)
    return result
