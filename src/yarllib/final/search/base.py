from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type, cast

from pandas import np

from yarllib.final.base import Agent
from yarllib.final.gym import SearchWrapper
from yarllib.final.search.collections import Collection, MinHeap
from yarllib.types import Action, State, Transition


class SearchAgent(Agent):
    """(Deterministic) Search Agent."""

    def _zero(self, _) -> float:
        """
        Return zero.

        Needed to avoid lambdas and so make the agent object easily pickable.
        :return: always returns zero.
        """
        return 0.0

    def __init__(
        self,
        nodes_class: Type[Collection] = MinHeap,
        heuristic: Optional[Callable] = None,
    ):
        """Initialize the agent."""
        super().__init__()
        self.container = nodes_class()
        self.heuristic = heuristic or self._zero

    def choose_best_action(self, state):
        """Choose best action."""
        return self.policy.get(self._get_observation(state), None)

    def _get_observation(self, state: Dict) -> Any:
        """Get the observation component."""
        return state["observation"]

    def _get_goal(self, state: Dict) -> Any:
        """Get the "desired goal" component."""
        return state["desired_goal"]

    def _compute_policy(self, best_previous: Dict) -> Dict[Any, Any]:
        """Compute the policy, given the best predecessor."""
        result = {}
        predecessor = self.goal
        while predecessor is not None:
            new_predecessor, action = best_previous.get(predecessor, (None, None))
            result[new_predecessor] = action
            predecessor = new_predecessor
        return result

    def start_episode(self, i):
        """Reset after each episode."""
        self.initial_observation = self.env.reset()
        self.initial_state = self._get_observation(self.initial_observation)
        self.goal = self._get_goal(self.initial_observation)
        self.current_state = self.initial_state
        self.is_goal_found = False
        self.policy = {}

        self.container.reset()
        self.labels: Dict[Any, int] = defaultdict(lambda: np.inf)
        self.best_previous: Dict[Any, Optional[Tuple[Any, Any]]] = defaultdict(
            lambda: None
        )
        self.enqueued = {self.current_state}
        self.container.add((0, self.current_state))
        self.labels[self.current_state] = 0

    def get_next_transitions(self, state: State) -> Dict[Action, List[Transition]]:
        """Get the next transitions from a state."""
        env = cast(SearchWrapper, self.env)
        return env.next_transitions(state)

    def done(self) -> bool:
        """Check that the search has finished."""
        return self.is_goal_found

    def explore(self) -> Iterator[Tuple[State, Action]]:
        """Explore the search graph."""
        while len(self.container) != 0 and not self.done():
            distance, self.current_state = self.container.pop()
            self.enqueued.discard(self.current_state)
            actions = self.get_next_transitions(self.current_state).keys()
            for a in actions:
                yield self.current_state, a

    def make_trial(self, state, action):
        next_transitions = self.get_next_transitions(state)[action]
        assert (
            len(next_transitions) == 1
        ), "Only deterministic environments are allowed."
        _probability, next_state, reward, _done = next_transitions[0]
        assert reward <= 0, "No positive reward allowed."
        return self.current_state, action, reward, next_state

    def backup(self, trial):
        """Back-up the new information."""
        current_state, action, reward, next_state = trial
        self.is_goal_found = self.is_goal_found or next_state == self.goal
        if next_state == self.initial_state:
            return None
        current_cost = self.labels[current_state]
        action_cost = -reward
        next_state_cost = self.labels[next_state]
        estimated_cost = self.heuristic(next_state, self.goal)
        new_cost = current_cost + action_cost + estimated_cost
        if next_state_cost > new_cost:
            self.labels[next_state] = new_cost
            self.best_previous[next_state] = (current_state, action)
            if next_state not in self.enqueued:
                self.enqueued.add(next_state)
                self.container.add((new_cost, next_state))

    def end_episode(self, episode):
        """Compute optimal policy if goal is found."""
        if self.is_goal_found:
            self.policy = self._compute_policy(self.best_previous)
