"""Some instantiations of search agents."""
from typing import Callable

from yarllib.final.search.base import SearchAgent
from yarllib.final.search.collections import FIFO, LIFO, MinHeap


class BFSearch(SearchAgent):
    """Breadth-first search agent."""

    def __init__(self):
        """Initialize the agent."""
        super().__init__(FIFO, self._zero)


class DFSearch(SearchAgent):
    """Depth-first search agent."""

    def __init__(self):
        """Initialize the agent."""
        super().__init__(LIFO, self._zero)


class AStarAgent(SearchAgent):
    """A* search agent."""

    def __init__(self, heuristic: Callable):
        """Initialize the agent."""
        super().__init__(MinHeap, heuristic)


class BestFirstSearch(AStarAgent):
    """Best-first search agent."""

    def __init__(self):
        """Initialize the agent."""
        super().__init__(self._zero)
