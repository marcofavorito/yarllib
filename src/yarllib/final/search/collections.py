import heapq
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Collection(ABC, Generic[T]):
    """ADT to store and retrieve candidate nodes in Shortest-Path algorithms."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the data structure."""

    @abstractmethod
    def add(self, item: Any):
        """Add an item."""

    @abstractmethod
    def pop(self) -> Any:
        """Pop an item."""

    @abstractmethod
    def __len__(self):
        """Return the length."""

    def empty(self) -> bool:
        """Return true if the container is empty, else False."""
        return len(self) == 0

    def __contains__(self, item):
        """Check that the container contains an item."""


class FIFO(Collection, Generic[T]):
    """FIFO collection."""

    def __init__(self):
        """Initialize."""
        self._queue = deque()

    def reset(self) -> None:
        """Reset."""
        self._queue = deque()

    def add(self, item: Any):
        """Add item."""
        self._queue.append(item)

    def pop(self) -> Any:
        """Pop item."""
        return self._queue.popleft()

    def __len__(self) -> int:
        """Return the length."""
        return len(self._queue)


class LIFO(Collection, Generic[T]):
    """LIFO collection."""

    def __init__(self):
        """Initialize."""
        self._stack = []

    def reset(self) -> None:
        """Reset."""
        self._stack = []

    def add(self, item: Any):
        """Add item."""
        self._stack.append(item)

    def pop(self) -> Any:
        """Pop item."""
        return self._stack.pop()

    def __len__(self) -> int:
        """Return the length."""
        return len(self._stack)


class MinHeap(Collection, Generic[T]):
    """MinHeap collection."""

    def __init__(self):
        """Initialize."""
        self._heap = []

    def reset(self) -> None:
        """Reset."""
        self._heap = []

    def add(self, item: Any):
        """Add item."""
        heapq.heappush(self._heap, item)

    def pop(self) -> Any:
        """Pop item."""
        return heapq.heappop(self._heap)

    def __len__(self) -> int:
        """Return the length."""
        return len(self._heap)
