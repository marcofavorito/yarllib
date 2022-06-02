from collections import defaultdict
from functools import partial

import numpy as np


class Function:
    def get(self, state, action):
        pass

    def set(self, state, action, value):
        pass


class TabularFunction(Function):
    def __init__(self):
        self.table = defaultdict(partial(np.ndarray, 0))

    def get(self, state, action):
        return self.table[(state, action)]

    def set(self, state, action, value):
        self.table[(state, action)] = value
