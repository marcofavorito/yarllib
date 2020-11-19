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

"""This module contains basic interfaces."""
from abc import ABC, abstractmethod

import gym


class AbstractAgent(ABC):
    """Abstract agent interface."""

    def train(self, env: gym.Env, *args, **kwargs):
        """Train an agent."""

    def test(self, env: gym.Env, *args, **kwargs):
        """Test an agent."""

    @abstractmethod
    def get_best_action(self, state):
        """Get the best action."""
