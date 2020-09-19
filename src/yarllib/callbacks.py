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

"""This module contains callbacks to customize the training/testing loop."""

from yarllib.core import LearningEventListener


class RenderEnv(LearningEventListener):
    """An OpenAI Gym renderer implemented as listener."""

    def on_episode_begin(self, *args, **kwargs) -> None:
        """On episode begin event."""
        self.context.environment.render()

    def on_step_end(self, *args, **kwargs) -> None:
        """On step end event."""
        self.context.environment.render()
