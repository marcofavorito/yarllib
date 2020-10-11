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

"""This module contains models for multi-agent decision processes."""
from abc import abstractmethod
from typing import Any, List, Optional, Sequence, Tuple, cast

import gym

from yarllib.core import Context, Model, Policy
from yarllib.helpers.base import assert_
from yarllib.types import AgentObservation, State


class AbstractMultiAgentModel(Model):
    """
    An abstract multi-agent model.

    The subclass must implement the
    """

    def __init__(self, models: Sequence[Model]):
        """Initialize a multi-agent model."""
        self.models = models

    @abstractmethod
    def process_state(self, state: State, model_id: int):
        """
        Process the state for a specific model.

        :param state: the state.
        :param model_id: the id of the submodel.
        :return: None
        """

    def get_action(self, current_state: State) -> Any:
        """Get an action."""
        assert_(
            isinstance(self.context.policy, MultiAgentPolicy),
            "Only multi-agent policy is allowed.",
        )
        policy = cast(MultiAgentPolicy, self.context.policy)
        return [
            p.get_action(self.process_state(current_state, i))
            for i, p in enumerate(policy.policies)
        ]

    def get_best_action(self, state: State) -> Any:
        """
        Get the best action.

        In the multi-agent case, an action is a list of actions.

        :param state: the state.
        :return: the best action.
        """
        return [
            a.get_best_action(self.process_state(state, i))
            for i, a in enumerate(self.models)
        ]

    def _call(self, method, *args, **kwargs):
        """Forward a call to all the sub-models."""
        for m in self.models:
            getattr(m, method)(*args, **kwargs)

    def on_session_begin(self, *args, **kwargs) -> None:
        """On session begin event."""
        self._call(self.on_session_begin.__name__, *args, **kwargs)

    def on_session_end(self, *args, **kwargs) -> None:
        """On session end event."""
        self._call(self.on_session_end.__name__, *args, **kwargs)

    def on_episode_begin(self, episode, **kwargs) -> None:
        """On episode begin event."""
        self._call(self.on_episode_begin.__name__, episode, **kwargs)

    def on_episode_end(self, episode, **kwargs) -> None:
        """On episode end event."""
        self._call(self.on_episode_end.__name__, episode, **kwargs)

    def on_step_begin(self, step, action, **kwargs) -> None:
        """On step begin event."""
        self._call(self.on_step_begin.__name__, step, action, **kwargs)

    def on_step_end(self, step, agent_observation: AgentObservation, **kwargs) -> None:
        """On step end event."""
        state, action, reward, state_p, done = agent_observation
        assert_(
            isinstance(state, (list, tuple)),
            f"Expected a tuple of states, found {type(state)}: {state}",
        )

        for i, model in enumerate(self.models):
            model.on_step_end(
                step,
                (self.process_state(state, i), action[i], reward, state_p[i], done),
            )


class MultiAgentPolicy(Policy):
    """Multi-agent policy."""

    def __init__(self, policies: List[Policy]):
        """
        Initialize a multi-agent policy.

        :param policies: the list of policies, one for each agent.
        """
        self.policies = policies

    @Policy.context.setter  # type: ignore
    def context(self, value: Optional[Context] = None) -> None:
        """Set the learning context."""
        self._context = value
        for p in self.policies:
            p._context = value

    @Policy.model.setter  # type: ignore
    def model(self, value: Optional[Model] = None) -> None:
        """Set the model."""
        assert_(
            value is None or isinstance(value, AbstractMultiAgentModel),
            "Only multi-agent models allowed.",
        )
        value = cast(AbstractMultiAgentModel, value)
        assert_(
            value is None or len(self.policies) == len(value.models),
            f"Number of policies {len(self.policies)} is different from number of models {len(value.models)}.",
        )
        self._model = value
        models = value.models if value is not None else [None] * len(self.policies)  # type: ignore
        for p, m in zip(self.policies, models):
            p.model = m

    @Policy.action_space.setter  # type: ignore
    def action_space(self, value: Optional[gym.spaces.Tuple] = None) -> None:
        """Set the action space."""
        assert_(
            value is None or isinstance(value, gym.spaces.Tuple),
            "Only tuple spaces allowed.",
        )
        self._action_space = value
        spaces = value.spaces if value is not None else [None] * len(self.policies)
        for p, s in zip(self.policies, spaces):
            p.action_space = s

    def get_action(self, state: State) -> Any:
        """Get the action from the model."""
        raise ValueError("Should not be called.")


class SharedStateLearners(AbstractMultiAgentModel):
    """Multi-agent reinforcement learning system with shared state among agents."""

    def process_state(self, state: State, model_id: int):
        """
        Process the state for a specific model.

        In this case, the state is shared among all the models.
        """
        return state


class IndependentLearners(AbstractMultiAgentModel):
    """
    Independent Learning agents. See (Tan 1993).

    Each agent has its own observations and action space, independently from
    the other agents.
    """

    def process_state(self, state: Tuple, model_id: int):
        """
        Process a tuple of state for a specific agent.

        :param state: the (joint) state.
        :param model_id: the id of the model.
        :return: return the ith component of the state.
        """
        assert_(
            len(state) == len(self.models),
            f"Expected {len(self.models)} observation for each agent, found {len(state)}.",
        )
        return state[model_id]
