from abc import ABC
from typing import Any, Iterator, Optional, Tuple

import gym

from yarllib.types import Action, State


class Agent(ABC):
    """Agent base class."""

    env: gym.Env

    def train(self, env: gym.Env, nb_episodes: int = 1000) -> None:
        """
        Train the agent.

        :param env: the environment.
        :param nb_episodes: number of episodes.
        :return: None.
        """
        self.setup(env)
        for i in range(nb_episodes):
            self.start_episode(i)
            for state, action in self.explore():
                trial = self.make_trial(state, action)
                estimate = self.backup(trial)
                loss = self.loss(state, action, estimate)
                self.update(state, action, loss)
                if self.done():
                    break
            self.end_episode(i)

    def test(self, env: gym.Env, nb_episodes: int = 1000) -> None:
        """Test the agent."""
        for i in range(nb_episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.choose_best_action(state)
                action = action if action is not None else env.action_space.sample()
                state, reward, done, info = env.step(action)

    def start_episode(self, episode) -> None:
        """
        Reset the agent.

        :param episode: the episode id.
        :return: None
        """

    def done(self) -> bool:
        """Check if the training is done."""
        return True

    def setup(self, env: gym.Env):
        """Set up the agent."""
        self.env = env

    def explore(self) -> Iterator[Tuple[State, Action]]:
        """Visit many state-action pairs."""

    def make_trial(self, state, action):
        """Make a trial from a state-action pair."""

    def backup(self, trial):
        """Back-up a trial."""

    def loss(self, state, action, estimate):
        """Compute the loss."""

    def update(self, state, action, loss):
        """Update the parameters."""

    def end_episode(self, episode):
        """
        End an episode.

        :param episode: the episode id.
        :return: None
        """

    def choose_best_action(self, state) -> Optional[Any]:
        """
        Choose the best action in a state.

        :param state: the state where to take the decision.
        :return: the action to take, or None if the policy is not defined.
        """
