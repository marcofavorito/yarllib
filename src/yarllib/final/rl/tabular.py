from collections import Iterator
from typing import Any

import gym
import numpy as np

from yarllib.final.base import Agent
from yarllib.final.gym import State, space_size
from yarllib.final.rl.base import RLAgent
from yarllib.helpers.base import SparseTable
from yarllib.types import Action


def eps_greedy(agent: "Agent", state: Any, epsilon: float = 0.1):
    """Run the epsilon-greedy policy."""
    return (
        agent.choose_best_action(state)
        if np.random.rand() > epsilon
        else agent.env.action_space.sample()
    )


class TDLearning(RLAgent):
    """Temporal-difference learning algorithm."""

    current_state = None

    def __init__(self, epsilon: float = 0.1, gamma: float = 0.99, lr: float = 0.1):
        """
        Initialize the TD learning agent.

        :param epsilon: the epsilon parameter in the epsilon-greedy policy.
        :param gamma: the discount factor.
        :param lr: the learning rate.
        """
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self._done = False

    def setup(self, env: gym.Env) -> None:
        """
        Set up the agent for training.

        :param env: the environment that will be used for training.
        :return: None
        """
        super().setup(env)
        self.nb_states = space_size(env.observation_space)
        self.nb_actions = space_size(env.action_space)
        self.Q = SparseTable(self.nb_states, self.nb_actions)

    def choose_best_action(self, state: State) -> Action:
        """
        Choose the best action.

        :param state: the state.
        :return: the best action according to the current Q-function.
        """
        return self.Q[state].argmax()

    def loss(self, state, action, estimate: float):
        """
        Compute the loss.

        :param state: the current state whose estimate is to update.
        :param action: the last action taken to update.
        :param estimate: the estimate.
        :return: the loss of the estimate.
        """
        return self.Q[state][action] - estimate

    def update(self, state, action, loss) -> None:
        """
        Update the estimate.

        :param state: the current state whose estimate is to update.
        :param action: the last action whose estimate is to update.
        :param loss: the computed loss.
        :return: None
        """
        self.Q[state, action] -= self.lr * loss


class QLearning(TDLearning):
    """Q-Learning algorithm."""

    def explore(self) -> Iterator:
        """
        Explore one step.

        :return: the current state and the action to be taken.
        """
        while not self.done():
            action = eps_greedy(self, self.current_state, epsilon=self.epsilon)
            yield self.current_state, action

    def backup(self, trial):
        """
        Back up the current trial.

        :param trial: the tuple (s, a, r, s', done, info).
        :return: the new estimate.
        """
        state, action, reward, statep, done, info = trial
        return reward + self.gamma * self.Q[statep].max()


class Sarsa(TDLearning):
    """Sarsa algorithm."""

    def start_episode(self, episode) -> None:
        """Start episode."""
        super().start_episode(episode)
        self._next_action = eps_greedy(self, self.current_state, epsilon=self.epsilon)

    def explore(self) -> Iterator:
        """
        Explore one step.

        :return: the current state and the action to be taken.
        """
        while not self.done():
            yield self.current_state, self._next_action

    def backup(self, trial):
        """
        Back up the current trial.

        :param trial: the tuple (s, a, r, s', done, info).
        :return: the new estimate.
        """
        state, action, reward, statep, done, info = trial
        self._next_action = eps_greedy(self, self.current_state, epsilon=self.epsilon)
        result = reward + self.gamma * self.Q[statep][self._next_action]
        return result


class ExpectedSarsa(TDLearning):
    """Implementation of Expected Sarsa."""

    def explore(self) -> Iterator:
        """
        Explore one step.

        :return: the current state and the action to be taken.
        """
        while not self.done():
            action = eps_greedy(self, self.current_state, epsilon=self.epsilon)
            yield self.current_state, action

    def backup(self, trial):
        """
        Back up the current trial.

        :param trial: the tuple (s, a, r, s', done, info).
        :return: the new estimate.
        """
        state, action, reward, statep, done, info = trial
        expected_q = self._compute_mean_q(statep)
        return reward + self.gamma * expected_q

    def _compute_mean_q(self, state) -> float:
        """Compute mean Q."""
        expected_q = 0
        q_max = np.max(self.Q[state])
        greedy_actions = np.count_nonzero(self.Q[state] == q_max)
        non_greedy_action_probability = self.epsilon / self.nb_actions
        greedy_action_probability = (
            (1 - self.epsilon) / greedy_actions
        ) + non_greedy_action_probability
        for a in range(self.nb_actions):
            qsa = self.Q[state][a]
            if qsa == q_max:
                expected_q += qsa * greedy_action_probability
            else:
                expected_q += qsa * non_greedy_action_probability
        return expected_q
