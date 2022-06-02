
import numpy as np
from typing import Tuple, Iterator

from yarllib.final.base import Agent

import gym

import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
from yarllib.final.gym import space_size
from yarllib.final.rl.base import RLAgent
from yarllib.helpers.base import get_machine_epsilon
from yarllib.types import State, Action


class REINFORCE(RLAgent):
    """
    REINFORCE implementation.

    Williams, Ronald J. "Simple statistical gradient-following
    algorithms for connectionist reinforcement learning."
    Machine learning 8.3-4 (1992): 229-256.
    """

    def __init__(self, model: tf.keras.Model, gamma: float = 0.99):
        """Initialize the agent."""
        self.model = model
        self.gamma = gamma

    def setup(self, env: gym.Env):
        """Set up."""
        super().setup(env)
        self.nb_actions = space_size(env.action_space)

    def start_episode(self, episode) -> None:
        """Start the episode."""
        super().start_episode(episode)
        self.states = []
        self.episode_log_probs = []
        self.rewards = []

    def explore(self) -> Iterator[Tuple[State, Action]]:
        """Choose the next action to take."""
        while not self.done():
            with tf.GradientTape() as tape:
                probs = self.model(np.array([self.current_state]), training=True)
                loss = self._
            dist = tfd.categorical.Categorical(self.nb_actions, dtype=tf.float32)
            action = dist.sample().numpy()[0]
            log_probs = np.log(probs)
            self.episode_log_probs.append(log_probs)
            self.states.append(self.current_state)
            yield self.current_state, action

    def make_trial(self, state, action):
        """Make the trial."""
        trial = super().make_trial(state, action)
        self.rewards.append(trial[2])
        return trial

    def end_episode(self, episode):
        """End the episode."""
        total_returns = []
        gt = 0.0
        for t, r in reversed(list(enumerate(self.rewards))):
            gt += r
            total_returns.append(gt)
            gt *= self.gamma
        total_returns = np.asarray(total_returns)
        total_returns = (total_returns - total_returns.mean()) / (total_returns.std() + get_machine_epsilon())

        policy_gradient = []
        assert len(self.episode_log_probs) == len(total_returns)
        for log_prob, gt in zip(self.episode_log_probs, total_returns):
            policy_gradient.append(-log_prob * gt)

        self.model.train_on_batch(self.states, policy_gradient)
