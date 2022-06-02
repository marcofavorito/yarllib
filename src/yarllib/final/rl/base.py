from yarllib.final.base import Agent


class RLAgent(Agent):
    """A reinforcement learning agent."""

    def start_episode(self, episode) -> None:
        """
        Start an episode.

        :param episode: the episode id.
        :return: None
        """
        self.current_state = self.env.reset()
        self._done = False

    def done(self) -> bool:
        """
        Check whether the episode is finished or not.

        In episodic RL, this is true if the episode finished.
        """
        return self._done

    def make_trial(self, state, action):
        """
        Make an action.

        :param state: the current state.
        :param action: the chosen action.
        :return: the tuple (s, a, r, s', done, info).
        """
        statep, reward, done, info = self.env.step(action)
        self.current_state = statep
        self._done = done
        return state, action, reward, statep, done, info
