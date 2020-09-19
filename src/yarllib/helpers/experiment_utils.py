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

"""This module includes utilities to run many experiments."""

import multiprocessing
from typing import Callable, List, Optional, Sequence, cast

import gym

from yarllib.helpers.base import ensure
from yarllib.helpers.history import History


def _do_job(agent, env, policy, seed, nb_episodes):
    """Run the agent training."""
    history = agent.train(env, policy, nb_episodes=nb_episodes, seed=seed)
    return history


def run_experiments(
    make_agent: Callable,
    env: gym.Env,
    policy,
    nb_runs: int = 50,
    nb_episodes: int = 500,
    nb_workers: int = 8,
    seeds: Optional[Sequence[int]] = None,
) -> List[History]:
    """
    Run many experiments with multiprocessing.

    :param make_agent: a callable to make an agent.
    :param env: the environment to use.
    :param policy: the policy.
    :param nb_runs: the number of runs.
    :param nb_episodes: the number of episodes.
    :param nb_workers: the number of workers.
    :param seeds: a list of seeds; if None, the range [0, nb_runs-1] is used.
    :return: a list of histories, one for each run.
    """
    data = []
    seeds = cast(List[int], ensure(seeds, list(range(0, nb_runs))))
    agent = make_agent()
    pool = multiprocessing.Pool(processes=nb_workers)
    results = [
        pool.apply_async(_do_job, args=(agent, env, policy, seed, nb_episodes))
        for seed in seeds
    ]
    for p in results:
        data.append(p.get())

    return data
