from copy import copy
from dataclasses import dataclass
import os
import random
from typing import List

import numpy as np

from config import Config

@dataclass
class AgentState:
    x: int = 0
    y: int = 0
    view: int = 0

    @property
    def coords(self):
        return self.x, self.y


class MazeAgent:
    def __init__(self,
                 conf: Config):
        self.learning_rate = conf.learning_rate
        self.discount_factor = conf.discount_factor
        self.exploration_rate = conf.exploration_rate
        self.exploration_decay_rate = conf.exploration_decay_rate
        self.state = None
        self.action = None
        self._num_actions = 3
        self._num_states = conf.grid_size ** 2 * 4
        self.grid_size = conf.grid_size
        self.q = np.zeros((self._num_states, self._num_actions))    # Q-table 

    def _build_state(self, observation: AgentState):
        #   Our observations consist of 3 features (x, y, heading), but we need to represent
        #   the state as a number to find the corresponding row in the Q-Table.
        return (observation.y * self.grid_size + observation.x) * 4 + observation.view

    def begin_episode(self, observation):
        self.state = self._build_state(observation)

        # Reduce exploration over time.
        self.exploration_rate *= self.exploration_decay_rate

        #   Based on the Q-Table, get the best action for our current state.
        return np.argmax(self.q[self.state])

    def act(self, observation, reward):
        next_state = self._build_state(observation)

        # Exploration/exploitation: choose a random action or select the best one.
        enable_exploration = (1 - self.exploration_rate) <= np.random.uniform(0, 1)

        #   If we choose exploration (enable_exploration == True), we perform a random action.
        #   If we choose exploitation, we perform the best possible action for this state.
        if enable_exploration:
            next_action = np.random.randint(0, self._num_actions)
        else:
            next_action = np.argmax(self.q[next_state])

        #   We have received a reward from our previous step, and we know our future
        #   state and what action to perform next.
        #   Now, recalculate Q[state, action] in the Q-Table using the update formula.

        self.q[self.state, self.action] = (1 - self.learning_rate) * self.q[
            self.state, self.action] + self.learning_rate * (reward + self.discount_factor * np.max(self.q[next_state]))

        self.state = next_state
        self.action = next_action
        return next_action

    def eval(self):
        self.exploration_rate = 0
        self.learning_rate = 0

def run_agent(env, agent, max_episodes_to_run, max_timesteps_per_episode):

    stat = []
    for episode_index in range(max_episodes_to_run):
        timeline = []
        observation = env.reset()
        action = agent.begin_episode(observation)

        timeline.append(copy(observation))

        for timestep_index in range(max_timesteps_per_episode):
            # Perform the action and observe the new state.
            observation, reward, done = env.step(action)
            timeline.append(copy(observation))

            # Get the next action from the agent, given our new state.
            action = agent.act(observation, reward)

            # Record this episode to the history and check if the goal has been reached.
            if done:
                break

        stat.append(timeline)
    return stat, agent
    
