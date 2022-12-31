from copy import copy
from enum import Enum
import random
from typing import List, Optional, Tuple

import numpy as np

from agent import AgentState
from config import Config


class PatchType(Enum):
    BLANK = ' '
    WALL = '#'
    FIRE = '$'
    TELEPORT = '@'
    ENTER = '>'
    EXIT = '<'
    PATH = '='


class Patch:
    def __init__(self, x: int, y: int, type: PatchType = PatchType.BLANK) -> None:
        self.x = x
        self.y = y
        self.type = type
        self.link = None

    def __repr__(self) -> str:
        return str(self.type.value)

class Maze:
    def __init__(self, conf: Config) -> None:
        self.conf = conf
        self.maze: List[List[Patch]] = [[None for _ in range(conf.grid_size)] for _ in range(conf.grid_size)]
        self.agent_state: AgentState = AgentState() 
        self.entry: Optional[Patch] = None
        self.exit: Optional[Patch] = None

    def __str__(self) -> str:
        return "\n".join(map(str, self.maze)).replace(',', '')

    def at(self, coords: Tuple[int, int]):
        return self.maze[coords[1]][coords[0]]

    def setup(self):
        types = [PatchType.BLANK, PatchType.FIRE, PatchType.WALL, PatchType.TELEPORT]
        probs = [1 - (self.conf.fire_prob + self.conf.teleport_prob + self.conf.wall_prob), self.conf.fire_prob, self.conf.wall_prob, self.conf.teleport_prob]
        print(list(zip(types, probs)))
        grid = [(i, j) for i in range(self.conf.grid_size) if not (i == 0 or i == self.conf.grid_size - 1) for j in range(self.conf.grid_size) if not (j == 0 or j == self.conf.grid_size - 1)]
        random.shuffle(grid)

        y1, x1 = grid[0]
        y2, x2 = grid[1]
        grid = grid[2:]

        self.entry = Patch(x1, y1, PatchType.ENTER)
        self.exit = Patch(x2, y2, PatchType.EXIT)
        self.maze[y1][x1] = self.entry
        self.maze[y2][x2] = self.exit        

        idx = 0
        while idx < len(grid):
            y, x = grid[idx]
            idx += 1

            patch_type = np.random.choice(types, 1, replace=False, p=probs).item() if idx != len(grid) else PatchType.BLANK


            self.maze[y][x] = Patch(x, y, patch_type)

            if patch_type == PatchType.TELEPORT:
                y_pair, x_pair = grid[idx]
                idx += 1
                self.maze[y_pair][x_pair] = Patch(x_pair, y_pair, PatchType.TELEPORT)

                self.maze[y_pair][x_pair].link = self.maze[y][x]
                self.maze[y][x].link = self.maze[y_pair][x_pair]

        for i in range(len(self.maze)):
            self.maze[i][-1] = Patch(self.conf.grid_size - 1, i, PatchType.WALL)
            self.maze[i][0] = Patch(0, i, PatchType.WALL)
            self.maze[-1][i] = Patch(i, self.conf.grid_size - 1, PatchType.WALL)
            self.maze[0][i] = Patch(i, 0, PatchType.WALL)


    def reset(self):
        self.agent_state.x = self.entry.x
        self.agent_state.y = self.entry.y
        self.agent_state.view = 0
        return self.agent_state

    def step(self, action):

        done = False
        next_state = copy(self.agent_state)

        if action == 0:     # turn right
            next_state.view = (self.agent_state.view + 1) % 4 if self.conf.turn_prob > np.random.uniform(0, 1) else (self.agent_state.view - 1) % 4 
            reward = -1

        elif action == 1:   # turn left
            next_state.view = (self.agent_state.view - 1) % 4 if self.conf.turn_prob > np.random.uniform(0, 1) else (self.agent_state.view + 1) % 4
            reward = -1

        elif action == 2:   # step forward
            if self.agent_state.view == 0:    # ^   go up     (y-1)
                next_state.y -= 1
            elif self.agent_state.view == 1:  # >   go right  (x+1)
                next_state.x += 1
            elif self.agent_state.view == 2:  # v   go down   (y+1)
                next_state.y += 1
            elif self.agent_state.view == 3:  # <   go left   (x-1)
                next_state.x -= 1

            next_state_type = self.at(next_state.coords).type

            if (next_state_type == PatchType.BLANK):
                reward = -1
            elif (next_state_type == PatchType.WALL):
                reward = -10
                next_state = copy(self.agent_state)
            elif (next_state_type == PatchType.FIRE):
                reward = -1000
                done = True
            elif (next_state_type == PatchType.TELEPORT):
                coords = copy(next_state.coords)
                next_state.x = self.at(coords).link.x
                next_state.y = self.at(coords).link.y
                reward = -1
            elif (next_state_type == PatchType.EXIT):
                reward = 1000
                done = True
            elif (next_state_type == PatchType.ENTER):
                reward = -1

        self.agent_state = next_state if self.conf.move_prob > np.random.uniform(0, 1) else self.agent_state
        return self.agent_state, reward, done

def eval_run(observations: List[AgentState], max_timesteps_per_episode: int, env: Maze):
    
    if env.at(observations[-1].coords).type == PatchType.EXIT:
        return True
    return False

def plot_stats(observations: List[List[AgentState]], max_timesteps_per_episode, env: Maze):
    hit_map = np.array(list(map(lambda x: eval_run(x, max_timesteps_per_episode, env), observations)))

    bin_size = 400

    bins = np.sum(hit_map.reshape(-1, bin_size), axis=-1)

    print(bins)
