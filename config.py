from typing import Dict
from dataclasses import dataclass
import yaml

@dataclass
class Config:
    fire_prob: float = 0
    wall_prob: float = 0
    teleport_prob: float = 0
    move_prob: float = 1
    turn_prob: float = 1
    grid_size: int = 10
    learning_rate: float = 1.0,
    discount_factor: float = 0.0,
    exploration_rate: float = 0.5,
    exploration_decay_rate: float = 0.99
    max_timesteps_per_episode: float = 100
    max_episodes_to_run: float = 10000


    @classmethod
    def parse(cls, conf: Dict):
        obj = cls()
        for attr in conf:
            setattr(obj, attr, conf[attr])
        return obj

    @classmethod
    def from_file(cls, path: str):
        return cls.parse(yaml.safe_load(open(path)))
