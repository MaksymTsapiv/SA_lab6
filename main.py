import random
import numpy as np
from agent import AgentState, MazeAgent, run_agent
from config import Config

from environment import PatchType, plot_stats, Maze

def main():
    random_state = 42   # random seed -- change it to change maze configuration

    np.random.seed(random_state)
    random.seed(random_state)
    print("Confing infornation\n")
    conf = Config.from_file('conf.yaml')
    print(conf)
    env = Maze(conf)
    env.setup()
    agent = MazeAgent(conf)
    print("\nThis is the maze you generated (I like this one) \n")
    print(env)
    print("\nThe agent is learning...")
    history, agent = run_agent(env, agent, conf.max_episodes_to_run, conf.max_timesteps_per_episode)

    print("This is how the learning went\n")
    plot_stats(history, env.conf.max_timesteps_per_episode, env)

    agent.eval()
    history, agent = run_agent(env, agent, 1, conf.max_timesteps_per_episode)
    
    print("\nThis is how the agent solves the maze after learning\n")
    for state in history[0]:
        env.at(state.coords).type = PatchType.PATH if env.at(state.coords).type == PatchType.BLANK else env.at(state.coords).type
    print(env)
    

    
if __name__ == "__main__":
    main()
