#-*- coding:utf-8 -*-
#edited by bighead 19-1-16

import numpy as np
import pandas as pd
from maze_env import Maze
from RL_brain_bighead_sarsa import SarsaLearningTable

def update():
    for episode in range(100):
        observation = env.reset()
        action = RL.choose_action(str(observation))
        while True:
            env.render()
            observation_, reward, done = env.step(action)
            action_ = RL.choose_action(str(observation_))
            RL.learn(str(observation), action, reward, str(observation_), action_)
            observation = observation_
            action = action_
            
            if done:
                break
    print("Game Over")
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()