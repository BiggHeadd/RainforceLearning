#-*- coding:utf-8 -*-
#edited by bighead 19-1-15

from maze_env import Maze
from RL_brain_bighead import QLearningTable

def update():
    for episode in range(100):
        observation = env.reset()
        while True:
            env.render()
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_

            if done:
                break

    print("Game Over")
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()