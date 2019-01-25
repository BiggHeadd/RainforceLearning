# -*- coding:utf-8 -*-
# Edited by bighead 19-1-25

from maze_env import Maze
from RL_brain import QLearningTable

def update():
    for episode in range(100):
        observation = env.reset()

        while True:
            # update env
            env.render()

            # choose action
            action = RL.choose_action(str(observation))

            # get reward & observation_ & done(boolean)
            observation_, reward, done = env.step(action)

            # learning
            RL.learn(str(observation), action, reward, str(observation_))

            # update
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
