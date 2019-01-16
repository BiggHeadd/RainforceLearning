#-*- coding:utf-8 -*-
#Edited by bighead 19-1-16

from maze_env import Maze
from RL_brain_bighead_sarsalambda import SersaLambdaTable

def update():
    for episode in range(100):
        observation = env.reset()

        action = RL.choose_action(str(observation))
        RL.eligibility_trace *= 0

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
    RL = SersaLambdaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
