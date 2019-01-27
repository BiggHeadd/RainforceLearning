# -*- coding: utf-8 -*-
# Edited by bighead

from RL_brain_sarsa_lambda import QLearningTable_sarsa_lambda
from RL_brain_sarsa import QLearningTable_sarsa
from maze_env import Maze

def update():
    for episode in range(100):
        # init state
        observation = env.reset()
        action = RL.choose_action(str(observation))

        RL.eligibility_trace *= 0
        while True:
            # update env
            env.render()

            # get reward & observation_ & done(boolean)
            observation_, reward, done  = env.step(action)

            # target state action_
            action_ = RL.choose_action(str(observation_))

            # learning
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # update
            observation = observation_
            action = action_

            if done:
                break

    print("Game Over")
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable_sarsa_lambda(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()
