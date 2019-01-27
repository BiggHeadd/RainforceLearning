# -*- coding:utf-8 -*-
# Edited by bighead 19-1-25

from RL_brain import QLearningTable
from maze_env import Maze

class QLearningTable_sarsa(QLearningTable):
    #init
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable_sarsa, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    #learn
    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)
