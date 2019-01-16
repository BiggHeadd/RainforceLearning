#-*- coding:utf-8 -*-
#Edited by bighead 19-1-16

import numpy as np
import pandas as pd
from RL_brain_bighead import QLearningTable
from maze_env import Maze

class SarsaLearningTable(QLearningTable):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_now = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_next = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_next = r
        self.q_table.loc[s, a] += self.lr * (q_next - q_now)

