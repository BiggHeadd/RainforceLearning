# -*- coding:utf-8 -*-
# Edited by bighead 19-1-26

from RL_brain import QLearningTable
import pandas as pd
import numpy as np

class QLearningTable_sarsa_lambda(QLearningTable):
    # init
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(QLearningTable_sarsa_lambda, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    # check the state
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            to_be_append = pd.Series(
              [0]*len(self.actions),
              index=self.q_table.columns,
              name=state,
            )
            self.q_table = self.q_table.append(to_be_append)

            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r

        error = q_target - q_predict

        self.eligibility_trace.ix[s, :] *=0
        self.eligibility_trace.ix[s, a] = 1

        # update q_table
        self.q_table += self.lr * error * self.eligibility_trace

        self.eligibility_trace *= self.gamma * self.lambda_
