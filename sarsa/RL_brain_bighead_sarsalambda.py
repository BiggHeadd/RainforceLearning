#-*- coding:utf-8 -*-
#Edited by bighead 19-1-16

import numpy as np
import pandas as pd
from RL_brain_bighead import QLearningTable

class SersaLambdaTable(QLearningTable):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(SersaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            to_be_append = pd.Series(
                    [0] * len(self.actions),
                    index = self.q_table.columns,
                    name = state,
                )
            self.q_table = self.q_table.append(to_be_append)

            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_now = self.q_table.ix[s,a]
        if s_ != 'terminal':
            q_next = r + self.gamma * self.q_table.ix[s_, a_]
        else:
            q_next = r

        gain = q_next - q_now

        self.eligibility_trace.ix[s,a] += 1

        self.q_table += self.lr * gain * self.eligibility_trace

        self.eligibility_trace *= self.gamma*self.lambda_
