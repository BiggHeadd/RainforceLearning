#-*- coding:utf-8 -*-
#edited by bighead 19-1-15

import numpy as np
import pandas as pd
import time
import sys

np.random.seed(2)

N_STATE = 6
ACTIONS = ['left', 'right']
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 13
FRESH_TIME = 0.3

def build_q_table(n_states, actions):
    table = pd.DataFrame(
            np.zeros((n_states, len(actions))),
                columns=actions,
    )
    print(table)
    return table

build_q_table(N_STATE, ACTIONS)

def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name

def get_env_feedback(S, A):
    if A == 'right':
        if S == N_STATE - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R

def update_env(S, episode, step_counter):
    env_list = ['-']*(N_STATE-1)+['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s'%(episode+1, step_counter)
        sys.stdout.write('\r{}'.format(interaction))
        time.sleep(2)
        sys.stdout.write('\r              ')
        sys.stdout.flush()
    else:
        env_list[S] = 'o'
        interaction = "".join(env_list)
        sys.stdout.write('\r{}'.format(interaction))
        sys.stdout.flush()
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATE, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True
            
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_
            update_env(S, episode, step_counter+1)
            step_counter+=1
    return q_table


def rl_sarsa():
    q_table = build_q_table(N_STATE, ACTIONS)
    for episode in range(MAX_EPISODES):
        S = 0
        step_counter = 0
        A = choose_action(S, q_table)
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            S_, R = get_env_feedback(S, A)
            if S_ != 'terminal':
                A_ = choose_action(S_, q_table)
                q_next = R + GAMMA * q_table.loc[S_, A_]
            else:
                q_next = R
                is_terminated = True
                print(q_table)
            q_now = q_table.loc[S, A]
            q_table.loc[S, A] += ALPHA* (q_next - q_now)
            
            S = S_
            A = A_
            update_env(S, episode, step_counter+1)
            step_counter+=1
    return q_table


if __name__ == "__main__":
    q_table = rl_sarsa()
    print('\r\rQ-table:\n')
    print(q_table)
    
    #q_table = build_q_table(N_STATE, ACTIONS)
    #A = choose_action(0, q_table)
    #print(A)
