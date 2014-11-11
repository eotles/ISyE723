'''
Created on Nov 11, 2014

@author: eotles
'''

import markovDecisionProblem as mdp

def main():
    N = float("inf")
    S = ['s_1','s_2']
    A = {'s_1': ['a_1,1','a_1,2'],
         's_2': ['a_2,1']}
    r_t = {mdp.stateAndAction('s_1', 'a_1,1'): 5,
           mdp.stateAndAction('s_1', 'a_1,2'): 10,
           mdp.stateAndAction('s_2', 'a_2,1'): -1}
    r_N = None
    p =   {mdp.stateAndAction('s_1', 'a_1,1'): {'s_1': 0.5, 's_2': 0.5},
           mdp.stateAndAction('s_1', 'a_1,2'): {'s_1': 0,   's_2': 1},
           mdp.stateAndAction('s_2', 'a_2,1'): {'s_1': 0,   's_2': 1}}
    
    model = mdp.model(N, S, A, r_t, r_N, p)
    model.linearProgramming()


if __name__ == '__main__':
    main()