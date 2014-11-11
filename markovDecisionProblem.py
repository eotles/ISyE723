'''
Created on Nov 11, 2014

@author: eotles
'''
import collections
import math
import sys

stateAndAction = collections.namedtuple("stateAndAction", ['state', 'action'])

#improper time horizon exception
class improperTimeHorizonException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class model(object):
    
    def __init__(self, N, S, A, r_t, r_N, p, l):
        self.N = N
        if(N == float("inf")):
            self.infHoriz = True
            self.l = l
        else:
            self.infHoriz = False
            self.r_N = r_N
        self.S = S
        self.A = A
        self.r_t = r_t
        self.p = p
    
    def backwardInduction(self):
        if self.infHoriz: raise(improperTimeHorizonException("Try An Infinite Time Horizon Method"))
      
    def valueIteration(self, epsilon):
        if (not self.infHoriz): raise(improperTimeHorizonException("Try A Finite Time Horizon Method"))
        convergenceLimit = epsilon*(1-self.l)/(2*self.l)
        v = dict()
        d = dict()
        
        #1 - select v_0 in V, specify E>0, set n=0
        n = 0
        tempV = dict()
        for state in self.S:
            tempV.update({state: 0})
        v.update({n: tempV})
        
        doStep2 = True
        #3 - if ||v_n+1 - v_n|| < E(1-l)/2l goto: 4 else n++ goto 2
        while(doStep2):
            n+=1
            #2 - for each s in S, compute v_n+1 = max_wrt_a_in_As(r(s,a) + sum_over_j_in_S(l*p(j|s,a)*v_n(j))
            tempV = dict()
            tempD = dict()
            for state in self.S:
                maxReward = sys.float_info.min
                argMaxReward = None                
                for action in self.A.get(state):
                    saPair = stateAndAction(state, action)
                    reward = self.r_t.get(saPair)
                    for nextState, transProb in self.p.get(saPair).iteritems():
                        reward += self.l * transProb * v.get(n-1).get(nextState)
                    if(reward > maxReward):
                        maxReward = reward
                        argMaxReward = action
                tempV.update({state: maxReward})
                tempD.update({state: argMaxReward})
            v.update({n: tempV})
            d.update({n: tempD})
                
            #3 - if ||v_n+1 - v_n|| < E(1-l)/2l goto: 4 else n++ goto 2
            distance = 0
            for state in self.S:
                distance += (v.get(n).get(state)-v.get(n-1).get(state))**2
            l2Norm = math.sqrt(distance)
            if(l2Norm < convergenceLimit):
                doStep2 = False
        
        #print(d)
        #print(v)
        
        #4 - for s in S choose d_E(s) belong to argmax_wrt_a_in_As(r(s,a) + sum_over_j_in_S(l*p(j|s,a)*v_n+1(j))
        for state in self.S:
            print("State: %s, Decision: %s, Value: %f" %(state, d.get(n).get(state), v.get(n).get(state)))
        
        
            
    
    def policyIteration(self):
        if (not self.infHoriz): raise(improperTimeHorizonException("Try A Finite Time Horizon Method"))
        
        
    def linearProgramming(self):
        if (not self.infHoriz): raise(improperTimeHorizonException("Try A Finite Time Horizon Method"))
        
        
        