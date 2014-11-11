'''
Created on Nov 11, 2014

@author: eotles
'''
import collections

stateAndAction = collections.namedtuple("stateAndAction", ['state', 'action'])

#improper time horizon exception
class improperTimeHorizonException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class model(object):
    
    def __init__(self, N, S, A, r_t, r_N, p):
        self.N = N
        if(N == float("inf")):
            self.infHoriz = True
        else:
            self.infHoriz = False
            self.r_N = r_N
        self.S = S
        self.A = A
        self.r_t = r_t
        self.p = p
    
    def backwardInduction(self):
        if self.infHoriz: raise(improperTimeHorizonException("Try An Infinite Time Horizon Method"))
      
    def valueIteration(self):
        if (not self.infHoriz): raise(improperTimeHorizonException("Try A Finite Time Horizon Method"))
        
        #1 - select v_0 in V, specify E>0, set n=0
        
        #2 - for each s in S, compute v_n+1 = max_wrt_a_in_As(r(s,a) + sum_over_j_in_S(l*p(j|s,a)*v_n(j))
        
        #3 - if ||v_n+1 - v_n|| < E(1-l)/2l goto: 4 else n++ goto 2
        
        #4 - for s in S choose d_E(s) belong to argmax_wrt_a_in_As(r(s,a) + sum_over_j_in_S(l*p(j|s,a)*v_n+1(j))
        
        
            
    
    def policyIteration(self):
        if (not self.infHoriz): raise(improperTimeHorizonException("Try A Finite Time Horizon Method"))
        
        
    def linearProgramming(self):
        if (not self.infHoriz): raise(improperTimeHorizonException("Try A Finite Time Horizon Method"))
        
        
        