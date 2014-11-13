'''
Created in Fall 2014

@author: eotles
'''
import collections
import math
import numpy as np
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
        
    #TODO: Implement this!
    def backwardInduction(self):
        if self.infHoriz: raise(improperTimeHorizonException("Try An Infinite Time Horizon Method"))
      
    def valueIteration(self, epsilon):
        if (not self.infHoriz): raise(improperTimeHorizonException("Try A Finite Time Horizon Method"))
        #1.  select v_0 in V, specify E>0, set n=0
        v = dict()
        d = dict()
        n = 0        
        convergenceLimit = epsilon*(1-self.l)/(2*self.l)
        v.update({n: self._initV0()})
        
        doStep2 = True
        while(doStep2):
            n+=1
            #2.  for each s in S, compute v_n+1 = max_wrt_a_in_As(r(s,a) + sum_over_j_in_S(l*p(j|s,a)*v_n(j))
            tempV = dict()
            tempD = dict()
            for state in self.S:
                #find max reward action
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
                
            #3.  if ||v_n+1 - v_n|| < E(1-l)/2l goto: 4 else n++ goto 2
            l2Norm = self._l2Norm(v.get(n), v.get(n-1))
            if(l2Norm < convergenceLimit):
                doStep2 = False
        
        #4.  for s in S choose d_E(s) belong to argmax_wrt_a_in_As(r(s,a) + sum_over_j_in_S(l*p(j|s,a)*v_n+1(j))
        print("Value Iteration (E: %s)" %(epsilon))
        self._printPolicy(d, v, n)
          
    def policyIteration(self):
        if (not self.infHoriz): raise(improperTimeHorizonException("Try A Finite Time Horizon Method"))
        #1.  set n = 0 and select arbitrary decision rule d_0 in D
        v = dict()
        d = dict()
        n = 0
        
        tempD = dict()
        for state in self.S:
            tempD.update({state: self.A.get(state)[0]})
        d.update({n: tempD})
        
        improvePolicy = True
        while improvePolicy:
            #2.  (Policy evaluation) Obtain v_n by solving: (I-l*P_dn)*v = r_dn
            P_dn = dict()
            matrix_P_dn = list()
            vector_R_dn = list()
            for state in self.S:
                saPair = stateAndAction(state, d.get(n).get(state))
                row = list()
                for colState in self.S:
                    row.append(self.p.get(saPair).get(colState))
                P_dn.update({saPair: self.p.get(saPair)})
                matrix_P_dn.append(row)
                vector_R_dn.append(self.r_t.get(saPair))
            
            matrix_P_dn = np.matrix(matrix_P_dn)
            vector_R_dn = np.matrix(vector_R_dn).transpose()
            
            matrix_tempV = ((matrix_P_dn**0 - self.l*matrix_P_dn)**-1)*vector_R_dn
            matrix_tempV= np.array(matrix_tempV).tolist()
            
            tempV2 = dict()
            for index, state in enumerate(self.S):
                tempV2.update({state: matrix_tempV[index][0]})
            
            v.update({n: tempV2})
                
            #3.  (Policy improvement) Choose d_n+1 to satisfy: d_n+1 belong to argmax_d_in_D(r_d + l*P_d*v_n), setting d_n+1=d_n if possible
            piResults = self._policyImprovement(v, n)
            d.update({n+1: piResults[1]})      
            #4.  If d_n+1 = d_n and set d*=d_n, otherwise n++ and goto step 2
            if(d.get(n+1) == d.get(n)):
                improvePolicy = False
            else:
                n+=1
        
        print("Policy Iteration")
        self._printPolicy(d, v, n)
              
    def modifiedPolicyIteration(self, epsilon, m):
        if (not self.infHoriz): raise(improperTimeHorizonException("Try A Finite Time Horizon Method"))        
        #1.  select v_0 in V, specify E>0, set n=0
        v = dict()
        d = dict()
        u = dict()
        n = 0
        v.update({n: self._initV0()})
        convergenceLimit = epsilon*(1-self.l)/(2*self.l)
        
        
        improvePolicy = True
        while(improvePolicy):
            #2.  (Policy Improvment_ choose d_n+1 to satisfy d_n+1 belong to argmax_d_in_D(r_d + l*P_d*v_n), setting d_n+1=d_n if possible
            piResults = self._policyImprovement(v, n)
            d.update({n+1: piResults[1]}) 
            
            #3.  (Partial policy evaluation)
            #3.a set k=0 and u_0_n = max(rd + l*Pd*v_n)
            u.update({(n,0): piResults[0]})
            
            #3.b if ||u_0_n - v_n|| < E(1-l)/(2l) go to step 4, else go to c
            if(self._l2Norm(u.get((n,0)), v.get(n)) < convergenceLimit):
                improvePolicy = False
                break
            else:
                #3c. If k = m,, go to (e). Otherwise, compute u_n_k+l by u_n_k+1 = r_dn+1 + l*P_dn+1 = L_dn+1 * u_k_n
                for k in range(0,m):
                    #3c.  Otherwise, compute u_n_k+l by u_n_k+1 = r_dn+1 + l*P_dn+1 = L_dn+1 * u_k_n
                    #3d. Increment k by 1 and return to (c).
                    tempU = dict()
                    for state in self.S:
                        saPair = stateAndAction(state, d.get(n+1).get(state))
                        u_n_k1 = self.r_t.get(saPair)
                        for nextState, transProb in self.p.get(saPair).iteritems():
                            u_n_k1 += self.l * transProb * u.get((n,k)).get(nextState)
                        tempU.update({state: u_n_k1})
                    u.update({(n,k+1): tempU})
                #e. Set v_n+1 = u_n_mn, increment n by 1, and go to step 2.
                v.update({n+1: u.get((n,m))})
                n+=1
        #4. Set d_E = d_n+1, and stop.
        print("Modified Policy Iteration (E: %s, m: %s)" %(epsilon, m))
        self._printPolicy(d, v, n)
                
        
    #TODO: Implement this!   
    def linearProgramming(self):
        if (not self.infHoriz): raise(improperTimeHorizonException("Try A Finite Time Horizon Method"))
    
    def _policyImprovement(self, v, n):
        tempV = dict()
        tempD = dict()
        for state in self.S:
            #find max reward action
            maxReward = sys.float_info.min
            argMaxReward = None  
            for action in self.A.get(state):
                saPair = stateAndAction(state, action)
                reward = self.r_t.get(saPair)
                for nextState, transProb in self.p.get(saPair).iteritems():
                        reward += self.l * transProb * v.get(n).get(nextState)
                if(reward > maxReward):
                    maxReward = reward
                    argMaxReward = action
            tempV.update({state: maxReward})
            tempD.update({state: argMaxReward})
        return((tempV,tempD))
    
    def _initV0(self):
        tempV = dict()
        for state in self.S:
            tempV.update({state: sys.float_info.min})
        return(tempV)
    
    def _l2Norm(self, vDict1, vDict2):
        distance = 0
        for state in self.S:
            distance += (vDict1.get(state)-vDict2.get(state))**2
        return(math.sqrt(distance))

    def _printPolicy(self, d, v, n):
        for state in self.S:
            print("State: %s, Decision: %s, Value: %f" %(state, d.get(n).get(state), v.get(n).get(state)))
        print('')
        
        

        
        