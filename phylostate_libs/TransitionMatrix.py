from treeswift import *
import numpy as np
from math import log
import scipy
import phylostate_libs as psl

class TransitionMatrix:
    def __init__(self,P):
        # P is a dictionary of dictionaries
        state_set = set(P.keys())
        for state in P:
            state_set = state_set.union(P[state].keys())
        N = len(state_set)
        self.lb2idx = {}
        self.idx2lb = [0]*N
        for i,s in enumerate(state_set):
            self.lb2idx[s] = i
            self.idx2lb[i] = s
        self.P_matrix = np.zeros((N,N))+psl.EPS
        for i in range(N):
            si = self.idx2lb[i]
            for j in range(N):
                sj = self.idx2lb[j]
                if si in P and sj in P[si]:
                    self.P_matrix[i][j] = P[si][sj]
            self.P_matrix[i] = self.P_matrix[i]/np.linalg.norm(self.P_matrix[i], ord=1)
    def __call__(self,s1,s2,nstep=1):
        return self.transit_prob(s1,s2,nstep)
    def get_TransitionMatrix(self,nstep):
        P_matrix_nstep = self.P_matrix
        for i in range(nstep-1):
            P_matrix_nstep = np.matmul(P_matrix_nstep,self.P_matrix)
        return P_matrix_nstep
    def transit_prob(self,s1,s2,nstep):
        P_matrix_nstep = self.get_TransitionMatrix(nstep)
        i = self.lb2idx[s1]
        j = self.lb2idx[s2]
        return P_matrix_nstep[i,j]
    def transit(self,s_state,nsample,nstep):
        # transit from a starting state s_state to another state in nstep
        P_matrix_nstep = self.get_TransitionMatrix(nstep)
        s_idx = self.lb2idx[s_state] # start index
        e_states = np.random.choice(self.idx2lb,nsample,p=P_matrix_nstep[s_idx][:])
        return e_states

