from treeswift import *
import numpy as np
from phylostate_libs.PhyloStateModel import PhyloStateModel
import cvxpy as cp
import phylostate_libs as psl

class ML_solver:
    def __init__(self,treeStr,leaf2state,state_set,root_distr):
        self.leaf2state = {}
        for x in leaf2state:
            self.leaf2state[x] = leaf2state[x]    
        P = self.get_random_P(state_set)
        self.model = PhyloStateModel(treeStr,P,root_distr) 

    def get_random_P(self,state_set):
        nstate = len(state_set)        
        P_matrix = np.random.dirichlet([1]*nstate,nstate)
        P = {s:{} for s in state_set}
        for i,state_i in enumerate(state_set):
            for j,state_j in enumerate(state_set):
                P[state_i][state_j] = P_matrix[i][j]
        return P

    def curr_llh(self):
        return self.model.compute_in_llh(self.leaf2state)

    def solve_EM(self,maxIters=100,eps_conv=1e-3):
        def Estep():
            self.model.compute_in_llh(self.leaf2state)
            self.model.compute_out_llh(self.leaf2state)
            sum_EP = self.model.compute_posterior(self.leaf2state)
            return sum_EP

        def Mstep(sum_EP):
            X = cp.Variable(sum_EP.shape)
            objective = cp.Maximize(cp.sum(cp.multiply(sum_EP,cp.log(X))))
            Nr,Nc = sum_EP.shape
            ones = np.zeros(Nr)+1
            constraints = [np.zeros_like(sum_EP)+1e-6 <= X, 
                           X <= np.zeros_like(sum_EP)+1, X @ ones == ones]
            prob = cp.Problem(objective,constraints)
            prob.solve(verbose=False,solver=cp.MOSEK)
            return X.value,prob.status,prob.value
        
        sum_EP = Estep()
        curr_llh = self.model.tree.llh

        for i in range(maxIters):    
            new_P,status,value = Mstep(sum_EP)
            #self.model.P_trans.P_matrix = np.round(new_P,3) + psl.EPS
            self.model.P_trans.P_matrix = new_P
            sum_EP = Estep()
            new_llh = self.model.tree.llh
            if abs(new_llh-curr_llh) < eps_conv:
                break 
            curr_llh = new_llh
        print("niter",i)
        return curr_llh
