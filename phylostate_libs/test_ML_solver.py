from problin_libs.sim_lib import get_balanced_tree
from phylostate_libs.PhyloStateModel import PhyloStateModel
from phylostate_libs.ML_solver import ML_solver
import numpy as np
import phylostate_libs as psl
import timeit

P = {0:{0:0.8,1:0.2},
     1:{1:0.6,2:0.4},
     2:{2:1}}
P1 = {0:{0:0.5,1:0.5},
     1:{1:0.3,2:0.7},
     2:{2:1}}

treeStr = get_balanced_tree(8,1)
root_distr = np.array([1-2*psl.EPS,psl.EPS,psl.EPS])
state_set = {0,1,2}

for j in range(100):
    trueModel = PhyloStateModel(treeStr,P,root_distr)
    lb2state_all,lb2state_leaves = trueModel.simulate()
    true_llh_all = trueModel.compute_llh_all_observed(lb2state_all)
    true_llh_leaves = trueModel.compute_in_llh(lb2state_leaves)
    print('true_llh',true_llh_leaves)
    
    start_time = timeit.default_timer()
    for i in range(1):    
        mySolver = ML_solver(treeStr,lb2state_leaves,state_set,root_distr)
        my_llh = mySolver.solve_EM()
        print(my_llh)
        if my_llh > true_llh_leaves:
            break
    stop_time = timeit.default_timer()
    print('optimal_llh',np.round(mySolver.model.P_trans.P_matrix,2))       
    print("Time:" + str(stop_time-start_time))
