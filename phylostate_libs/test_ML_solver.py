from laml_libs.sim_lib import get_balanced_tree
from phylostate_libs.PhyloStateModel import PhyloStateModel
from phylostate_libs.ML_solver import ML_solver
import numpy as np
import phylostate_libs as psl
import timeit
import random
from treeswift import *

P = {0:{0:0.8,1:0.2},
     1:{1:0.6,2:0.4},
     2:{2:1}}
P1 = {0:{0:0.5,1:0.5},
     1:{1:0.3,2:0.7},
     2:{2:1}}

treeStr = get_balanced_tree(11,1)
T = read_tree_newick(treeStr)
L = [node.label for node in T.traverse_leaves()]
L_small = random.sample(L,1800)
T_pruned = T.extract_tree_with(L_small,suppress_unifurcations=False)
treeStr = T_pruned.newick()

root_distr = np.array([1-2*psl.EPS,psl.EPS,psl.EPS])
state_set = {0,1,2}

for j in range(1):
    trueModel = PhyloStateModel(treeStr,P,root_distr)
    lb2state_all,lb2state_leaves = trueModel.simulate()
    true_llh_all = trueModel.compute_llh_all_observed(lb2state_all)
    true_llh_leaves = trueModel.compute_in_llh(lb2state_leaves)
    print('true_llh',true_llh_leaves)
    
    optimal_llh = -float("inf")
    optimal_answer = None
    for i in range(10):    
        start_time = timeit.default_timer()
        mySolver = ML_solver(treeStr,lb2state_leaves,state_set,root_distr)
        my_llh = mySolver.solve_EM()
        stop_time = timeit.default_timer()
        print("New llh",my_llh)
        print("Time:" + str(stop_time-start_time))
        if my_llh > optimal_llh:
            optimal_llh = my_llh
            optimal_answer = mySolver
    print('optimal',optimal_llh,np.round(optimal_answer.model.P_trans.P_matrix,3))      
    optimal_tree = optimal_answer.model.tree 
    
    #for node in optimal_tree.traverse_preorder():
    #    p = node.node_posterior
    #    s = np.argmax(p)
    #    print(node.label,s,lb2state_all[node.label]) 
