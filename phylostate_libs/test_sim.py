from problin_libs.sim_lib import get_balanced_tree
from phylostate_libs.PhyloStateModel import PhyloStateModel
import numpy as np
import phylostate_libs as psl

P = {0:{0:0.8,1:0.2},
     1:{1:0.6,2:0.4},
     2:{2:1}}
P1 = {0:{0:0.5,1:0.5},
     1:{1:0.3,2:0.7},
     2:{2:1}}

treeStr = get_balanced_tree(5,1)
root_distr = np.array([1-2*psl.EPS,psl.EPS,psl.EPS])
myModel = PhyloStateModel(treeStr,P,root_distr)
lb2state_all,lb2state_leaves = myModel.simulate()
my_llh_all = myModel.compute_llh_all_observed(lb2state_all)
my_llh_leaves = myModel.compute_in_llh(lb2state_leaves)

yourModel = PhyloStateModel(treeStr,P1,root_distr)
your_llh_all = yourModel.compute_llh_all_observed(lb2state_all)
your_llh_leaves = yourModel.compute_in_llh(lb2state_leaves)

print(my_llh_all,your_llh_all,my_llh_all-your_llh_all)
print(my_llh_leaves,your_llh_leaves,my_llh_leaves-your_llh_leaves)
