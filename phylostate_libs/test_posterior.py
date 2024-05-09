from laml_libs.sim_lib import get_balanced_tree
from phylostate_libs.PhyloStateModel import PhyloStateModel
import numpy as np
import phylostate_libs as psl

P = {0:{0:0.8,1:0.2},
     1:{1:0.6,2:0.4},
     2:{2:1}}
     
treeStr = get_balanced_tree(5,1)
root_distr = np.array([1-2*psl.EPS,psl.EPS,psl.EPS])
myModel = PhyloStateModel(treeStr,P,root_distr)
lb2state_all,lb2state_leaves = myModel.simulate()
print(myModel.compute_in_llh(lb2state_leaves))
print(myModel.compute_out_llh(lb2state_leaves))
print(myModel.compute_posterior(lb2state_leaves))
