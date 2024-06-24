from laml_libs.sim_lib import get_balanced_tree
from phylostate_libs.PhyloStateModel import PhyloStateModel
from treeswift import * 
import numpy as np
import phylostate_libs as psl

P = {0:{0:0.8,1:0.2},
     1:{1:0.6,2:0.4},
     2:{2:1}}
     
treeStr1 = get_balanced_tree(5,1)
#print(treeStr)
import random 
# randomly collapse branches
t = read_tree_newick(treeStr1)
print(t.num_nodes())
for node in t.traverse_postorder():
    r = random.random()
    if not node.is_leaf() and not node.is_root() and r < 0.3:
        parent = node.parent
        children = node.children
        parent.remove_child(node)
        for child in children:
            parent.add_child(child)
#t.suppress_unifurcations()
treeStr2 = t.newick()
print(t.num_nodes())

num_polytomies = 0 
for node in t.traverse_preorder():
    if len(node.children) > 2:
        num_polytomies += 1
print(f"num polytomies:", num_polytomies)

root_distr = np.array([1-2*psl.EPS,psl.EPS,psl.EPS])

# no polytomies
myModel1 = PhyloStateModel(treeStr1,P,root_distr)
lb2state_all,lb2state_leaves = myModel1.simulate()
print(myModel1.compute_in_llh(lb2state_leaves))
print(myModel1.compute_out_llh_og(lb2state_leaves))
print(myModel1.compute_out_llh(lb2state_leaves))
print(myModel1.compute_posterior(lb2state_leaves))

# with polytomies
myModel2 = PhyloStateModel(treeStr2,P,root_distr)
lb2state_all,lb2state_leaves = myModel2.simulate()
print(myModel2.compute_in_llh(lb2state_leaves))
print(myModel2.compute_out_llh(lb2state_leaves))
print(myModel2.compute_posterior(lb2state_leaves))
