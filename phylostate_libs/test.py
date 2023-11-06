from phylostate_libs.TransitionMatrix import TransitionMatrix
from phylostate_libs.PhyloStateModel import PhyloStateModel
from treeswift import *
import numpy as np
import phylostate_libs as psl
from problin_libs.sim_lib import get_balanced_tree

from sys import argv


root_distr = np.array([1-psl.EPS,psl.EPS])
h = int(argv[1])
p = float(argv[2])

P = {0:{0:1-p,1:p},
     1:{1:1}}

treeStr = get_balanced_tree(h,1)
tree = read_tree_newick(treeStr)
r_left,r_right = tree.root.children
#leaves = [node.label for node in tree.traverse_leaves()]
leaves_left = [node.label for node in r_left.traverse_leaves()]
leaves_right = [node.label for node in r_right.traverse_leaves()]
l1 = leaves_left[0]
l2 = leaves_right[0]

model = PhyloStateModel(treeStr,P,root_distr)
lb2state_leaves = {x.label:1 for x in tree.traverse_leaves()}
#print(p,h,np.exp(model.compute_in_llh(lb2state_leaves)))

#lb2state_leaves[l1] = 0
#lb2state_leaves[l2] = 0
print(p,h,np.exp(model.compute_in_llh(lb2state_leaves)))

'''eps = 0.01
p0 = 0
Lmax = 0
pmax = None

while p0 <= 1.1:
    P[0][1] = p0
    P[0][0] = 1-p0
    model = PhyloStateModel(treeStr,P,root_distr)
    L = np.exp(model.compute_in_llh(lb2state_leaves))
    if L > Lmax:
        Lmax = L
        pmax = p0
    p0 += eps
print(pmax)   ''' 
