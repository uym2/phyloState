#! /usr/bin/env python

# simulate data under the state model 0-->1-->2

import random
from os import mkdir
import numpy as np
from phylostate_libs.sim_lib import get_balanced_tree
from phylostate_libs.PhyloStateModel import PhyloStateModel

p_list = [0.1,0.3,0.5]
nreps = 50
root_distr = np.array([1,0,0])

outdir = "sim_linear_3states"
mkdir(outdir)

tree_height = 8 # balance tree 
treeStr = get_balanced_tree(tree_height,1)
with open(outdir + "/model_tree.nwk",'w') as fout:
    fout.write(treeStr)

modelID = 1
for p_01 in p_list:
    for p_12 in p_list:
        P = {0:{0:1-p_01,1:p_01},
             1:{1:1-p_12,2:p_12},
             2:{2:1}}
        model =  PhyloStateModel(treeStr,P,root_distr)
        subdir = outdir + "/model"+ str(modelID)
        mkdir(subdir)
        with open(subdir+"/params.txt",'w') as fout:
            fout.write("p_01 " + str(p_01) + "\n")
            fout.write("p_12 " + str(p_12) + "\n")            
        for j in range(nreps):
            lb2state_all,lb2state_leaves = model.simulate()
            repdir = subdir + "/rep" + str(j+1)
            mkdir(repdir)
            with open(repdir + "/all_nodes.txt",'w') as fout:
                    for x in lb2state_all: 
                        fout.write(str(x) + " " + str(lb2state_all[x])+"\n")
            with open(repdir + "/leaf_nodes.txt",'w') as fout:
                    for x in lb2state_leaves: 
                        fout.write(str(x) + " " + str(lb2state_leaves[x])+"\n")
        modelID += 1                                                                
