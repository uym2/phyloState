import os
from treeswift import *
from math import *
import random
from random import lognormvariate, randint

def get_balanced_tree(tree_height,branch_length,num_nodes=None):
# create a fully balanced tree with height = `tree_height`
# each branch length = `branch_length`
    root = Node("n0",branch_length)
    #root = Node("n0",branch_length)
    root.h = 0
    node_list = [root] # serve as a stack
    idx = 1
    while node_list:
        pnode = node_list.pop()
        h = pnode.h
        if h < tree_height:
            #cnode1 = Node(str(idx),branch_length)
            cnode1 = Node("n"+str(idx),branch_length)
            cnode1.h = h+1
            node_list.append(cnode1)
            pnode.add_child(cnode1)
            if num_nodes:
                if num_nodes < idx:
                    break
            idx += 1
            
            cnode2 = Node("n"+str(idx),branch_length)
            #cnode2 = Node(str(idx),branch_length)
            cnode2.h = h+1

            node_list.append(cnode2)
            pnode.add_child(cnode2)
            if num_nodes:
                if num_nodes < idx:
                    break
            
            idx += 1
    tree = Tree()
    tree.root = root
    return tree.newick() 
