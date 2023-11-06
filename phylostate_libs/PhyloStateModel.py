import numpy as np
from treeswift import *
from phylostate_libs.TransitionMatrix import TransitionMatrix
from phylostate_libs.utils import log_sum_exp_matrix
import scipy

class PhyloStateModel:
    def __init__(self,treeStr,P,root_distr):
        self.tree = read_tree_newick(treeStr)
        self.P_trans = TransitionMatrix(P)
        self.root_distr = np.array(root_distr)
        # Labeling
        autoIdx = 1
        nodeLabels = set([])
        for node in self.tree.traverse_preorder():
            if node.label in nodeLabels:
                node.label = "AutoLabel_" + str(autoIdx)
            nodeLabels.add(node.label)                
    
    def simulate(self):
        lb2state_all = {}
        lb2state_leaves = {}
        for node in self.tree.traverse_preorder():
            if node.is_root():
                p = self.root_distr
                lb2state_all[node.label] = np.random.choice(self.P_trans.idx2lb,1,p=p)[0]
            else:
                s_state = lb2state_all[node.parent.label]
                lb2state_all[node.label] = self.P_trans.transit(s_state,1,1)[0]
            if node.is_leaf():
                lb2state_leaves[node.label] = lb2state_all[node.label]
        return lb2state_all,lb2state_leaves

    def compute_llh_all_observed(self,node2state):
    # compute log-likelihood when all node's states are observed
        llh = 0
        for node in self.tree.traverse_postorder():
            if node.is_root():
                return llh
            s_p = node2state[node.parent.label]
            s_c = node2state[node.label]
            llh += np.log(self.P_trans(s_p,s_c))        
        # should have returned before getting here!    
        return llh   

    def compute_in_llh(self,leaf2state):
    # marginalize over all possible internal node states --> Felsenstein's algorithm
    # assume all branches have unit length
        N = len(self.P_trans.idx2lb)
        for node in self.tree.traverse_postorder():
            # in_llh_node
            if node.is_leaf():
                idx = self.P_trans.lb2idx[leaf2state[node.label]]
                node.in_llh = np.array([-float("inf")]*N)
                node.in_llh[idx] = 0.0
            else:
                node.in_llh = np.array([0.0]*N)
                for cnode in node.children:
                    node.in_llh = node.in_llh + cnode.in_llh_edge
            # in_llh_edge        
            if not node.is_root():        
                node.in_llh_edge = log_sum_exp_matrix(self.P_trans.P_matrix,node.in_llh)             
        self.tree.llh = scipy.special.logsumexp(self.tree.root.in_llh,b=self.root_distr)
        return self.tree.llh    

    def compute_out_llh(self,leaf2state):
    # assume compute_in_llh has been called on the input tree
    # so every node u in the tree has the attribute u.in_llh
        for node in self.tree.traverse_preorder():
            if node.is_root():
                node.out_llh = np.log(self.root_distr)
            else:
                v,w = node.parent.children
                sis_node = w if (node is v) else v
                llh_edge = log_sum_exp_matrix(self.P_trans.P_matrix,sis_node.in_llh)
                llh = llh_edge + node.parent.out_llh
                node.out_llh = log_sum_exp_matrix(self.P_trans.P_matrix.T,llh)         

    def compute_posterior(self,leaf2state):
        # assume that both compute_in_llh and compute_out_llh has been called
        # on the input tree, so every node u has attributes u.in_llh and u.out_llh
        sum_EP = np.zeros_like(self.P_trans.P_matrix) # sum of edge posteriors
        N = len(self.P_trans.idx2lb)
        for node in self.tree.traverse_preorder():
            node.node_posterior = np.exp(node.in_llh + node.out_llh - self.tree.llh)
            if not node.is_root():
                pnode = node.parent
                l_edge = node.in_llh_edge  
                log_post = np.log(pnode.node_posterior)
                curr_EP = np.zeros_like(sum_EP)
                for i in range(N):
                    for j in range(N):
                        curr_EP[i][j] = np.exp(log_post[i]-l_edge[i]+
                                               node.in_llh[j]+np.log(self.P_trans.P_matrix[i][j]))
                sum_EP = sum_EP + curr_EP        
        return sum_EP             
