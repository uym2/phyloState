import scipy
import numpy as np

def log_sum_exp_matrix(M,x):
    # M: a NxN matrix
    # x: a Nx1 vector
    # output: y = log(Mx)
    N = x.shape[0]
    y = np.zeros(N)
    for i in range(N):
        y[i] = scipy.special.logsumexp(x,b=M[i][:])
    return y
