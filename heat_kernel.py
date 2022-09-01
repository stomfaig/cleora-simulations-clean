import math
import torch

"""
In this module we are generating a heat kernel, and apply it to some input. There is not support for running
any other experiment here, since the existing Cleora module can do just that

We are going to use the fact that H_t = sum_{i} e^{-lambda_i t} phi_{i}(x) phi_{i}(y)
"""

class HeatKernel:

    """
        Want to be able to init this with the vectors already
    """

    def __init__(self, evals, evecs):
        self.evals = evals
        self.evecs = torch.transpose(evecs, 0, 1)
        self.outer_pruducts = [torch.outer(evec, evec) for evec in self.evecs]    

    def __call__(self, t = 1):
        
        n = self.evecs.shape[1] # Since it was transposed.
        
        sum = torch.eye(n, n) 

        # self.evals is not a list tho...?
        for (i, outer) in enumerate(self.outer_pruducts):
            sum += ( abs(self.evals[i]) ** float(t) - 1) * outer # can leave the reals.
        return sum

class ContinuousMatrix:

    def __init__(self, f, evals, evecs):
        self.f = f
        self.evals = evals
        self.evecs = torch.transpose(evecs, 0, 1)
        self.outer_products = [torch.outer(evec, evec) for evec in self.evecs]
    
    def __call__(self, t = 1):
        n = self.evecs.shape[1]

        sum = torch.eye(n, n).type(torch.complex64)
        for (i, outer) in enumerate(self.outer_products):
            sum += (self.f(t, self.evals[i].type(torch.complex64)) - 1) * outer.type(torch.complex64)

        return sum

class DeformedLaplacianMatrix:

    def __init__(self, discrete_laplacian, adjacency_matrix):
        self.discrete_laplacian = discrete_laplacian
        self.adjacency_matrix = adjacency_matrix

    def __call__(self, r=1):
        return self.discrete_laplacian - (r-1) * self.adjacency_matrix