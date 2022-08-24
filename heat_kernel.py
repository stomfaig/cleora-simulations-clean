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

    def __init__(self, graph_data, evec_count=100):

        """
        The sampling here might need to be different.
        """

        self.graph_data = graph_data
        self.evals, self.evecs = torch.lobpcg(graph_data['laplacian_matrix'], k=evec_count, largest=True)
        self.evecs = torch.transpose(self.evecs, 0, 1)
        self.outer_pruducts = [torch.outer(evec, evec) for evec in self.evecs]

        

    @classmethod
    def pre_calculated_evecs(self, evecs):
        self.evecs = torch.transpose(self.evecs, 0, 1)
        self.outer_pruducts = [torch.outer(evec, evec) for evec in self.evecs]

    def __call__(self, t = 1):
        
        sum = torch.zeros(self.graph_data['vertex_count'], self.graph_data['vertex_count']) 

        # self.evals is not a list tho...?
        for (i, outer) in enumerate(self.outer_pruducts):
            sum += math.e ** ((-1) * self.evals[i] * t) * outer

        return sum