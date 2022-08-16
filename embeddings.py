from ast import Raise
import torch
from functools import cache

""" Structure of graph_data:
graph_data = {
        'dataset_name' : dataset_name,

        'vertex_count' : g.vcount(),
        'graph' : g,
        'complement_graph' : g.complementer(loops = False),

        'adjacency_matrix' : adjacency_matrix,
        'laplacian_matrix' : discrete_laplacian_matrix,
        'degree_matrix' : degree_matrix,
        'random_walk_matrix' : random_walk_matrix,

        'transformation_matrix' : transformation_matrix
    }

    }"""

def largest_evec(graph_data, evec_num, embedding_dim):

    O = graph_data['transformation_matrix']
    T = graph_data['transformed_laplacian']

    _, evecs = torch.lobpcg(T, k=evec_num, largest=True)

    return torch.matmul(torch.transpose(O, 0, -1), torch.concat((evecs, torch.zeros(1, embedding_dim)), 0))

def smallest_evec(graph_data):
    Raise(NotImplementedError)

def largest_evec_noise(graph_data, noise_dim, embedding_dim):
    Raise(NotImplementedError)

def smallest_evec_noise(graph_data, noise_dims):
    Raise(NotImplementedError)


