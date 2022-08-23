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

def largest_evec(graph_data, evec_num):
    laplacian_matrix = graph_data['laplacian_matrix']
    _, evecs = torch.lobpcg(laplacian_matrix, k = evec_num, largest=True)
    return evecs

def smallest_evec(graph_data, evec_num):
    laplacian_matrix = graph_data['laplacian_matrix']
    _, evecs = torch.lobpcg(laplacian_matrix, k = evec_num, largest=False)
    return evecs

def largest_evec_noise(graph_data, noise_dim, embedding_dim):
    Raise(NotImplementedError)

def smallest_evec_noise(graph_data, noise_dims):
    Raise(NotImplementedError)


