import cleora
import link_prediction
import pickle
import igraph as ig
import numpy as np
import torch
import sys
from embeddings import *
import heat_kernel

""" Structure of data:
graph_data = {
        'dataset_name' : dataset_name,

        'vertex_count' : g.vcount(),
        'graph' : g,
        'complement_graph' : g.complementer(loops = False),

        'adjacency_matrix' : adjacency_matrix,
        'laplacian_matrix' : laplacian_matrix,
        'degree_matrix' : degree_matrix,
        'random_walk_matrix' : random_walk_matrix,

        'laplacian_evecs' : laplacian_evecs
    }"""


def processing(filename, iteration_num, embedding_dim):

    print('*) Importing.')
    graph_data = pickle.load( open('data/' + filename, "rb" ))

    print('*) Calculating eigenvecors')
    largest_evec_init = largest_evec(graph_data, 100)

    print('*) Running experiments.')
    experiment = cleora.Cleora(graph_data, iteration_num, embedding_dim, [
        largest_evec_init
    ]) 
    data = experiment.run()

    """
        H = heat_kernel.HeatKernel.pre_calculated_evecs(largest_evec_init)
        heat_kernel_pure = [torch.matmul(H(i/10), largest_evec_init) for i in range(iteration_num)]
    """
    
    """
        Link prediction metric setup:
        def link_prediction_setup(graph_data, train_ratio=0.8, testset_edges=1000, testset_vertices=10000):
    """

    link_prediction_metric = link_prediction.link_prediction_setup(graph_data, testset_edges=100, testset_vertices=10000)

    #exp_pre_processed = [remove_frequencies(data[0][i], smallest_evec(graph_data, 100)) for i in range(len(data[0]))]
    pure_processed = link_prediction_metric(data[0])
    exp_processed = link_prediction_metric(data[1])
    #min_processed = link_prediction_metric(data[2])
    #heat_kernel_processed = link_prediction_metric(heat_kernel_pure)

    print(pure_processed)
    print(exp_processed)
    #print(min_processed)
    #print(heat_kernel_processed)

    return 0

if __name__ == '__main__':
    def main():
        filename = sys.argv[1]

        return processing(filename, 5, 100)

    print(main())