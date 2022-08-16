from winreg import ExpandEnvironmentStrings
import cleora
import link_prediction
import pickle
import igraph as ig
import numpy as np
import torch
import sys
from embeddings import *

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

    print('Importing.')
    graph_data = pickle.load( open('data/' + filename, "rb" ))

    """
        First we create data
        cleora.Cleora constuctor:
        def __init__(self, graph_data, iteration_num, random_embedding_dim, embeddings=[]):
    """
    experiment = cleora.Cleora(graph_data, iteration_num, embedding_dim, [
        largest_evec(graph_data, 100, embedding_dim)
    ]) 
    data = experiment.run()

    
    """
        Link prediction metric setup:
        def link_prediction_setup(graph_data, train_ratio=0.8, testset_edges=1000, testset_vertices=10000):
    """

    link_prediction_metric = link_prediction.link_prediction_setup(graph_data, testset_edges=100, testset_vertices=2000)


    pure_processed = link_prediction_metric(data[0])
    exp_processed = link_prediction_metric(data[1])

    print(pure_processed)
    print(exp_processed)

    return 0

if __name__ == '__main__':
    def main():
        filename = sys.argv[1]

        return processing(filename, 10, 100)

    print(main())