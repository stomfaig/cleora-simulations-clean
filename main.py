import cleora
import igraph as ig
import torch
import sys
import embeddings
import heat_kernel
import igraph as ig
import torch
import os
import random

from multiprocess import Pool
from torch_geometric.datasets import Planetoid
from link_prediction import link_prediction_setup

def multi(f, data):
    with Pool(5) as p:
        return p.map(f, data)


def prepare_embedding(embedding):

    def randomise_empty(r):
        if torch.linalg.norm(r) == 0:
            return (2 * torch.rand(len(r)) - 1)
        else:
             return r

    return torch.stack(list(map(
        randomise_empty,
        embedding
    )))

def planetoid_import(dataset_name):

    g = ig.Graph()

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    dataset = Planetoid(path, dataset_name, "public")
    data = dataset[0]

    v_num = len(data.x[:-1])
    print(f"Number of vertices: {v_num}")
    
    edge_indexes = data.edge_index
    sources = edge_indexes[0]
    targets = edge_indexes[1]

    print(f"Graph is directed: {data.is_directed()}")
    if data.is_directed():
        print(f"Number of edges: {len(sources) / 2}")
    else:
        print(f"Number of edges: {len(sources)}")
    edges = set()
    if data.is_undirected():
        for idx in range(len(sources)):
            if sources[idx] < targets[idx]:
                edges.add((sources[idx].item(), targets[idx].item()))
    else:
        for idx in range(len(sources)):
            edges.add((sources[idx].item(), targets[idx].item()))


    """
    This part removes the empty vertices and scales the graph down. We have to do this, even if there are no isolated vertices in the graph,
    since the vertex labeling might not be consistent, i.e. although there are v vertices, the max vertex id can be v<.
    """

    scale_graph = ig.Graph()
    scale_graph.add_vertices(v_num + 1)
    scale_graph.add_edges(list(edges))
    scale_graph.delete_vertices(scale_graph.vs.select(_degree=0))

    vertex_count = len(scale_graph.vs)
    edges_list_t = list(
        map(
            lambda e: (e.source, e.target),
            iter(scale_graph.es)
        )
    )

    return vertex_count, edges_list_t

def get_adjacency_matrix(g):
    return torch.tensor(g.get_adjacency().data).type(torch.complex64)

def get_discrete_laplacian_matrix(g):
    return torch.tensor(g.laplacian()).type(torch.FloatTensor)

def normalise_or(r):
    norm = torch.linalg.norm(r)

    if norm == 0:
        return torch.zeros(len(r))
    return r / norm

def normalise_rows(embedding):
           
        return torch.stack(
            list(map(
                normalise_or,
                embedding
            ))
        )

train_ratio = 0.8
iteration_num = 3
embedding_dim = 100   

if __name__ == '__main__':

    def main():

        dataset = sys.argv[1]
        vertex_count, edges = planetoid_import(dataset)

        g, link_prediction_metric = link_prediction_setup(vertex_count, edges) # More custom parameters are avaialable.

        discrete_laplacian_matrix = get_discrete_laplacian_matrix(g)
        sevals, sevecs = torch.lobpcg(discrete_laplacian_matrix, k = 500, largest=False)

        random_embedding = embeddings.random_embedding_generator(len(g.vs), embedding_dim)

        data = cleora.Cleora.instant_run(
            iteration_num,
            [
                random_embedding
            ],
            discrete_laplacian_matrix
        )

        continous_laplacian = heat_kernel.ContinuousMatrix(lambda t, x: x ** t, sevals, sevecs)

        continously_propagated = [normalise_rows(torch.matmul(continous_laplacian(i/5), random_embedding.type(torch.complex64))) for i in range(5)]

        print(f"normal: {multi(link_prediction_metric, data[0])}")
        print(f"conti: {multi(link_prediction_metric, continously_propagated)}")

        return 0

    print(main())