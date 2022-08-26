import cleora
import link_prediction
import igraph as ig
import torch
import sys
import embeddings
import heat_kernel
import igraph as ig
import torch
import os

from multiprocess import Pool
from torch_geometric.datasets import Planetoid

train_ratio = 0.8
iteration_num = 10
embedding_dim = 100   

def multi(f, data):
    with Pool(5) as p:
        return p.map(f, data)

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

    return v_num, edges

if __name__ == '__main__':

    def main():

        dataset = sys.argv[1]
        _vertex_count, edges = planetoid_import(dataset)
        
        scale_graph = ig.Graph()
        scale_graph.add_vertices(_vertex_count + 1)
        scale_graph.add_edges(list(edges))
        scale_graph.delete_vertices(scale_graph.vs.select(_degree=0))

        vertex_count = len(scale_graph.vs)
        edges_list_t = list(
            map(
                lambda e: (e.source, e.target),
                iter(scale_graph.es)
            )
        )

        del scale_graph

        index = int(train_ratio * len(edges_list_t))
        train_edges = set(edges_list_t[:index])
        test_edges = edges_list_t[index:]

        g = ig.Graph()
        g.add_vertices(vertex_count + 1)
        g.add_edges(train_edges)
        
        print("M) Calculating adjacency matrix.")
        adjacency_matrix = torch.tensor(g.get_adjacency().data).type(torch.FloatTensor)
        print("M) Calculating laplacian matrix.")
        discrete_laplacian_matrix = torch.tensor(g.laplacian()).type(torch.FloatTensor)

        _, levecs = torch.lobpcg(discrete_laplacian_matrix, k = embedding_dim, largest=True)
        _, sevecs = torch.lobpcg(discrete_laplacian_matrix, k = embedding_dim, largest=False)


        vs = list(
            map(
                lambda v: v.index,
                sorted(
                    g.vs, 
                    key=lambda v: v.degree()
                )
            )
        )

        #At this point, the occupied RAM is about 3.4gb

        print('M) Running experiment')
        experiment = cleora.Cleora(iteration_num, [
            embeddings.random_embedding_generator(len(g.vs), embedding_dim),
            levecs,
            sevecs
        ], adjacency_matrix)

        data = experiment.run()

        link_prediction_metric = link_prediction.link_prediction_setup(g, vs, len(g.vs), set(train_edges), test_edges, testset_edges=100, testset_vertices=2000)


        print('*) processing pure')
        pure_processed = multi(link_prediction_metric, data[0])
        print('*) processing largest evec')
        levec_processed = multi(link_prediction_metric, data[1])
        print('*) processing smallest evec')
        sevec_processed = multi(link_prediction_metric, data[2])

        print(pure_processed)
        print(levec_processed)
        print(sevec_processed)

        return 0

    print(main())