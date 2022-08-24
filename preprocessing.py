import pickle
import igraph as ig
import torch
import os
import math

from torch_geometric.datasets import Planetoid

def get_transformation_matrix(n):

    #Source matrix
    M = torch.triu(torch.ones(n, n))

    for i in range(n-1):
        M[i+1, 0] = 1
        M[i+1, i+1] = -i-1

    #Target matrix
    N = torch.zeros(n, n)
    N[n-1, 0] = math.sqrt(n)

    for i in range(1, n):
        N[i-1,i] = math.sqrt(i ** 2 + i)


    """
    Do we need any of this available on the other side? (except for O of course)
    """

    M_inv = torch.inverse(M)
    O = torch.matmul(N, M_inv)

    return O

def planetoid_import(dataset_name): #maybe add options here for eigen_no and largest

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


    g.add_vertices(v_num+1)
    g.add_edges(edges)

    calc_and_save(dataset_name, g) 


def calc_and_save(dataset_name, g): #if all other methods cal this one at some point, consider removing the named options.


    """
        For now we disregard all vertices with deg 0.
    """
    g.delete_vertices(g.vs.select(_degree=0))

    """
        get_adjacency() returns an igrapgh.Matrix type, from which we extract the data using the attribute .data
        laplacian() returns a 2d list, hence no need for extracting the data, since there is no wrapper.
    

        To speed up experiments, we want to pre-calculate as many things as we can. Hence, it wouldbe nice to have the 
        complement of the graph also pre-calculated, but it seems like it takes up tons of space to store the complement of a large graph.
        Proposed solution: 
    """

    print("Calculating adjacency matrix.")
    adjacency_matrix = torch.tensor(g.get_adjacency().data).type(torch.FloatTensor)
    print("Calculating laplacian matrix.")
    discrete_laplacian_matrix = torch.tensor(g.laplacian()).type(torch.FloatTensor)
    
    print("Calculating degree matrix")
    degree_matrix = torch.diagflat(
        torch.tensor(g.degree())
    ).type(torch.FloatTensor)

    print("Calculating random walk matrix")
    random_walk_matrix = torch.matmul(
        torch.diagflat(
            torch.tensor(list(map(lambda d: float(1/d),
            g.degree())))
        ),
        adjacency_matrix
    )
    
    print("loading data into dict")
    graph_data = {
        'dataset_name' : dataset_name,

        'vertex_count' : g.vcount(),
        'graph' : g,

        'adjacency_matrix' : adjacency_matrix,
        'laplacian_matrix' : discrete_laplacian_matrix,
        'degree_matrix' : degree_matrix,
        'random_walk_matrix' : random_walk_matrix
    }

    print("Saving graph data!")
    with open('data/' + dataset_name + '.pkl', 'wb') as file:
        pickle.dump(graph_data, file)

    print("Done!")


if __name__ == '__main__':
    def main():

        planetoid_import('Pubmed')

        return 0

    main()