import torch
import cleora
import heat_kernel
import time

def cleora_base_functionality():

    iteration_num = 10
    n = 10000
    random_embedding_dim = 100

    unit = torch.zeros(n, n)
    unit.fill_diagonal_(1)

    graph_data = {
        'vertex_count' : n,
        'random_walk_matrix' : unit
    }

    c = cleora.Cleora(graph_data, iteration_num, random_embedding_dim)


    #mult_by_rwm test TODO for some reason this is not working
    """mult_test_matrix = torch.rand(n, random_embedding_dim)
    if c.mult_by_rwm(mult_test_matrix) != mult_test_matrix:
        print('mult_by_rwm test failed')
        return -1
    """
    #normalise_rows test TODO the whole thing
    
    return 0

def test_heat_kernel(N=100):

    """
        Since the heat kernel class only tries to access the laplacian and the vertex count fields of the
        graph data dict, we can just create fake data by filling those in.
    """

    vals = torch.rand(int(N * (N+1) / 2))

    real_symmetric_matrix = torch.zeros(N, N)
    i, j = torch.triu_indices(N, N)
    real_symmetric_matrix[i, j] = vals
    real_symmetric_matrix.T[i, j] = vals

    fake_graph_data = {
        'vertex_count' : N,
        'laplacian_matrix' : real_symmetric_matrix,        
    }

    start = time.process_time()
    H = heat_kernel.HeatKernel(fake_graph_data, 100)
    stop = time.process_time()
    print(f"Elapsed time for generateing the kernel: {stop - start}")
    
    return H(0)

if __name__ == '__main__':
    def main():
        
        print(test_heat_kernel(1000))

    main()


