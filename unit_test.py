import torch
import cleora
import link_prediction
import preprocessing

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



if __name__ == '__main__':
    def main():
        
        if cleora_base_functionality() != 0:
            return -1


        return 0

    main()


