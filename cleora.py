import torch
import pickle
import numpy as np


class Cleora:

    def __init__(self, graph_data, iteration_num, random_embedding_dim, embeddings=[]):
        self.iteration_num = iteration_num
        self.graph_data = graph_data
        
        random_embedding = torch.tensor([
            [
               np.random.uniform(-1, 1) for j in range(random_embedding_dim)
            ] for i in range(graph_data['vertex_count'])
        ]) 
        self.embeddings = embeddings
        self.embeddings.insert(0, random_embedding) #TODO is there a more compact way?


        return

    def run(self, normalise=True):
        records = []

        for embedding in self.embeddings:
            record = []
            for _ in range(self.iteration_num):
                record.append(embedding)
                
                embedding = self.mult_by_rwm(embedding) #TODO can we make these manipulate the values?
                if normalise:
                    embedding = self.normalise_rows(embedding)

            records.append(record)

        return records

    def normalise_rows(self, embedding):
           
        return torch.stack(
            list(map(
                lambda r: r / torch.linalg.norm(r),
                embedding
            ))
        )

    def mult_by_rwm(self, embedding):
        return torch.matmul(self.graph_data['random_walk_matrix'], embedding)

    def edge_wise_mult(self, embedding):
        return