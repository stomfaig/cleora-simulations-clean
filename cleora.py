from signal import raise_signal
import torch
import pickle
import numpy as np


"""
Current usage:
supply embeddings and a propagation matrix.
returns an len(embeddings) long array, with each dimension containing the iterations on the corresponding embeding.


future TODO:
provide a propagation matrix for each embedding?
"""

def normalise_or(r):
    norm = torch.linalg.norm(r)

    if norm == 0:
        return torch.zeros(len(r))
    return r / norm

class Cleora:

    def instant_run(iteration_num, embeddings, propagation_matrix):
        records = []

        for embedding in embeddings:
            record = []
            for _ in range(iteration_num):
                record.append(embedding)
                
                embedding = torch.matmul(propagation_matrix, embedding)
                embedding = torch.stack(
                    list(map(
                        normalise_or,
                        embedding
                    ))
                )

            records.append(record)

        return records

    def __init__(self, iteration_num, embeddings=[], propagation_matrix = None):
        
        if propagation_matrix is None:
            raise ValueError('No propagation matrix provided.')
        
        self.iteration_num = iteration_num
        self.embeddings = embeddings
        self.propagator = self.get_progation_function(propagation_matrix)

    def run(self, normalise=True):
        records = []

        for embedding in self.embeddings:
            record = []
            for _ in range(self.iteration_num):
                record.append(embedding)
                
                embedding = self.propagator(embedding)
                if normalise:
                    embedding = self.normalise_rows(embedding)

            records.append(record)

        return records

    def normalise_rows(self, embedding):
           
        return torch.stack(
            list(map(
                normalise_or,
                embedding
            ))
        )

    def get_progation_function(self, propagation_matrix=None):

        if propagation_matrix is None:
            raise ValueError('No propagation matrix provided.')

        def propagate(embedding):
            return torch.matmul(propagation_matrix, embedding) 

        return propagate