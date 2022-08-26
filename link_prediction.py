from wsgiref.util import request_uri
import cleora
import torch
from functools import cache
from tqdm import tqdm
import random
from sklearn.linear_model import LogisticRegression
import numpy as np

def vectorise(f):

    def inner(data):
        results = []

        if isinstance(data, list):
            for elem in data:
                results.append(f(elem))
        else:
            results.append(f(data))
        return results
    
    return inner
        

def _mrr(positions):
    return sum(list(
        map(
            lambda x: 1/float(x),
            iter(positions)
        )
    )) / float(len(positions))

def _hr10(positions):
    return sum(list(
        map(
        lambda pos: int(pos <= 10),
        iter(positions)
        ) 
    )) / float(len(positions))


"""
    New signature: train_edges, test_edges, vertic_num, testset_vertices
"""


def link_prediction_setup(g, vs, vertex_count, training_edges, test_edges, testset_edges=1000, testset_vertices=10000):


    """
        For some reason, when trying to run this in multiprocessing, tons of memory gets used up.
    """
    @vectorise
    def link_prediction(embedding):

        X = []; Y = []
        #g = graph_data['graph'] # Every instance has its own copy from the graph?

        """
            Since the hadamard prod is symmetric, we can save memory by caching in the same order always. May take some
            extratime 'cause of the two function calls.
        """
        
        @cache
        def hadamard_calc(l):
            i, j = l
            return torch.multiply(embedding[i], embedding[j]).numpy()


        """
            We need this complicated structure because we can only hash tuples, but it has to be sorted to save space.
        """
        def hadamard(i, j):
            return hadamard_calc(tuple(
                sorted([i,j])
            ))

        #print("Generating training data:")
        #print("*) Complementing input graph")

        #print("*) Generating training data")
        for e in training_edges:
            X.append(
                hadamard(e[0], e[1])
            )
            Y.append(1)

            fake_edge = (-1, -1)
            while True:
                x = random.randrange(0, vertex_count)
                y = random.randrange(0, vertex_count)
            
                if (x != y) & ((x, y) not in training_edges) & ((y, x) not in training_edges): #This could be solved more elegantly I think
                    fake_edge = (x, y)
                    break

            X.append(
                hadamard(fake_edge[0], fake_edge[1])
            )
            Y.append(0)


        """
        Logistic regression is only implemented in numpy by defautl, so either a torch implementation is needed, or a to-back conversion...
        """

        #print("*) Fitting Logistic Regression")
        clf = LogisticRegression(random_state=0).fit(X, Y)

        """
            For the same reason as above we first sort and then cache. This comes with much smaller improvement, since we were cahing only a number,
            but may cause a performance overhead, so might change it back.
        """
        @cache
        def prediction_calc(l):
            return clf.predict_proba([hadamard_calc(l)])[0][1]

        def prediction(i, j):
            return prediction_calc(tuple(
                sorted([i,j])
            ))

        #print("Evaluating model:")
        #print("*) Sorting vertices")

        positions = []
        selected_test_edges = random.sample(
            test_edges,
            testset_edges
        )

        #print(clf.classes_)

        #print("*) Evaluating on test edges")
        for e in selected_test_edges:
            source = e[0]
            target = e[1]

            neighbours = set(g.neighbors(source, mode="out"))
            most_popular_vertices = [vs[-i - 1] for i in range(testset_vertices) if not vs[-i - 1] in neighbours]
        
            original = prediction(source, target)

            position = 1
            for vertex in most_popular_vertices:
                if prediction(source, vertex) > original:
                    position += 1

            positions.append(position)
            
        #print("*) Calculating metrics")
        return [_mrr(positions), _hr10(positions)]

    return link_prediction

