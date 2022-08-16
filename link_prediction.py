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


#TODO potentially refactor a bit more?
#TODO either transform fully to torch or to numpy, since converting potentially leads to overheads...
def link_prediction_setup(graph_data, train_ratio=0.8, testset_edges=1000, testset_vertices=10000):

    @vectorise
    def link_prediction(embedding):

        X = []; Y = []
        g = graph_data['graph']

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

        print("Generating training data:")
        print("*) Complementing input graph")
        comp = graph_data['complement_graph']

        train_test_boundary = int(len(g.es) * train_ratio)
        edge_list = list(g.es)
        training_edges = edge_list[:train_test_boundary]

        print("*) Generating training data")
        for e in tqdm(training_edges):
            X.append(
                hadamard(e.source, e.target)
            )
            Y.append(1)

            fake_edge = random.choice(comp.es)
            X.append(
                hadamard(fake_edge.source, fake_edge.target)
            )
            Y.append(0)


        """
        Logistic regression is only implemented in numpy by defautl, so either a torch implementation is needed, or a to-back conversion...
        """

        print("*) Fitting Logistic Regression")
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

        print("Evaluating model:")
        print("*) Sorting vertices")
        vertices = list(
            map(
                lambda v: v.index,
                sorted(
                    g.vs,
                    key=lambda v: v.degree()
                )
            )
        )

        positions = []
        test_edges = random.sample(
            edge_list[train_test_boundary:],
            testset_edges
        )

        print(clf.classes_)

        print("*) Evaluating on test edges")
        for e in tqdm(test_edges):
            source = e.source
            target = e.target

            neighbours = set(g.neighbors(source, mode="out")); 
            most_popular_vertices = [vertices[-i - 1] for i in range(testset_vertices) if not vertices[-i - 1] in neighbours] 
        
            original = prediction(source, target)

            position = 1
            for vertex in most_popular_vertices:
                if prediction(source, vertex) > original:
                    position += 1

            positions.append(position)
            
        print("*) Calculating metrics")
        return [_mrr(positions), _hr10(positions)]

    return link_prediction

