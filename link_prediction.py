import torch
import random
import numpy as np
import igraph as ig
from functools import cache
from sklearn.linear_model import LogisticRegression


def flatten(complex_list):
    return np.concatenate((complex_list.real, complex_list.imag), 0).astype(np.float32)

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

def link_prediction_setup(vertex_count, edges, train_ratio=0.8, test_edges_num=100, test_vertices=1000):

    index = int(train_ratio * len(edges))
    random.shuffle(edges)
    train_edges = set(edges[:index])
    free_edges = edges[index:] # in reality this is sampled twice for fake and for eval edges

    fake_edges = []
    for _ in range(len(train_edges)):
        (x, y) = (-1, -1)
        while(((x,y) in train_edges) or ((y, x) in train_edges) or (x == y)):
            x = random.randint(0, vertex_count)
            y = random.randint(0, vertex_count)
        fake_edges.append((x,y))

    test_edges = random.sample(
        free_edges,
        test_edges_num
    )

    g = ig.Graph()
    g.add_vertices(vertex_count + 1)
    g.add_edges(train_edges)

    vs = list(
        map(
            lambda v: v.index,
            sorted(
                g.vs, 
                key=lambda v: v.degree()
            )
        )
    )
    


    def link_prediction(embedding): # rewrite to work for multiple embeddings at the same time.

        X = []; Y = []
        
        @cache
        def hadamard_calc(l):
            i, j = l
            return torch.multiply(embedding[i], embedding[j]).numpy()

        def hadamard(i, j):
            return flatten(hadamard_calc(tuple(
                sorted([i,j])
            )))

        for (real_edge, fake_edge) in zip(train_edges, fake_edges):
            X.append(
                hadamard(real_edge[0], fake_edge[1])
            )
            Y.append(1)

            X.append(
                hadamard(fake_edge[0], fake_edge[1])
            )
            Y.append(0)

            clf = LogisticRegression().fit(X, Y)

        @cache
        def prediction_calc(l):
            return clf.predict_proba([flatten(hadamard_calc(l))])[0][1]

        def prediction(i, j):
            return prediction_calc(tuple(
                sorted([i,j])
            ))

        positions = []

        for e in test_edges:
            source = e[0]
            target = e[1]

            neighbours = set(g.neighbors(source, mode="out"))
            most_popular_vertices = [vs[-i - 1] for i in range(test_vertices) if not vs[-i - 1] in neighbours]
        
            original = prediction(source, target)

            position = 1
            for vertex in most_popular_vertices:
                if prediction(source, vertex) > original:
                    position += 1

            positions.append(position)
            
        return [_mrr(positions), _hr10(positions)]

    return (g, link_prediction)

