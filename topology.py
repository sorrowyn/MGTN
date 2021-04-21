import numpy as np
np.set_printoptions(threshold=np.inf) 
adj_file = './model/topology/coco_adj.pkl'
num_classes = 80
t = 0.4
def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    return _adj
adj = gen_A(num_classes, t, adj_file)
print(adj)

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities, naive_greedy_modularity_communities
# G = nx.from_numpy_matrix(adj,create_using=nx.DiGraph())
G = nx.from_numpy_matrix(adj)
c = list(naive_greedy_modularity_communities(G))
print(c)


# [frozenset({0, 2, 4, 5, 6, 8, 9, 10, 11, 16, 18, 25, 28, 30, 31, 33, 34, 36, 37, 40, 44, 48, 49, 56, 58, 59, 60, 62, 63, 64, 65, 67, 68, 72, 73, 74, 76, 79}), 
# frozenset({1, 3, 69, 70, 71, 77, 13, 15, 78, 14, 17, 19, 22, 26, 27, 29, 32, 35, 38, 41, 43, 46, 47, 50, 51, 52, 54, 57, 61}), 
# frozenset({66, 7, 39, 42, 75, 12, 45, 20, 21, 53, 55, 23, 24})]
