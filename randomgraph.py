import math
import random

class RandomGraph():
    def __init__(self, graph_size, edge_propability):
        self.graph = [[] for n in range(graph_size)] # adjacency list
        self.edges = 0
        self.generate_graph(graph_size, edge_propability)

    def generate_graph(self, n, p):
        for node in range(n):
            for neighbor in range(node+1, n):
                if random.random() < p: # < bcs 0 inclusive but 1 not
                    self.edges += 1
                    self.graph[node].append(neighbor)
                    self.graph[neighbor].append(node) # undirected