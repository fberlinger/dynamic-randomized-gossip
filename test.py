import math
import random
import matplotlib
import matplotlib.pyplot as plt
from helperfcts import *
from heap import Heap
from randomgraph import RandomGraph

## validation of random graph ##
print('validation of random graph:')
n = 1000
p = 0.5
Gnp = RandomGraph(n, p)
#print(Gnp.graph)
print('edges in graph = {}'.format(Gnp.edges)) # edge count
print('expected edges = {}\n'.format(round(n*(n-1)/2 *p))) # edge expectation
## - - - - ##

val = [-4, 0, 3]
print(random.randint(0, 1))

print([0 for ii in range(5)])