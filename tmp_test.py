import math
import random
from operator import add 
import matplotlib
import matplotlib.pyplot as plt
from lib_helperfcts import *
from lib_heap import Heap
from lib_randomgraph import RandomGraph

## validation of random graph Gnp ##
print('validation of random graph:')
n = 5
p = 0.5
Gnp = RandomGraph('Gnp', n, p)
print(Gnp.graph)
print('edges in graph = {}'.format(Gnp.edges)) # edge count
print('expected edges = {}\n'.format(round(n*(n-1)/2 *p))) # edge expectation
## - - - - ##

## validation of random graph Gnm ##
print('validation of GnN:')
n = 5
m = 8
gnm = RandomGraph('Gnm', n, m)
print(gnm.graph)
## - - - - ##

## validation of random graph grid #
print('validation of grid:')
n = 17
grid = RandomGraph('grid', n, m)
print(grid.graph)
## - - - - ##


import numpy as np

results = [0.38, 0.29, 0.26, 0.14, -0.03, -0.06, -0.11, -0.16, -0.23, -0.48]

print(np.var(results))

