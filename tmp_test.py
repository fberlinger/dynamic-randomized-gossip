import math
from math import factorial as fac
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

def binomial(n, k):
    try:
        binom = fac(n) / (fac(k) * fac(n - k))
    except ValueError:
        binom = 0
    return binom

n = 100
A = 10

pr_gossip = 1 - fac(n) / fac(n - A) / n**A

print(pr_gossip)

