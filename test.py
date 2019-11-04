from helperfcts import *
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

