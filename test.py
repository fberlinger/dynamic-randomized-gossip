import math
import random
import matplotlib
import matplotlib.pyplot as plt
from helperfcts import *
from heap import Heap
from randomgraph import RandomGraph
from averagevalues import AverageValues

## validation of random graph ##
print('validation of random graph:')
n = 1000
p = 0.5
Gnp = RandomGraph(n, p)
#print(Gnp.graph)
print('edges in graph = {}'.format(Gnp.edges)) # edge count
print('expected edges = {}\n'.format(round(n*(n-1)/2 *p))) # edge expectation
## - - - - ##

## validation of average values ##
print('validation of average values:')
no_agents = 10000
mu = 0
sigma = 1
AvgVal = AverageValues(no_agents, mu, sigma)
#print(AvgVal.agent)
plt.hist(AvgVal.agent, bins=100)
plt.show()
## - - - - ##