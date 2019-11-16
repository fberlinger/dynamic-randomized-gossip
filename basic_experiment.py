import math
import random
import matplotlib
import matplotlib.pyplot as plt

from helperfcts import *
from heap import Heap
from randomgraph import RandomGraph


def run_simulation(no_agents, graph_size, edge_probability, mu, sigma, clock_rate, simulation_time, parameter_type, comm_limit=1):
    H = Heap(no_agents)
    Gnp = RandomGraph(graph_size, edge_probability)

    # get agents started
    val = []
    for uuid in range(no_agents):
        clock = exp_rv(clock_rate)
        pos = random.randint(0, graph_size-1)
        Gnp.agents[pos].add(uuid)
        H.insert(uuid, clock, pos)
        val.append(random.gauss(mu, sigma))
    val_0 = val.copy()
    mean = sum(val) / no_agents
    mean_0 = mean
    var = sum([((x - mean) ** 2) for x in val]) / no_agents
    std = var ** 0.5
    std_rt_val = [std] # val
    std_rt_t = [0] # time


    # run agents until time t
    while True:
        # pop next agent from heap
        (uuid, event_time, pos) = H.delete_min()
        
        # stop simulation if out of time
        if event_time > simulation_time:
            break
        
        # remove current agent from node and pick random agent at same node to average values, update standard deviation
        Gnp.agents[pos].remove(uuid)
        if len(Gnp.agents[pos]): # i.e. not alone on node
            '''
            if comm_limit == 1 or len(Gnp.agents[pos]) == 1:
                neighbor = random.randint(0, len(Gnp.agents[pos]))
                self_val = val[uuid]
                neighbor_val = val[neighbor]
                avg_val = (self_val + neighbor_val) / 2
                val[uuid] = avg_val
                val[neighbor] = avg_val
                
                old_var = std_rt_val[-1] ** 2
                diff = (2*(avg_val - mean_0)**2 - (self_val - mean_0)**2 - (neighbor_val - mean_0)**2) / no_agents
                new_std = abs((old_var + diff)) ** 0.5
                std_rt_val.append(new_std)
                std_rt_t.append(event_time)

            elif comm_limit == 2:
                neighbors = random.sample(range(len(Gnp.agents[pos])), 2)
                self_val = val[uuid]
                neighbor1_val = val[neighbors[0]]
                neighbor2_val = val[neighbors[1]]
                avg_val = (self_val + neighbor1_val + neighbor2_val) / 3
                val[uuid] = avg_val
                val[neighbors[0]] = avg_val
                val[neighbors[1]] = avg_val

                old_var = std_rt_val[-1] ** 2
                diff = (3*(avg_val - mean_0)**2 - (self_val - mean_0)**2 - (neighbor1_val - mean_0)**2 - (neighbor2_val - mean_0)**2) / no_agents
                new_std = abs((old_var + diff)) ** 0.5
                std_rt_val.append(new_std)
                std_rt_t.append(event_time)

            elif comm_limit > 2:
            '''
            
            # could handle 1 neighbor case separately for speed, i.e. to avoid shuffling the entire list. however, expected number of neighbors on a node is very small, so fast on average.
            neighbors = list(Gnp.agents[pos])
            no_valid_neighbors = min(len(neighbors), comm_limit)
            random.shuffle(neighbors) # random permutation
            
            avg_val = val[uuid]
            diff = -(val[uuid] - mean_0)**2               
            for ii in range(no_valid_neighbors):
                avg_val += val[neighbors[ii]]
                diff -= (val[neighbors[ii]] - mean_0)**2
            avg_val /= (no_valid_neighbors + 1)
            diff += (no_valid_neighbors + 1) * (avg_val - mean_0)**2
            diff /= no_agents

            val[uuid] = avg_val
            for ii in range(no_valid_neighbors):
                val[neighbors[ii]] = avg_val

            old_var = std_rt_val[-1] ** 2
            new_std = abs((old_var + diff)) ** 0.5
            std_rt_val.append(new_std)
            std_rt_t.append(event_time)
                
        # move current agent to random neighboring node
        next_pos_ind = random.randint(0, len(Gnp.graph[pos])-1)
        next_pos = Gnp.graph[pos][next_pos_ind]
        Gnp.agents[next_pos].add(uuid)
        
        # update current agent's clock and insert it in heap
        next_clock = event_time + exp_rv(clock_rate)
        H.insert(uuid, next_clock, next_pos)


    # print and plot
    if parameter_type == 'single':
        analyze_basics(val_0, val, mean_0, std_rt_val, std_rt_t, simulation_time, no_agents)

    # or return results
    elif parameter_type == 'sweep':
        return (std_rt_val, std_rt_t)