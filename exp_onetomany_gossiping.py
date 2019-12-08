"""This experiment extends the gossip algorithm for multi-agent systems via random walk proposed in Oliva et al. (2019) by changing the gossip communication protocol and allowing more than two agents on the same node to average their values simultaneously.
"""
import math
import random
import matplotlib
import matplotlib.pyplot as plt

from lib_helperfcts import *
from lib_heap import Heap
from lib_randomgraph import RandomGraph

def run_simulation(no_agents, graph_type, graph_size, edge_probability, mu, sigma, clock_rate, simulation_time, parameter_type, comm_limit):
    """Runs the simulation. Uses a heap for the ordering of agent actions according to Poisson process clocks and a random graph as underlying data structures. Moves agents at random and let's them gossip at random, whereby they average their values. Studies how agent values converge to the mean over time.

    Several agents on the same node can gossip at a time.
    
    Args:
        no_agents (int): Nuber of agents
        graph_type (string): Gnm
        graph_size (int): Number of nodes n
        edge_probability (float): Probability that an edge forms
        mu (float): Mean of initial agent values
        sigma (float): Standard deviation of initial agent values
        clock_rate (float): Poisson process rate for agent clocks
        simulation_time (int): Duration of experiment
        parameter_type (string): Single or sweep, affects analysis and plotting
        comm_limit (int): Number of neighboring agents with which the active agent on a node can average values
    
    Returns:
        tuple: Sigmas and corresponding times for parameter_type='sweep'
    """
    # initialize data structures
    H = Heap(no_agents)
    Gnp = RandomGraph(graph_type, graph_size, edge_probability)

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
        else:
            std_rt_val.append(std_rt_val[-1])
            std_rt_t.append(event_time)
                
        # move current agent to random neighboring node
        if len(Gnp.graph[pos]): # neighbor exists, i.e., node not isolated
            next_pos_ind = random.randint(0, len(Gnp.graph[pos])-1)
            next_pos = Gnp.graph[pos][next_pos_ind]
            Gnp.agents[next_pos].add(uuid)
        else: # stay on your sad island
            next_pos = pos
            Gnp.agents[next_pos].add(uuid)
        
        # update current agent's clock and insert it in heap
        next_clock = event_time + exp_rv(clock_rate)
        H.insert(uuid, next_clock, next_pos)

    # decomission data structures    
    del Gnp
    del H

    # print and plot
    if parameter_type == 'single':
        analyze_single(val_0, val, mean_0, std_rt_val, std_rt_t, simulation_time, no_agents)

    # or return results
    elif parameter_type == 'sweep':
        return (std_rt_val, std_rt_t)


if __name__ == "__main__":
    no_agents = 200
    mu = 0 # average value
    sigma = 10 # gaussian noise
    graph_type = 'Gnp'
    graph_size = 20 # n
    edge_probability = 0.5 # p
    clock_rate = 0.01 # Poisson process for asynchronous agent actions
    simulation_time = 1000
    comm_limit_list = [1, 2, 5, no_agents-1]
    parameter_type = 'sweep'

    plt.figure(figsize=(10,5))
    for comm_limit in comm_limit_list:
        (std_rt_val, std_rt_t) = run_simulation(no_agents, graph_type, graph_size, edge_probability, mu, sigma, clock_rate, simulation_time, parameter_type, comm_limit)
        plt.plot(std_rt_t, std_rt_val, label='interact with up to {} neighbor(s)'.format(comm_limit))
    plt.xlabel('Simulation time [s]', fontsize='large', fontweight='bold')
    plt.ylabel('Standard deviation [-]', fontsize='large', fontweight='bold')
    plt.legend()
    plt.grid('k')
    plt.show()