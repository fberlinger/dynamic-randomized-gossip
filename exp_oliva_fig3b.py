"""This experiment runs agents on a Gnm graph and studies how individual agent errors (deviations from the mean) decrease over time as the agents are gossiping at random.
"""
import math
import random
import matplotlib
import matplotlib.pyplot as plt

from lib_helperfcts import *
from lib_heap import Heap
from lib_randomgraph import RandomGraph

def run_simulation(no_agents, graph_type, graph_size, edge_number, init_val, clock_rate, simulation_steps, multi_sim=0):
    """Runs the simulation. Uses a heap for the ordering of agent actions according to Poisson process clocks and a random graph as underlying data structures. Moves agents at random and let's them gossip at random, whereby the average their values. Studies how agent values converge to the mean over time.
    
    Args:
        no_agents (int): Nuber of agents
        graph_type (string): Gnm
        graph_size (int): Number of nodes n
        edge_number (int): Nuber of edges m
        init_val (list of floats): Initial agent values
        clock_rate (float): Poisson process rate for agent clocks
        simulation_steps (int): Number of simulation steps
        multi_sim (int, optional): Single experiment or multiple simulation runs, affects analysis
    
    Returns:
        tuple: Error and variance over time
    """
    # initialize data structures
    H = Heap(no_agents)
    Gnm = RandomGraph(graph_type, graph_size, edge_number)
    
    # get agents started
    init_mean = sum(init_val) / no_agents
    val = []
    err_0 = []
    err_t = [[] for n in range(no_agents)]
    for uuid in range(no_agents):
        clock = exp_rv(clock_rate)
        pos = random.randint(0, graph_size-1)
        Gnm.agents[pos].add(uuid)
        H.insert(uuid, clock, pos)
        val.append(init_val[uuid])
        err_0.append(init_val[uuid] - init_mean)
        err_t[uuid].append(init_val[uuid] - init_mean)
    mean_0 = sum(err_0) / no_agents
    var_0 = sum([((x - mean_0) ** 2) for x in err_0]) / no_agents
    var_t = [var_0]

    # run agents for simulation_steps-many steps
    steps = 0
    while True:
        # pop next agent from heap
        (uuid, event_time, pos) = H.delete_min()
        
        # stop simulation if out of steps
        if steps >= simulation_steps:
            break
        steps += 1
        
        # remove current agent from node and pick random agent at same node to average values, update standard deviation
        Gnm.agents[pos].remove(uuid)
        if len(Gnm.agents[pos]): # i.e. not alone on node
            neighbor_id = random.choice(list(Gnm.agents[pos]))
            self_val = val[uuid]
            neighbor_val = val[neighbor_id]
            avg_val = (self_val + neighbor_val) / 2
            val[uuid] = avg_val
            val[neighbor_id] = avg_val
            
            err_t[uuid].append(avg_val-init_mean)
            err_t[neighbor_id].append(avg_val-init_mean)
            for agent in range(no_agents):
                if agent == uuid or agent == neighbor_id:
                    continue
                else:
                    err_t[agent].append(err_t[agent][-1])

            old_var = var_t[-1]
            diff = (2*(avg_val - mean_0)**2 - (self_val - mean_0)**2 - (neighbor_val - mean_0)**2) / no_agents
            new_var = abs(old_var + diff)
            var_t.append(new_var)

        else:
            for agent in range(no_agents):
                err_t[agent].append(err_t[agent][-1])
            var_t.append(var_t[-1])
        
        # move current agent to random neighboring node
        if len(Gnm.graph[pos]): # neighbor exists, i.e., node not isolated
            next_pos_ind = random.randint(0, len(Gnm.graph[pos])-1)
            next_pos = Gnm.graph[pos][next_pos_ind]
            Gnm.agents[next_pos].add(uuid)
        else: # stay on your sad island
            next_pos = pos
            Gnm.agents[next_pos].add(uuid)
        
        # update current agent's clock and insert it in heap
        next_clock = event_time + exp_rv(clock_rate)
        H.insert(uuid, next_clock, next_pos)
    
    # decomission data structures    
    del Gnm
    del H

    # print and plot
    if not multi_sim:
        fig = plt.figure(figsize=(10,15))
        for agent in range(no_agents):
            plt.plot(list(range(simulation_steps+1)), err_t[agent])
        plt.xlim((0, 1000))
        plt.ylim((-0.6, 0.4))
        plt.xlabel('Iterations k', fontsize='large', fontweight='bold')
        plt.ylabel('Agent errors y(k)', fontsize='large', fontweight='bold')
        plt.show()
    else:
        return (err_t, var_t)


if __name__ == "__main__":
    # parameters (from Oliva et al. (2019))
    no_agents = 10
    init_val = [0.38, 0.29, 0.26, 0.14, -0.03, -0.06, -0.11, -0.16, -0.23, -0.48]
    graph_type = 'Gnm'
    graph_size = 40 # n
    edge_number = 161 # m
    clock_rate = 1 # Poisson process for asynchronous agent actions
    simulation_steps = 1000

    run_simulation(no_agents, graph_type, graph_size, edge_number, init_val, clock_rate, simulation_steps)