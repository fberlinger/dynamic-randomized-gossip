"""This experiment runs agents on a G graph and studies how individual agent errors (deviations from the mean) decrease over time as the agents are gossiping at random.
"""
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from operator import add

from lib_helperfcts import *
from lib_heap import Heap
from lib_randomgraph import RandomGraph

def run_simulation(no_agents, init_val, clock_rate, graph_type, graph_size, edge_number, simulation_steps, multi_sim=0):
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
    G = RandomGraph(graph_type, graph_size, edge_number)
    
    # get agents started
    init_mean = sum(init_val) / no_agents
    val = []
    err_0 = []
    err_t = [[] for n in range(no_agents)]
    for uuid in range(no_agents):
        clock = exp_rv(clock_rate)
        pos = random.randint(0, graph_size-1)
        G.agents[pos].add(uuid)
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
        
        # pick random agent at same node to average values, update standard deviation
        for agent in range(no_agents): # most agents keep same error
            err_t[agent].append(err_t[agent][-1])

        neighbor_id = random.choice(list(G.agents[pos]))

        if neighbor_id == uuid: # no update
            var_t.append(var_t[-1])
        else:
            self_val = val[uuid]
            neighbor_val = val[neighbor_id]
            avg_val = (self_val + neighbor_val) / 2
            val[uuid] = avg_val
            val[neighbor_id] = avg_val

            err_t[uuid][-1] = avg_val-init_mean # gossiping agents update error
            err_t[neighbor_id][-1] = avg_val-init_mean

            old_var = var_t[-1]
            diff = (2*(avg_val - mean_0)**2 - (self_val - mean_0)**2 - (neighbor_val - mean_0)**2) / no_agents
            new_var = abs(old_var + diff)
            var_t.append(new_var)

        # move current agent to random neighboring node
        if len(G.graph[pos]): # neighbor exists, i.e., node not isolated
            next_pos_ind = random.randint(0, len(G.graph[pos])-1)
            next_pos = G.graph[pos][next_pos_ind]
            G.agents[pos].remove(uuid)
            G.agents[next_pos].add(uuid)
        else:
            next_pos = pos
        
        # update current agent's clock and insert it in heap
        next_clock = event_time + exp_rv(clock_rate)
        H.insert(uuid, next_clock, next_pos)
    
    # decomission data structures    
    del H
    del G

    # print and plot
    if not multi_sim:
        fig = plt.figure(figsize=(10,15))
        for agent in range(no_agents):
            plt.plot(list(range(simulation_steps+1)), err_t[agent])
        plt.xlim((0, 1000))
        plt.ylim((-0.6, 0.4))
        plt.xlabel('Iterations k', fontsize='large', fontweight='bold')
        plt.ylabel('Agent errors y(k)', fontsize='large', fontweight='bold')
        plt.savefig('./data/oliva/errors.png')
        plt.show()
        plt.close()
    else:
        return (err_t, var_t)


if __name__ == "__main__":
    # parameters (from Oliva et al. (2019))
    no_agents = 10
    init_val = [0.38, 0.29, 0.26, 0.14, -0.03, -0.06, -0.11, -0.16, -0.23, -0.48]
    clock_rate = 1 # Poisson process for asynchronous agent actions
    graph_type = 'Gnm'
    graph_size = 40 # n
    edge_number = 161 # m
    simulation_steps = 1000

    # errors
    run_simulation(no_agents, init_val, clock_rate, graph_type, graph_size, edge_number, simulation_steps)

    # expected errors
    multi_sim = True
    no_sim = 500
    err_all = [[0]*simulation_steps for n in range(no_agents)]
    for sim in range(no_sim):
        (err_t, var_t) = run_simulation(no_agents, init_val, clock_rate, graph_type, graph_size, edge_number, simulation_steps, multi_sim)
        for n in range(no_agents):
            err_all[n] = list(map(add, err_all[n], err_t[n]))    

    fig = plt.figure(figsize=(10,15))
    for n in range(no_agents):
        err_all[n] = list(map(lambda x: x/no_sim, err_all[n])) 
        plt.plot(list(range(simulation_steps)), err_all[n])

    plt.xlim((0, 1000))
    plt.ylim((-0.6, 0.4))
    plt.xlabel('Iterations k', fontsize='large', fontweight='bold')
    plt.ylabel('Agent error means $E[y(k)]$', fontsize='large', fontweight='bold')
    plt.savefig('./data/oliva/expected_errors.png')
    plt.show()
    plt.close()

    # expected variances
    var_all = [0]*simulation_steps
    for sim in range(no_sim):
        (err_t, var_t) = run_simulation(no_agents, init_val, clock_rate, graph_type, graph_size, edge_number, simulation_steps, multi_sim)
        var_all = list(map(add, var_all, var_t))    

    fig = plt.figure(figsize=(10,15))
    var_all = list(map(lambda x: x/no_sim*no_agents, var_all)) 
    plt.plot(list(range(simulation_steps)), var_all)

    plt.xlim((0, 1000))
    plt.ylim((0, 0.8))
    plt.xlabel('Iterations k', fontsize='large', fontweight='bold')
    plt.ylabel('Agent error variancde $E[y(k)^T y(k)]$', fontsize='large', fontweight='bold')
    plt.savefig('./data/oliva/expected_variance.png')
    plt.show()
    plt.close()