"""This experiment extends the gossip algorithm for multi-agent systems via random walk proposed in Oliva et al. (2019) by enabling decentralized consensus detection such that the agents can terminate their gossiping once their values converged.
"""
import math
import random
import matplotlib
import matplotlib.pyplot as plt

from lib_helperfcts import *
from lib_heap import Heap
from lib_randomgraph import RandomGraph


def run_simulation(no_agents, graph_type, graph_size, edge_probability, mu, sigma, clock_rate, simulation_time, parameter_type, switch_off, comm_limit=1):
    """Runs the simulation. Uses a heap for the ordering of agent actions according to Poisson process clocks and a random graph as underlying data structures. Moves agents at random and let's them gossip at random, whereby they average their values. Studies how agent values converge to the mean over time.

    Agents self-terminate gossiping algorithm once a pre-specified percentage (off_percentage) of agents is switched off; individual agents switch off when having subsequently seen a pre-specified number of other agents (n_off) that are switched off or when a pre-specified number of other agents (n_conv) that with whose values disagree by less than the initial standard deviation times a shrink_factor.
    
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
        switch_off (tuple of floats): Contains termination conditions for decentralized consensus detection
        comm_limit (int): Number of neighboring agents with which the active agent on a node can average values
    
    Returns:
        tuple: Sigmas and corresponding times for parameter_type='sweep'
    """
    # unpack switch off conditions
    (off_percentage, shrink_factor, n_off, n_conv) = switch_off
    # initialize data structures
    H = Heap(no_agents)
    Gnp = RandomGraph(graph_type, graph_size, edge_probability)

    # get agents started
    status = ['on' for ii in range(no_agents)] # agent status [on/off]
    agents_off = 0
    off_counter = [0 for ii in range(no_agents)]
    conv_counter = [0 for ii in range(no_agents)]
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
    agents_off_val = [] # number of switched off agents
    agents_off_t = [] # corresponding switch off times

    step_count = 0 # how many edges all of the agents travelled
    step_count_rt = [0]

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
                ## DECENTRALIZED CONSENSUS DETECTION ##
                if status[neighbors[ii]] == 'off':
                    off_counter[uuid] += 1
                    no_valid_neighbors -= 1
                    continue
                else:
                    off_counter[uuid] = 0
                    off_counter[neighbors[ii]] = 0
                if val[uuid] - val[neighbors[ii]] < sigma * shrink_factor:
                    conv_counter[uuid] += 1
                    conv_counter[neighbors[ii]] += 1
                else:
                    conv_counter[uuid] = 0
                    conv_counter[neighbors[ii]] = 0
                #######################################
                avg_val += val[neighbors[ii]]
                diff -= (val[neighbors[ii]] - mean_0)**2
            avg_val /= (no_valid_neighbors + 1)
            diff += (no_valid_neighbors + 1) * (avg_val - mean_0)**2
            diff /= no_agents

            val[uuid] = avg_val
            for ii in range(no_valid_neighbors):
                ## DECENTRALIZED CONSENSUS DETECTION ##
                if status[neighbors[ii]] == 'off':
                    continue
                #######################################    
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

        ## DECENTRALIZED CONSENSUS DETECTION ##
        agents_off_val.append(agents_off)
        agents_off_t.append(event_time)
        if off_counter[uuid] >= n_off or conv_counter[uuid] >= n_conv:
            status[uuid] = 'off' # switch off
            agents_off += 1
            if agents_off >= off_percentage*no_agents: # terminate!
                simulation_time = round(event_time)
                step_count_rt.append(step_count)
                break
            else:
                step_count_rt.append(step_count)
                continue # never goes back on the heap, never goes back in action
        #######################################
        
        # update current agent's clock and insert it in heap
        next_clock = event_time + exp_rv(clock_rate)
        H.insert(uuid, next_clock, next_pos)

        step_count += 1
        step_count_rt.append(step_count)


    # print and plot
    if parameter_type == 'single':
        analyze_single(val_0, val, mean_0, std_rt_val, std_rt_t, simulation_time, no_agents, agents_off_val, agents_off_t, step_count)

    # or return results
    elif parameter_type == 'sweep':
        return (std_rt_val, std_rt_t, step_count_rt)

if __name__ == "__main__":
    no_agents = 200
    mu = 0 # average value
    sigma = 10 # gaussian noise
    graph_type = 'Gnp'
    graph_size = 1000 # n
    edge_probability = 0.5 # p
    clock_rate = 1 # Poisson process for asynchronous agent actions
    simulation_time = 1000
    parameter_type = 'single'
    comm_limit = 1

    # decentralized consensus detection parameters
    off_percentage = 0.8 # abort simulation when off_percentage of all agents are off
    shrink_factor = 1/100 # agent assumes convergence when delta val < sigma * shrink_factor
    n_off = math.inf # agent switches off when having subsequently seen n_off other switched off agents
    n_conv = math.inf # agent switches off when having subsequently seen n_conv agents with delta val < sigma * shrink_factor
    switch_off = (off_percentage, shrink_factor, n_off, n_conv)
    
    # decentralized consensus detection INACTIVE
    run_simulation(no_agents, graph_type, graph_size, edge_probability, mu, sigma, clock_rate, simulation_time, parameter_type, switch_off)

    n_off = 5 # agent switches off when having subsequently seen n_off other switched off agents
    n_conv = 5 # agent switches off when having subsequently seen n_conv agents with delta val < sigma * shrink_factor
    switch_off = (off_percentage, shrink_factor, n_off, n_conv)
    
    # decentralized consensus detection ACTIVE
    run_simulation(no_agents, graph_type, graph_size, edge_probability, mu, sigma, clock_rate, simulation_time, parameter_type, switch_off)