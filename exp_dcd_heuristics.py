"""This experiment extends the gossip algorithm for multi-agent systems via random walk proposed in Oliva et al. (2019) by enabling decentralized consensus detection such that the agents can terminate their gossiping once their values converged.
"""
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from lib_helperfcts import *
from lib_heap import Heap
from lib_randomgraph import RandomGraph


def run_simulation(agents, graph, dcd, simulation):
    """Runs the simulation. Uses a heap for the ordering of agent actions according to Poisson process clocks and a random graph as underlying data structures. Moves agents at random and let's them gossip at random, whereby they average their values. Studies how agent values converge to the mean over time.
    
    Agents self-terminate gossiping algorithm once a pre-specified percentage (off_percentage) of agents is switched off; individual agents switch off when having subsequently seen a pre-specified number of other agents (n_off) that are switched off or when a pre-specified number of other agents (n_abs_chg) that with whose values disagree by less than the initial standard deviation times a shrink_factor.
    
    Args:
        agents (tuple): Agent parameters
            no_agents (int): Nuber of agents
            clock_rate (float): Poisson process rate for agent clocks
            comm_limit (int): Number of neighboring agents with which the active agent on a node can average values
        graph (tuple): Graph parameters
            graph_type (string): Gnm
            graph_size (int): Number of nodes n
            edge_probability (float): Probability that an edge forms
        dcd (tuple): Decentralized concensus detection parameters
        simulation (tuple): Simulation parameters
            simulation_time (int): Duration of experiment
            simulation_type (string): single or sweep, affects analysis

    Returns:
        tuple: Sigmas and corresponding times for parameter_type='sweep'
    """
    # unpack
    (no_agents, clock_rate, comm_limit) = agents
    (graph_type, graph_size, edge_probability) = graph
    (heuristic, n_fix_enc, e_abs_chg, n_abs_chg, w_std_est, d_std_est, w_conv_rate, p_conv_rate, w_unknown_env, z_unknown_env, pos_feedback, n_off) = dcd
    (simulation_time, simulation_type) = simulation
    # initialize data structures
    H = Heap(no_agents)
    G = RandomGraph(graph_type, graph_size, edge_probability)
    # 0-mean gaussian noise
    mu = 0
    sigma = 1
    # heuristics
    c_off = [0 for ii in range(no_agents)] # off counter (pos. feedback)
    c_fix_enc = [0 for ii in range(no_agents)] # ecounter counter
    c_abs_chg = [0 for ii in range(no_agents)] # abs_chg counter
    var_vals = [[] for ii in range(no_agents)] # variance (std_est,unknown_env)
    var_hat = [math.inf]*no_agents # estimated variance per agent (std_est)
    std_hat_0 = [0]*no_agents # initially estimated std per agent (unknown_env)
    std_hat = math.inf # tmp estimated std
    rate_logd = [[] for ii in range(no_agents)] # log deltas (conv_rate)
    rate_ks = [[] for ii in range(no_agents)] #k's (conv_rate)
    rate = [math.inf]*no_agents # rate per agent (conv_rate)

    # get agents started
    status = ['on' for ii in range(no_agents)] # agent status [on/off]
    agents_off = 0
    val = []
    for uuid in range(no_agents):
        clock = exp_rv(clock_rate)
        pos = random.randint(0, graph_size-1)
        G.agents[pos].add(uuid)
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
        G.agents[pos].remove(uuid)
        if len(G.agents[pos]): # i.e. not alone on node
            neighbors = list(G.agents[pos])
            no_valid_neighbors = min(len(neighbors), comm_limit)
            random.shuffle(neighbors) # random permutation
            
            avg_val = val[uuid]
            diff = -(val[uuid] - mean_0)**2               
            for ii in range(no_valid_neighbors):
                ## DECENTRALIZED CONSENSUS DETECTION ##
                if pos_feedback == 'on':
                    if status[neighbors[ii]] == 'off':
                        c_off[uuid] += 1
                        no_valid_neighbors -= 1
                        continue
                    else:
                        c_off[uuid] = 0
                        c_off[neighbors[ii]] = 0
                
                if heuristic == 'fix_enc':
                    c_fix_enc[uuid] += 1

                elif heuristic == 'abs_chg':
                    if abs(val[uuid] - val[neighbors[ii]]) < e_abs_chg:
                        c_abs_chg[uuid] += 1
                        c_abs_chg[neighbors[ii]] += 1
                    else:
                        c_abs_chg[uuid] = 0
                        c_abs_chg[neighbors[ii]] = 0
                
                elif heuristic == 'std_est':
                    var_vals[uuid].append(val[neighbors[ii]])
                    if len(var_vals[uuid]) > w_std_est:
                        var_oldest = var_vals[uuid].pop(0)
                        var_newest = var_vals[uuid][-1]
                        update = (-(var_oldest - mean_0)**2 + (var_newest - mean_0)**2) / (w_std_est - 1.5)
                        var_hat[uuid] = abs((var_hat[uuid] + update))
                        std_hat = var_hat[uuid] ** 0.5
                    elif len(var_vals[uuid]) ==  w_std_est:
                        var_hat[uuid] = sum([((x - mean_0) ** 2) for x in var_vals[uuid]]) / (w_std_est - 1.5)
                        std_hat = var_hat[uuid] ** 0.5                        

                elif heuristic == 'conv_rate':
                    rate_logd[uuid].append(math.log(0.5*max(abs(val[uuid] - val[neighbors[ii]]), p_conv_rate)))
                    rate_ks[uuid].append(event_time)
                    if len(rate_logd[uuid]) > w_conv_rate:
                        rate_logd[uuid].pop(0)
                        rate_ks[uuid].pop(0)
                        x = np.asarray(rate_ks[uuid])
                        y = np.asarray(rate_logd[uuid])
                        (b, a) = np.polyfit(x, y, 1)
                        rate[uuid] = -(b * math.exp(a+b*rate_ks[uuid][-1]))

                elif heuristic == 'unknown_env':
                    var_vals[uuid].append(val[neighbors[ii]])
                    if len(var_vals[uuid]) > w_unknown_env: # current estimate
                        var_oldest = var_vals[uuid].pop(0)
                        var_newest = var_vals[uuid][-1]
                        update = (-(var_oldest - mean_0)**2 + (var_newest - mean_0)**2) / (w_std_est - 1.5)
                        var_hat[uuid] = abs((var_hat[uuid] + update))
                        std_hat = var_hat[uuid] ** 0.5
                    elif len(var_vals[uuid]) ==  w_std_est: # initial estimate
                        var_hat[uuid] = sum([((x - mean_0) ** 2) for x in var_vals[uuid]]) / (w_std_est - 1.5)
                        std_hat = var_hat[uuid] ** 0.5
                        std_hat_0[uuid] = std_hat

                elif heuristic == 'none':
                    pass

                else:
                    print('Unavailable error heuristic. Please choose from none, fix_enc, abs_chg, std_est, conv_rate, or unknown_env.')
                    return
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
        if len(G.graph[pos]): # neighbor exists, i.e., node not isolated
            next_pos_ind = random.randint(0, len(G.graph[pos])-1)
            next_pos = G.graph[pos][next_pos_ind]
            G.agents[next_pos].add(uuid)
        else: # stay on your sad island
            next_pos = pos
            G.agents[next_pos].add(uuid)

        ## DECENTRALIZED CONSENSUS DETECTION ##
        agents_off_val.append(agents_off)
        agents_off_t.append(event_time)

        if c_off[uuid] >= n_off or c_fix_enc[uuid] > n_fix_enc or c_abs_chg[uuid] >= n_abs_chg or std_hat < d_std_est or std_hat < z_unknown_env*std_hat_0[uuid] or rate[uuid] < p_conv_rate:

            status[uuid] = 'off' # switch off
            agents_off += 1
            if agents_off >= no_agents: # terminate!
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

    # decomission data structures    
    del H
    del G

    # print and plot
    if simulation_type == 'single':
        analyze_single(val_0, val, mean_0, std_rt_val, std_rt_t, simulation_time, no_agents, agents_off_val, agents_off_t, step_count)

    # or return results
    elif simulation_type == 'sweep':
        print('For {}, total number of edges travelled by all agents is {}'.format(heuristic, step_count))
        return (std_rt_val, std_rt_t, step_count_rt, agents_off_t, agents_off_val)

if __name__ == "__main__":
    # agents
    no_agents = 200
    clock_rate = 1 # Poisson process for asynchronous agent actions
    comm_limit = 1 # no of neighbors with whom a value can be averaged
    agents = (no_agents, clock_rate, comm_limit)

    # graph
    graph_type = 'Gnp'
    graph_size = 1000 # n
    edge_probability = 0.5 # p
    graph = (graph_type, graph_size, edge_probability)

    # decentralized consensus detection (DCD)
    heuristic = 'conv_rate'
    n_fix_enc = 5 # no of encounters before switching off
    e_abs_chg = 0.05 # convergence when delta_val < e_abs_chg
    n_abs_chg = 5 # agent switches off when having subsequently seen n_abs_chg agents with delta_val < e_abs_chg
    w_std_est = 5 # sliding window size for continuous estimation of standard deviation
    d_std_est = 0.01 # agent switches off when std_est < d_std_est
    w_conv_rate = 10 # sliding window size for continuous estimation of convergence rate
    p_conv_rate = 0.0001 # agent switches off when conv_rate < p_conv_rate
    w_unknown_env = 10
    z_unknown_env = 0.01

    pos_feedback = 'off'
    n_off = 5 # agent switches off when having subsequently seen n_off other switched off agents
    dcd = (heuristic, n_fix_enc, e_abs_chg, n_abs_chg, w_std_est, d_std_est, w_conv_rate, p_conv_rate, w_unknown_env, z_unknown_env, pos_feedback, n_off)

    # simulation
    simulation_time = 150
    simulation_type = 'single'
    simulation = (simulation_time, simulation_type)

    run_simulation(agents, graph, dcd, simulation)