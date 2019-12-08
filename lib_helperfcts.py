"""This library provides helperfunctions for generating exponentially distributed random variables and analyzing and visualizing simulation results.
"""
import math
import random
import matplotlib
import matplotlib.pyplot as plt

def exp_rv(param):
    """Draw a uniform random number between 0 and 1 and returns an exponentially distributed random number with parameter param.
    
    Args:
        param (float): Parameter of exponentially distributed random number
    
    Returns:
        float: Exponentially distributed random number
    """
    x = random.random()
    return -math.log(1-x)/param

def analyze_single(val_0, val, mean_0, std_rt_val, std_rt_t, simulation_time, no_agents, agents_off_val=0, agents_off_t=0, step_count=0):
    """Analyzes a single simulation experiment and visualized initial and final distributions of agent values as well as convergence over time.
    
    Args:
        val_0 (list of floats): Initial agent values
        val (list of floats): Agent values
        mean_0 (float): Initial mean of agent values
        std_rt_val (list of floats): Standard deviation of agent values over time
        std_rt_t (list of floats): Corresponding agent clock ticks times
        simulation_time (int): Duration of simulation
        no_agents (int): Number of agents
        agents_off_val (list of int, optional): Number of switched off agents
        agents_off_t (list of float, optional): Corresponding switch off times
        step_count (int, optional): Number of edges travelled by all agents
    """
    # print
    mean = sum(val) / no_agents
    print('Initial mean and standard deviation in agent values are ({:.3f}, {:.3f})'.format(mean_0, std_rt_val[0]))
    print('Final mean and standard deviation in agent values are ({:.3f}, {:.3f})'.format(mean, std_rt_val[-1]))

    # plot number of switched off agents over time if decentralized consensus detection is active
    if agents_off_val:
        no_plots = 4
        print('Total number of nodes travelled by all agents is {}'.format(step_count))
    else:
        no_plots = 3

    # plot
    lim = math.ceil(max(max(val_0), abs(min(val_0)))) # histogram limits
    fig, axs = plt.subplots(no_plots, 1, figsize=(10,15))
    axs[0].plot(std_rt_t, std_rt_val)
    axs[0].set_xlabel('Simulation time [s]', fontsize='large', fontweight='bold')
    axs[0].set_ylabel('Standard deviation', fontsize='large', fontweight='bold')
    axs[0].grid('k')

    axs[1].hist(val_0, bins=25, range=[-lim, lim])
    axs[1].set_xlabel('Agent values', fontsize='large', fontweight='bold')
    axs[1].set_ylabel('No of agents', fontsize='large', fontweight='bold')
    axs[1].text(0.9, 0.9, 't = 0s', ha='center', va='center', transform=axs[1].transAxes, fontsize=15)
    axs[1].grid('k')

    axs[2].hist(val, bins=25, range=[-lim, lim])
    axs[2].set_xlabel('Agent values', fontsize='large', fontweight='bold')
    axs[2].set_ylabel('No of agents', fontsize='large', fontweight='bold')
    axs[2].text(0.9, 0.9, 't = {}s'.format(simulation_time), ha='center', va='center', transform=axs[2].transAxes, fontsize=15)
    axs[2].grid('k')

    if agents_off_val:
        axs[3].plot(agents_off_t, agents_off_val)
        axs[3].set_xlabel('Simulation time [s]', fontsize='large', fontweight='bold')
        axs[3].set_ylabel('No of switched off agents', fontsize='large', fontweight='bold')
        axs[3].grid('k')

    plt.show()