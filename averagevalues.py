import math
import random

class AverageValues():
    def __init__(self, no_agents, mu, sigma):
        self.agent = [random.gauss(mu,sigma) for n in range(no_agents)]