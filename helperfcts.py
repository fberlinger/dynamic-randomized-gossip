import math
import random

def exp_rv(param):
    """Draw a uniform random number between 0 and 1 and returns an exponentially distributed random number with parameter param.
    
    Args:
        param (float): parameter of exponentially distributed random number
    
    Returns:
        float: exponentially distributed random number
    """
    x = random.random()
    return -math.log(1-x)/param