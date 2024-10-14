import numpy as np

def random_policy(env):
    """
    Randomly select actions for each arm in the environment.
    """
    num_arms = env.num_arms
    action = np.random.randint(0, 2, size=num_arms)  # Random 0 or 1 for each arm
    return action

def whittle_policy(env):
    """
    Calls the whittle policy from the environment.
    """
    return env.compute_whittle_actions()  # Already implemented in the simulator
