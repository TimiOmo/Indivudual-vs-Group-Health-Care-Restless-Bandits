import numpy as np

def random_policy(env):
    """
    Randomly select actions for each arm in the environment, constrained by the budget.
    """
    num_arms = env.num_arms
    budget = env.budget
    
    # Generate a random binary action for each arm
    action = np.zeros(num_arms, dtype=int)
    
    # Randomly choose `budget` number of arms to treat (set to 1)
    selected_arms = np.random.choice(num_arms, budget, replace=False)
    action[selected_arms] = 1
    
    return action



def whittle_policy(env):
    """
    Calls the whittle policy from the environment.
    """
    return env.compute_whittle_actions()  # Already implemented in the simulator
