import gym
import numpy as np
from gym import spaces
from compute_whittle import compute_whittle

class RMABSimulator(gym.Env):
    """
    Restless Multi-Armed Bandit Simulator for Healthcare Resource Allocation.

    This environment models a healthcare scenario where you have limited resources to allocate among a group of patients.
    Patients can either be treated as individuals or grouped by characteristics.
    """
    # will be a transition matrix
    transitions = ([0.3, 0.7], # Probability unhealthy patient stay  unhealthy become healthy without treatment
                   [0.1, 0.9] # Probability healthy patient becomes unhealthy or stay healthy with  treatment
                   )
    state = 0
    discount_factor = 0.9
    subsidy = 0.1

    def __init__(self, num_arms=10, budget=3, grouping=False):
        # Parameters
        self.num_arms = num_arms  # This is the number of patients (or groups)
        self.budget = budget  # The number of treatments available
        self.grouping = grouping  # Wheather arms are groups or people

        # Transition Probabilities
        self.prob_recover = 0.7  # Probability of recovering to a healthy state when treated
        self.prob_deteriorate = 0.1  # Probability of deteriorating to an unhealthy state when untreated

        # State Space and Action Space
        self.state_space = spaces.Discrete(2)  # Assuming binary state (0 or 1)
        self.observation_space = spaces.MultiDiscrete([2] * num_arms)
        self.action_space = spaces.MultiBinary(num_arms)  # Binary action for each arm

        # Internal State
        self.state = None
        self.reset()

    def reset(self):
        """
        Reset the environment to an initial state.
        Each patient starts in a random state (0 or 1).
        """
        self.state = np.random.choice([0, 1], size=self.num_arms)
        return self.state

    def compute_whittle_actions(self):
        """
        Compute the Whittle Index for each patient and decide which patients to treat.
        """
        whittle_indices = []
        for i in range(self.num_arms):
            state = self.state[i]
            whittle_index = compute_whittle(self.transitions, state, self.discount_factor, self.subsidy)
            whittle_indices.append((whittle_index, i))
    
        # Sort arms by Whittle Index in descending order (treat those with the highest index)
        whittle_indices.sort(reverse=True, key=lambda x: x[0])

        # Select top `budget` patients to treat
        action = np.zeros(self.num_arms, dtype=int)
        for _, patient_index in whittle_indices[:self.budget]:
            action[patient_index] = 1

        return action

    # Then the step function can remain the same, but now actions are guided by Whittle Index
    def step(self, action):
        """
        Apply an action to the environment and transition to the next state.
        Action is a binary vector indicating which patients are being treated.

        Parameters:
        action (np.array): A binary vector of size `num_arms` indicating which patients receive treatment.

        Returns:
        - next_state: The updated state after taking the action.
        - reward: The reward obtained after taking the action.
        - done: Whether the episode has ended.
        - info: Additional information (for debugging).
        """
        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Calculate reward and transition to next state
        reward = 0
        next_state = self.state.copy()

        for i in range(self.num_arms):
            if action[i] == 1:  # If the arm is being treated
                if self.state[i] == 0:
                    # Treatment causes recovery with 70% probability
                    if np.random.rand() < self.prob_recover:
                        next_state[i] = 1
                        reward += 1  # Reward for successful treatment
            else:
                # Non-treated patients might deteriorate with 10% probability
                if self.state[i] == 1 and np.random.rand() < self.prob_deteriorate:
                    next_state[i] = 0

        # Update internal state
        self.state = next_state

         # Calculate the percentage of healthy individuals (state == 1)
        healthy_percentage = np.mean(self.state) * 100

        # Determine if the episode is done (for now, we assume it's never-ending)
        done = False

        # Additional information (can be used for debugging)
        info = {}

        return next_state, reward, healthy_percentage, done, info


    def render(self, mode='human'):
        """
        Render the environment (optional).
        This could be a simple printout or a more complex graphical representation.
        """
        print(f"Current State: {self.state}")

    def close(self):
        """
        Cleanup any resources if needed (optional).
        """
        pass

    def compute_group_state(self):
        """
        Compute the average transition probability of the group.
        If `grouping` is True, consider group-level characteristics.
        Otherwise, compute based on individual states.
        """
        total_prob = 0
        if self.grouping:
            # Logic for grouping (can be based on patient characteristics, etc.)
            group_avg_prob = np.mean(self.state)  # Example of how grouping can be simplified
        else:
            # Compute transition probabilities based on individual states
            for i in range(self.num_arms):
                if self.state[i] == 0:  # Unhealthy
                    # Use the probability of transitioning from unhealthy to healthy
                    total_prob += self.transitions[0][1]  # Transition from state 0 to state 1
                else:  # Healthy
                    # Use the probability of transitioning from healthy to unhealthy
                    total_prob += self.transitions[1][0]  # Transition from state 1 to state 0
        
            # Calculate the average probability across the group
            group_avg_prob = total_prob / self.num_arms

        return group_avg_prob




if __name__ == "__main__":
    # Example usage of the RMABSimulator
    env = RMABSimulator(num_arms=5, budget=2, grouping=False)
    state = env.reset()
    print(f"Initial State: {state}")

    # Example step
    action = np.array([1, 0, 1, 0, 0])  # Example action where we treat patients 0 and 2
    next_state, reward, done, info = env.step(action)
    print(f"Next State: {next_state}, Reward: {reward}")
