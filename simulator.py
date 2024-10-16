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
    # state = 0
    discount_factor = 0.75
    subsidy = 0.4

    def __init__(self, num_arms=10, budget=3, grouping=False, subsidy=0.4, discount_factor=0.75):
        # Parameters
        self.num_arms = num_arms  # This is the number of patients (or groups)
        self.budget = budget  # The number of treatments available
        self.grouping = grouping  # Whether arms are groups or people
        self.subsidy = subsidy  # Set subsidy value from CLI
        self.discount_factor = discount_factor  # Set discount factor from CLI
        self.groups = {}

        # Transition Probabilities
        self.base_recover_prob = 0.9
        self.base_deteriorate_prob = 0.1

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
        features are randomly generated as well
        """
        self.state = np.random.choice([0, 1], size=self.num_arms)
        self.features = np.zeros((self.num_arms, 4))  # Define a feature array with 4 features per arm


        for i in range(self.num_arms):
            # Age: Normalize between 0 and 1 (e.g., 0 = young, 1 = old)
            self.features[i, 0] = np.random.uniform(0, 1)
            # Sex: Binary (0 = female, 1 = male)
            self.features[i, 1] = np.random.choice([0, 1])
            # Race: 5 categories encoded as numbers (e.g., 0 = White, 1 = Black, 2 = Asian, 3 = Hispanic, 4 = Other/Mixed)
            self.features[i, 2] = np.random.choice([0, 1, 2, 3, 4])
            # Pre-existing conditions: Binary (0 = no, 1 = yes)
            self.features[i, 3] = np.random.uniform(0, 1)

        if self.grouping:
            self.group_patients()
        
        return self.state
    
    def adjust_probabilities(self, features, base_recover_prob=0.9, base_deteriorate_prob=0.05):
        """
        Adjust transition probabilities based on patient features using weighted sum approach.
        
        Arguments:
        - features: An array of features [age, gender, race, pre-existing condition (severity)].
        - base_recover_prob: The base recovery probability before adjustments.
        - base_deteriorate_prob: The base deterioration probability before adjustments.

        Returns:
        - recover_prob: Adjusted recovery probability.
        - deteriorate_prob: Adjusted deterioration probability.
        """

        # Feature weights
        age_weight = -0.2  # Older age decreases recovery, increases deterioration
        sex_weight = 0.05  # A small amount for sex
        race_weights = [0, -0.05, -0.1, -0.15, -0.1]  # Weights for each race, adjust to represent health disparities
        pre_existing_weight = -0.3  # Severe pre-existing conditions significantly reduce recovery

        # Extract features (age, gender, race, pre-existing condition severity)
        age, sex, race, pre_existing_condition = features

        # Adjust recovery probability based on features
        recover_prob = base_recover_prob
        recover_prob += age_weight * age  # Age negatively affects recovery
        recover_prob += sex_weight * sex  # Gender affects recovery slightly
        recover_prob += race_weights[int(race)]  # Race-based health disparity
        recover_prob += pre_existing_weight * pre_existing_condition  # Pre-existing conditions lower recovery

        # Adjust deterioration probability based on features
        deteriorate_prob = base_deteriorate_prob
        deteriorate_prob -= age_weight * age  # Older people deteriorate faster
        deteriorate_prob -= sex_weight * sex  # Minor gender difference
        deteriorate_prob -= race_weights[int(race)]  # Race-based health disparity
        deteriorate_prob -= pre_existing_weight * pre_existing_condition  # Pre-existing conditions worsen deterioration

        # Ensure probabilities remain within valid bounds [0, 1]
        recover_prob = np.clip(recover_prob, 0, 1)
        deteriorate_prob = np.clip(deteriorate_prob, 0, 1)

        return recover_prob, deteriorate_prob   

    def compute_whittle_actions(self):
        """
        Compute the Whittle Index for each patient or group and decide which patients to treat.
        """
        whittle_indices = []

        if self.grouping:
            # Compute Whittle index for groups
            for group_key in self.groups:
                avg_recover_prob, avg_deteriorate_prob = self.adjust_group_probabilities(group_key)
                # Use the state of the first patient in the group as a representative
                state = self.state[self.groups[group_key][0]]
                whittle_index = compute_whittle([avg_recover_prob, avg_deteriorate_prob], state, self.discount_factor, self.subsidy)
                whittle_indices.append((whittle_index, group_key))
        else:
            # Compute Whittle index for individual patients
            for i in range(self.num_arms):
                recover_prob, deteriorate_prob = self.adjust_probabilities(self.features[i])
                state = self.state[i]
                whittle_index = compute_whittle([recover_prob, deteriorate_prob], state, self.discount_factor, self.subsidy)
                whittle_indices.append((whittle_index, i))

        # Sort by Whittle index in descending order (treat the highest index)
        whittle_indices.sort(reverse=True, key=lambda x: x[0])

        # Select top `budget` number of treatments
        action = np.zeros(self.num_arms, dtype=int)

        if self.grouping:
            # Treat the top-ranked groups
            for _, group_key in whittle_indices[:self.budget]:
                # Treat all individuals in the group
                for i in self.groups[group_key]:
                    action[i] = 1
        else:
            # Treat the top-ranked individuals
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
            features = self.features[i]  # Extract the features for arm i
            recover_prob, deteriorate_prob = self.adjust_probabilities(self.features[i])

        if action[i] == 1:  # If the arm is being treated
            if self.state[i] == 0 and np.random.rand() < recover_prob:
                next_state[i] = 1
                reward += 1  # Reward for successful treatment
        else:
            if self.state[i] == 1 and np.random.rand() < deteriorate_prob:
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
    
    def group_patients(self):
        """
        Group patients based on their features (e.g., age, sex, race, and pre-existing conditions).
        """
        self.groups = {}
        for i in range(self.num_arms):
            # Example: Group by Age and Pre-existing conditions
            age_group = "young" if self.features[i, 0] < 0.33 else "middle-aged" if self.features[i, 0] < 0.66 else "old"
            pre_condition_group = "low" if self.features[i, 3] < 0.33 else "moderate" if self.features[i, 3] < 0.66 else "high"

            # Use a tuple (age_group, pre_condition_group) as the group key
            group_key = (age_group, pre_condition_group)

            if group_key not in self.groups:
                self.groups[group_key] = []
            self.groups[group_key].append(i)  # Add patient index to the group
    
    def adjust_group_probabilities(self, group_key):
        """
        Adjust transition probabilities for a group based on average feature values.
        """
        group = self.groups[group_key]
        total_recover_prob = 0
        total_deteriorate_prob = 0

        # Average over all patients in the group
        for i in group:
            recover_prob, deteriorate_prob = self.adjust_probabilities(self.features[i])
            total_recover_prob += recover_prob
            total_deteriorate_prob += deteriorate_prob

        # Return the average probabilities for the group
        avg_recover_prob = total_recover_prob / len(group)
        avg_deteriorate_prob = total_deteriorate_prob / len(group)

        return avg_recover_prob, avg_deteriorate_prob
    
    def calculate_average_probabilities(self):
        """
        Calculate the average recovery and deterioration probabilities for all arms (patients).
        
        Returns:
        - avg_recover_prob: The average recovery probability across all arms.
        - avg_deteriorate_prob: The average deterioration probability across all arms.
        """
        total_recover_prob = 0
        total_deteriorate_prob = 0

        # Loop through each arm (patient)
        for i in range(self.num_arms):
            features = self.features[i]
            recover_prob, deteriorate_prob = self.adjust_probabilities(features)
            
            total_recover_prob += recover_prob
            total_deteriorate_prob += deteriorate_prob
        
        # Compute the average by dividing by the number of arms
        avg_recover_prob = total_recover_prob / self.num_arms
        avg_deteriorate_prob = total_deteriorate_prob / self.num_arms
        
        return avg_recover_prob, avg_deteriorate_prob





if __name__ == "__main__":
    # Example usage of the RMABSimulator
    env = RMABSimulator(num_arms=5, budget=2, grouping=False)
    state = env.reset()
    print(f"Initial State: {state}")

    # Example step
    action = np.array([1, 0, 1, 0, 0])  # Example action where we treat patients 0 and 2
    next_state, reward, done, info = env.step(action)
    print(f"Next State: {next_state}, Reward: {reward}")

    