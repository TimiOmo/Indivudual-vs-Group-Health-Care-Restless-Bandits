import gym
import numpy as np
import torch
from gym import spaces
from compute_whittle import compute_whittle
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transition_NN import TransitionNN
from models.transition_NN import TransitionNN  # Correct file name

import torch
model = TransitionNN(input_dim=4, hidden_dim=16, output_dim=4)
model.load_state_dict(torch.load('models/trained_model.pth'))
print("Model weights loaded successfully!")


class RMABSimulator(gym.Env):
    """
    Restless Multi-Armed Bandit Simulator for Healthcare Resource Allocation.

    This environment models a healthcare scenario where you have limited resources to allocate among a group of patients.
    Patients can either be treated as individuals or grouped by characteristics.
    """
    discount_factor = 0.95
    subsidy = 0.35

    def __init__(self, num_arms=10, budget=3, grouping=False, subsidy=0.4, discount_factor=0.75, model_path=None):
        """
        Initialize the RMABSimulator environment.

        Parameters:
        - num_arms: Number of patients or groups.
        - budget: Maximum number of treatments available at each step.
        - grouping: Whether patients are treated as groups or individuals.
        - subsidy: Baseline subsidy for treatment decisions.
        - discount_factor: Discount factor for future rewards.
        - model_path: Path to the pretrained neural network model (optional).
        """
        # Parameters
        self.num_arms = num_arms
        self.budget = budget
        self.grouping = grouping
        self.subsidy = subsidy
        self.discount_factor = discount_factor
        self.groups = {}

        # Load the neural network model if provided
        self.model = None
        if model_path:
            self.model = self.load_model(model_path)

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

    def load_model(self, model_path):
        """
        Load the pretrained neural network model from the specified path.
        """
        model = TransitionNN(input_dim=4, hidden_dim=16, output_dim=4)  # Adjust hidden_dim if needed
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set model to evaluation mode
        return model

    def reset(self):
        """
        Reset the environment to an initial state with features that are distributed 
        to better reflect real-world healthcare scenarios.
        """
        self.state = np.random.choice([0, 1], size=self.num_arms)  # Random initial health state
        self.features = np.zeros((self.num_arms, 4))  # Define a feature array with 4 features per arm

        for i in range(self.num_arms):
            # Age: Skewed distribution towards older individuals
            age = np.random.beta(2, 5)  # Beta distribution with more older individuals (values closer to 1)
            self.features[i, 0] = age

            # Sex: Assuming equal probability, 50% male, 50% female
            self.features[i, 1] = np.random.choice([0, 1])

            # Race: Adjust probabilities based on real-world demographics (example for the U.S.)
            race = np.random.choice(
                [0, 1, 2, 3, 4], 
                p=[0.6, 0.13, 0.06, 0.18, 0.03]
            )  # Example probabilities: White (60%), Black (13%), Asian (6%), Hispanic (18%), Other/Mixed (3%)
            self.features[i, 2] = race

            # Pre-existing Conditions: Higher probability for older people
            pre_existing_prob = 0.1 + 0.8 * age  # Increase likelihood of pre-existing condition with age
            self.features[i, 3] = np.random.binomial(1, pre_existing_prob)  # Binary: 0 (no) or 1 (yes)

        if self.grouping:
            self.group_patients()
        
        return self.state

    def adjust_probabilities(self, features):
        """
        Adjust transition probabilities based on patient features using the neural network model.
        If the model is not provided, fall back to the default manual logic.
        """
        if self.model:
            # Use the neural network model to predict probabilities
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            transition_matrix = self.model(features_tensor).squeeze(0).detach().numpy()  # Remove batch dimension
            recover_prob = transition_matrix[0, 1]  # Probability of recovering (0 -> 1)
            deteriorate_prob = transition_matrix[1, 0]  # Probability of deteriorating (1 -> 0)
        else:
            # Fallback manual probability adjustment
            recover_prob = self.base_recover_prob
            deteriorate_prob = self.base_deteriorate_prob

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
                state = self.state[self.groups[group_key][0]]  # Representative state
                whittle_index = compute_whittle(
                    [avg_recover_prob, avg_deteriorate_prob], state, self.discount_factor, self.subsidy
                )
                whittle_indices.append((whittle_index, group_key))
        else:
            # Compute Whittle index for individual patients
            for i in range(self.num_arms):
                recover_prob, deteriorate_prob = self.adjust_probabilities(self.features[i])
                state = self.state[i]
                whittle_index = compute_whittle(
                    [recover_prob, deteriorate_prob], state, self.discount_factor, self.subsidy
                )
                whittle_indices.append((whittle_index, i))

        # Sort by Whittle index in descending order (treat the highest index)
        whittle_indices.sort(reverse=True, key=lambda x: x[0])

        # Select top `budget` number of treatments
        action = np.zeros(self.num_arms, dtype=int)

        if self.grouping:
            # Treat the top-ranked groups
            for _, group_key in whittle_indices[:self.budget]:
                for i in self.groups[group_key]:
                    action[i] = 1
        else:
            # Treat the top-ranked individuals
            for _, patient_index in whittle_indices[:self.budget]:
                action[patient_index] = 1

        return action

    def step(self, action):
        """
        Apply an action to the environment and transition to the next state.
        Action is a binary vector indicating which patients are being treated.

        Parameters:
        action (np.array): A binary vector of size `num_arms` indicating which patients receive treatment.

        Returns:
        - next_state: The updated state after taking the action.
        - reward: The total reward obtained after taking the action.
        - healthy_percentage: Percentage of healthy individuals after this step.
        - done: Whether the episode has ended (for now, always False).
        - info: Additional information (for debugging).
        """
        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Initialize variables for the total reward and next state
        reward = 0
        next_state = self.state.copy()

        # Loop through each arm/patient to apply actions and calculate rewards
        for i in range(self.num_arms):
            # Extract the patient's features and adjust probabilities based on features
            features = self.features[i]
            recover_prob, deteriorate_prob = self.adjust_probabilities(features)

            if action[i] == 1:  # If the arm is being treated
                if self.state[i] == 0:  # If the patient is currently unhealthy
                    # Treating has a chance to make the patient healthy
                    if np.random.rand() < recover_prob:
                        next_state[i] = 1  # Patient becomes healthy
                        reward += 1       # Reward for successful recovery
                else:
                    # Patient is already healthy, treat to maintain health (optional reward)
                    reward += 1  # Reward for maintaining health
            else:  # If the arm is not being treated
                if self.state[i] == 1:  # If the patient is currently healthy
                    # Patient may become unhealthy without treatment
                    if np.random.rand() < deteriorate_prob:
                        next_state[i] = 0  # Patient becomes unhealthy

        # Update the environment state
        self.state = next_state

        # Calculate the percentage of healthy individuals (state == 1)
        healthy_percentage = np.mean(self.state) * 100

        # Determine if the episode is done (for now, always False)
        done = False

        # Additional information for debugging or analysis
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

if __name__ == "__main__":
    # Example usage of the RMABSimulator with a pretrained model
    env = RMABSimulator(num_arms=5, budget=2, grouping=False, model_path="models/trained_model.pth")

    state = env.reset()
    print(f"Initial State: {state}")

    # Example step
    action = env.compute_whittle_actions()
    next_state, reward, healthy_percentage, done, info = env.step(action)
    print(f"Next State: {next_state}, Reward: {reward}, Healthy Percentage: {healthy_percentage}")
