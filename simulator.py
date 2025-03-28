import gym
import numpy as np
import json
import torch
from gym import spaces
from compute_whittle import compute_whittle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Define the neural network architecture (must match training configuration)
class TransitionNN(torch.nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=4):
        super(TransitionNN, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.Sigmoid()  # ensures outputs in [0,1]
        )

    def forward(self, x):
        return self.net(x)

class RMABSimulator(gym.Env):
    """
    Restless Multi-Armed Bandit Simulator for Healthcare Resource Allocation.

    This environment models a healthcare scenario where you have limited resources
    to allocate among a group of patients. Patients can either be treated as individuals
    or grouped by characteristics. Transitions use a 2x2x2 approach.
    
    New Options:
      - use_trained_model: When True, loads a trained NN to generate transition probabilities.
      - model_path & hidden_dim: Parameters for the trained model.
      - data_file: If provided, load synthetic features and transitions from the file.
    """

    def __init__(self, num_arms=10, budget=3, grouping=False,
                 subsidy=0.4, discount_factor=0.75,
                 use_trained_model=False, model_path="model/trained_model_no_val.pth", hidden_dim=16,
                 data_file="model/synthetic_2x2x2.json"):
        super(RMABSimulator, self).__init__()
        
        # Core parameters
        self.num_arms = num_arms        # Number of patients (or groups)
        self.budget = budget            # Number of treatments available
        self.grouping = grouping        # Whether arms are grouped
        self.subsidy = subsidy          # For external Whittle computations
        self.discount_factor = discount_factor

        # New options for integrating the trained model and synthetic data file
        self.use_trained_model = use_trained_model
        self.data_file = data_file

        # If using a trained model, load it now (make sure architecture matches training)
        if self.use_trained_model:
            if model_path is None:
                raise ValueError("Model path must be provided if use_trained_model is True.")
            self.trained_model = TransitionNN(input_dim=4, hidden_dim=hidden_dim, output_dim=4)
            self.trained_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.trained_model.eval()

        # Spaces: Each arm has a binary state (0 or 1) and each action is binary per arm.
        self.observation_space = spaces.MultiDiscrete([2] * num_arms)
        self.action_space = spaces.MultiBinary(num_arms)
        
        # Internal structures for state, features, and transitions
        self.state = None       # Current state vector of all arms
        self.features = None    # Per-arm features (shape: [num_arms, 4])
        self.transitions = None # Transition matrices (shape: [num_arms, 2, 2, 2])
        
        self.reset()

    def reset(self):
        """
        Reset the environment to an initial state.
        
        Two modes:
          1) If a data_file is provided, load features and transitions from that file.
          2) Otherwise, generate random features and compute transitions.
             - If use_trained_model is True, use the NN to compute transition probabilities.
             - Otherwise, use the original adjust_probabilities method.
        """
        # Option 1: Load synthetic data from file
        if self.data_file is not None:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            # If the file has a nested "data_obj" (from our earlier generator), extract it.
            if "data_obj" in data:
                data = data["data_obj"]
            self.features = np.array(data["features"], dtype=np.float32)
            self.transitions = np.array(data["transitions"], dtype=np.float32)
        else:
            # Option 2: Generate features for each arm randomly
            self.features = np.zeros((self.num_arms, 4))
            for i in range(self.num_arms):
                # Age: Skewed distribution (beta distribution)
                age = np.random.beta(2, 5)
                # Sex: 50% male/female
                sex = np.random.choice([0, 1])
                # Race: Weighted categories
                race = np.random.choice([0, 1, 2, 3, 4],
                                        p=[0.6, 0.13, 0.06, 0.18, 0.03])
                # Pre-existing condition: Increases with age
                pre_existing_prob = 0.1 + 0.8 * age
                pre_existing_cond = np.random.binomial(1, pre_existing_prob)
                self.features[i, :] = [age, sex, race, pre_existing_cond]
            
            # Build transitions for each arm
            self.transitions = np.zeros((self.num_arms, 2, 2, 2))
            if self.use_trained_model:
                # Use the trained neural network to compute transition probabilities.
                # Convert features to tensor.
                features_tensor = torch.tensor(self.features, dtype=torch.float32)
                with torch.no_grad():
                    predictions = self.trained_model(features_tensor).cpu().numpy()
                for i in range(self.num_arms):
                    # Assume NN outputs: [p(s'=1|s=0,a=0), p(s'=1|s=1,a=0),
                    #                         p(s'=1|s=0,a=1), p(s'=1|s=1,a=1)]
                    p00, p01, p10, p11 = predictions[i]
                    mat = np.zeros((2, 2, 2))
                    # For action 0 ("no treat")
                    mat[0, 0, 1] = p00
                    mat[0, 0, 0] = 1 - p00
                    mat[0, 1, 1] = p01
                    mat[0, 1, 0] = 1 - p01
                    # For action 1 ("treat")
                    mat[1, 0, 1] = p10
                    mat[1, 0, 0] = 1 - p10
                    mat[1, 1, 1] = p11
                    mat[1, 1, 0] = 1 - p11
                    self.transitions[i] = mat
            else:
                # Use the original deterministic method to compute transitions.
                for i in range(self.num_arms):
                    recover_prob, deteriorate_prob = self.adjust_probabilities(self.features[i])
                    self.transitions[i] = self._build_transition_matrix(recover_prob, deteriorate_prob)

        # Initialize states: Randomly assign each arm a state of 0 or 1.
        self.state = np.random.choice([0, 1], size=self.num_arms)

        # If grouping is enabled, use KMeans clustering on scaled features.
        if self.grouping:
            feats_scaled = StandardScaler().fit_transform(self.features)
            kmeans = KMeans(n_clusters=5, random_state=0).fit(feats_scaled)
            self.groups = {}
            for i in range(self.num_arms):
                c = kmeans.labels_[i]
                if c not in self.groups:
                    self.groups[c] = []
                self.groups[c].append(i)
        return self.state

    def _build_transition_matrix(self, recover_prob, deteriorate_prob):
        """
        Build a 2x2x2 transition matrix for a single arm using the provided
        recover and deteriorate probabilities.
        
        Definitions:
          - Action 0 ("no treat"):
              * If state=0: next state remains 0 (no recovery).
              * If state=1: next state becomes 0 with probability deteriorate_prob.
          - Action 1 ("treat"):
              * If state=0: next state becomes 1 with probability recover_prob.
              * If state=1: remains healthy with probability 1.
        """
        mat = np.zeros((2, 2, 2))
        # Action 0
        mat[0, 0, 0] = 1.0
        mat[0, 0, 1] = 0.0
        mat[0, 1, 0] = deteriorate_prob
        mat[0, 1, 1] = 1 - deteriorate_prob
        # Action 1
        mat[1, 0, 1] = recover_prob
        mat[1, 0, 0] = 1 - recover_prob
        mat[1, 1, 1] = 1.0
        mat[1, 1, 0] = 0.0
        return mat

    def adjust_probabilities(self, features, base_recover_prob=0.9,
                             base_deteriorate_prob=0.05, noise_level=0.02):
        """
        Compute recover_prob and deteriorate_prob from patient features.
        This method is called once per arm during reset() when not using a trained model.
        """
        age_weight = -0.2 + np.random.uniform(-noise_level, noise_level)
        sex_weight =  0.05 + np.random.uniform(-noise_level, noise_level)
        race_weights = [
            0      + np.random.uniform(-noise_level, noise_level),
            -0.05  + np.random.uniform(-noise_level, noise_level),
            -0.1   + np.random.uniform(-noise_level, noise_level),
            -0.15  + np.random.uniform(-noise_level, noise_level),
            -0.1   + np.random.uniform(-noise_level, noise_level)
        ]
        pre_existing_weight = -0.3 + np.random.uniform(-noise_level, noise_level)
        age, sex, race, pre_existing_condition = features

        # For treatment (action 1, state 0) use recover_prob
        recover_prob = base_recover_prob + age_weight * age + sex_weight * sex + \
                       race_weights[int(race)] + pre_existing_weight * pre_existing_condition

        # For no treatment (action 0, state 1) use deteriorate_prob
        deteriorate_prob = base_deteriorate_prob - age_weight * age - sex_weight * sex - \
                           race_weights[int(race)] - pre_existing_weight * pre_existing_condition

        recover_prob = np.clip(recover_prob, 0, 1)
        deteriorate_prob = np.clip(deteriorate_prob, 0, 1)

        return recover_prob, deteriorate_prob

    def step(self, action):
        """
        Executes one step in the simulation:
          1) Uses the current state's transition matrices and the chosen action
             to sample the next state for each arm.
          2) Computes the reward (sum of healthy arms in the previous state).
          3) Returns next_state, reward, healthy_percentage, done flag, and info.
        """
        # Validate the action input.
        assert self.action_space.contains(action), "Invalid action!"
        assert np.sum(action) <= self.budget, f"Exceeded budget {self.budget}"

        # Record the current state to compute the reward.
        old_state = self.state.copy()
        reward = np.sum(old_state)

        # Compute the next state by sampling for each arm.
        next_state = np.zeros_like(self.state)
        for i in range(self.num_arms):
            s = self.state[i]   # current state
            a = action[i]       # action for this arm (0 or 1)
            p = self.transitions[i, a, s, :]  # probability vector for next state
            s_next = np.random.choice([0, 1], p=p)
            next_state[i] = s_next

        # Update the environment's state.
        self.state = next_state

        # Calculate the healthy percentage in the new state.
        healthy_percentage = np.mean(self.state) * 100
        done = False
        info = {}
        return next_state, reward, healthy_percentage, done, info

    def compute_whittle_actions(self):
        """
        Computes actions based on the Whittle index.
        If grouping is enabled, a two-level Whittle approach is used.
        Otherwise, a standard per-arm approach is applied.
        """
        if not self.grouping:
            # Standard per-arm Whittle computation.
            whittle_indices = []
            for i in range(self.num_arms):
                s = self.state[i]
                w_index = compute_whittle(self.transitions[i], s, self.discount_factor)
                whittle_indices.append((w_index, i))
            whittle_indices.sort(reverse=True, key=lambda x: x[0])
            action = np.zeros(self.num_arms, dtype=int)
            for _, arm_idx in whittle_indices[:self.budget]:
                action[arm_idx] = 1
            return action
        else:
            # Two-level Whittle computation for grouped arms.
            group_indices = []
            for group_id in self.groups:
                rep_arm = self.groups[group_id][0]
                s_rep = self.state[rep_arm]
                cluster_trans = self.build_cluster_transition(group_id)
                w_group = compute_whittle(cluster_trans, s_rep, self.discount_factor)
                group_indices.append((w_group, group_id))
            group_indices.sort(reverse=True, key=lambda x: x[0])
            action = np.zeros(self.num_arms, dtype=int)
            remaining_budget = self.budget
            for _, grp_id in group_indices:
                if remaining_budget <= 0:
                    break
                arms_in_group = self.groups[grp_id]
                arm_whittle_list = []
                for arm_idx in arms_in_group:
                    s_arm = self.state[arm_idx]
                    w_arm = compute_whittle(self.transitions[arm_idx],
                                            s_arm, self.discount_factor)
                    arm_whittle_list.append((w_arm, arm_idx))
                arm_whittle_list.sort(reverse=True, key=lambda x: x[0])
                for _, chosen_arm in arm_whittle_list:
                    if remaining_budget > 0:
                        action[chosen_arm] = 1
                        remaining_budget -= 1
                    else:
                        break
            return action

    def build_cluster_transition(self, cluster_id):
        """
        For a given group (cluster), builds an "average" 2x2x2 transition matrix
        by summing and then averaging the transitions of all arms in the group.
        """
        members = self.groups[cluster_id]
        cluster_trans = np.zeros((2, 2, 2))
        for idx in members:
            cluster_trans += self.transitions[idx]
        cluster_trans /= len(members)
        return cluster_trans

    def render(self, mode='human'):
        """ Render the current state (for debugging/visualization). """
        print("Current State:", self.state)

    def close(self):
        pass

if __name__ == "__main__":
    # Quick test of the environment.
    # For example, to use the trained model and synthetic data (if available),
    # you could set: use_trained_model=True, model_path="trained_model_no_val.pth", and data_file="synthetic_2x2x2.json"
    env = RMABSimulator(num_arms=5, budget=2, grouping=False,
                        use_trained_model=True, model_path="trained_model_no_val.pth",
                        hidden_dim=32, data_file=None)
    state = env.reset()
    print("Initial State:", state)
    action = np.array([1, 0, 1, 0, 0])
    next_state, reward, hp, done, info = env.step(action)
    print("Next State:", next_state, "Reward:", reward, "Healthy %:", hp)
