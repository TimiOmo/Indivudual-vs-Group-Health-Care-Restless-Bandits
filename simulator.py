import gym
import numpy as np
from gym import spaces
from compute_whittle import compute_whittle

class RMABSimulator(gym.Env):
    """
    Restless Multi-Armed Bandit Simulator for Healthcare Resource Allocation.

    This environment models a healthcare scenario where you have limited resources
    to allocate among a group of patients. Patients can either be treated as individuals
    or grouped by characteristics. Now uses a 2x2x2 transition approach.
    """

    def __init__(self, num_arms=10, budget=3, grouping=False,
                 subsidy=0.4, discount_factor=0.75):
        super(RMABSimulator, self).__init__()
        
        # Core parameters
        self.num_arms = num_arms        # Number of patients (or groups)
        self.budget = budget            # Number of treatments available
        self.grouping = grouping        # Whether arms are grouped
        self.subsidy = subsidy          # For use by external Whittle computations
        self.discount_factor = discount_factor
        
        # Spaces
        # Each arm is in state {0,1}, so MultiDiscrete(2) for each arm
        self.observation_space = spaces.MultiDiscrete([2]*num_arms)
        # Each arm can be {0,1} for "no treat" vs. "treat"
        self.action_space = spaces.MultiBinary(num_arms)
        
        # Internal structures
        self.state = None               # Will hold current states of all arms
        self.features = None            # Will hold per-arm features
        self.transitions = None         # Will hold the 2x2x2 transitions for each arm
        
        # Initialize at creation
        self.reset()

    def reset(self):
        """
        Reset the environment to an initial state with features that are
        distributed to reflect real-world healthcare scenarios.

        After generating per-arm features, we build a 2x2x2 transition matrix
        for each arm. Then we randomize each arm's initial state.
        """
        # 1) Randomize features
        self.features = np.zeros((self.num_arms, 4))
        for i in range(self.num_arms):
            # Age: Skewed distribution, more older individuals
            age = np.random.beta(2, 5)  
            # Sex: 50% male/female
            sex = np.random.choice([0, 1])
            # Race: Weighted categories
            race = np.random.choice([0, 1, 2, 3, 4],
                                    p=[0.6, 0.13, 0.06, 0.18, 0.03])
            # Pre-existing conditions: Probability increases with age
            pre_existing_prob = 0.1 + 0.8 * age
            pre_existing_cond = np.random.binomial(1, pre_existing_prob)

            self.features[i, :] = [age, sex, race, pre_existing_cond]

        # 2) Build transitions for each arm (shape: [num_arms, 2 (action), 2 (cur_state), 2 (next_state)])
        self.transitions = np.zeros((self.num_arms, 2, 2, 2))
        for i in range(self.num_arms):
            # Compute recover_prob / deteriorate_prob from features
            recover_prob, deteriorate_prob = self.adjust_probabilities(self.features[i])
            # Store the resulting 2x2x2 for that arm
            self.transitions[i] = self._build_transition_matrix(recover_prob, deteriorate_prob)

        # 3) Random initial states for all arms (0 or 1)
        self.state = np.random.choice([0, 1], size=self.num_arms)

        # If grouping is enabled, group the patients
        if self.grouping:
            self.group_patients()

        return self.state

    def _build_transition_matrix(self, recover_prob, deteriorate_prob):
        """
        Build a 2x2x2 matrix for a single arm given its (recover_prob, deteriorate_prob).

        We'll define:
          - action = 0 => "no treat"
             * If state=0 (unhealthy), next state=0 with prob=1 (no spontaneous recovery).
             * If state=1 (healthy), next state=0 with prob=deteriorate_prob, else remain 1.
          - action = 1 => "treat"
             * If state=0 (unhealthy), next state=1 with prob=recover_prob, else remain 0.
             * If state=1 (healthy), remain healthy with prob=1.
        """

        # transitions[action, current_state, next_state]
        # Initialize a blank 2x2 matrix for each action
        mat = np.zeros((2,2,2))

        # Action 0 (no treat)
        # state=0 => next_state=0 with prob=1
        mat[0, 0, 0] = 1.0
        mat[0, 0, 1] = 0.0
        # state=1 => next_state=0 with prob=deteriorate_prob
        mat[0, 1, 0] = deteriorate_prob
        mat[0, 1, 1] = 1 - deteriorate_prob

        # Action 1 (treat)
        # state=0 => next_state=1 with prob=recover_prob
        mat[1, 0, 1] = recover_prob
        mat[1, 0, 0] = 1 - recover_prob
        # state=1 => remain 1 with prob=1
        mat[1, 1, 1] = 1.0
        mat[1, 1, 0] = 0.0

        return mat

    def adjust_probabilities(self, features, base_recover_prob=0.9,
                             base_deteriorate_prob=0.05, noise_level=0.02):
        """
        Compute recover_prob and deteriorate_prob from the patient's features.

        This gets called once per arm in reset() to build the transitions array.
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

        # Extract features
        age, sex, race, pre_existing_condition = features

        # Adjust recovery probability (used when action=1 & state=0 => might become state=1)
        recover_prob = base_recover_prob
        recover_prob += age_weight * age
        recover_prob += sex_weight * sex
        recover_prob += race_weights[int(race)]
        recover_prob += pre_existing_weight * pre_existing_condition

        # Adjust deterioration probability (used when action=0 & state=1 => might become state=0)
        deteriorate_prob = base_deteriorate_prob
        deteriorate_prob -= age_weight * age
        deteriorate_prob -= sex_weight * sex
        deteriorate_prob -= race_weights[int(race)]
        deteriorate_prob -= pre_existing_weight * pre_existing_condition

        # Clip to [0,1]
        recover_prob = np.clip(recover_prob, 0, 1)
        deteriorate_prob = np.clip(deteriorate_prob, 0, 1)

        return recover_prob, deteriorate_prob

    def step(self, action):
        """
        Apply an action to the environment and transition to the next state.
        Action is a binary vector indicating which patients are being treated.

        Returns:
            next_state: updated state after this step
            reward: total reward obtained
            healthy_percentage: % of arms in state=1
            done: always False in this example
            info: additional debug info
        """
        # Validate the action
        assert self.action_space.contains(action), "Invalid action!"
        assert np.sum(action) <= self.budget, f"Exceeded budget {self.budget}"

        next_state = np.zeros_like(self.state)
        reward = 0

        for i in range(self.num_arms):
            s = self.state[i]          # current state (0 or 1)
            a = action[i]              # 0 or 1
            # Probability distribution for next state
            p = self.transitions[i, a, s, :]  # shape = (2,)
            # Sample next state
            s_next = np.random.choice([0,1], p=p)

            # Compute reward logic (similar to old code):
            # If we're treating (a=1):
            #   - If s=1 (patient healthy), we give +1
            #   - If s=0 and s_next=1, we give +1 for recovery
            if a == 1:
                if s == 1:
                    reward += 1
                elif s == 0 and s_next == 1:
                    reward += 1

            next_state[i] = s_next

        self.state = next_state
        healthy_percentage = np.mean(self.state) * 100
        done = False
        info = {}

        return next_state, reward, healthy_percentage, done, info

    def compute_whittle_actions(self):
        """
        Compute the Whittle index for each patient or group and decide which patients to treat.
        Currently still uses a simpler approach in `compute_whittle()`,
        which expects [recover_prob, deteriorate_prob].
        """
        whittle_indices = []

        # For grouping or single patients, we still rely on "adjust_probabilities" each time.
        if self.grouping:
            for group_key in self.groups:
                avg_recover_prob, avg_deteriorate_prob = self.adjust_group_probabilities(group_key)
                # Use the state of the first patient as representative
                rep_state = self.state[self.groups[group_key][0]]
                index = compute_whittle([avg_recover_prob, avg_deteriorate_prob],
                                        rep_state, self.discount_factor, self.subsidy)
                whittle_indices.append((index, group_key))
        else:
            # For each individual
            for i in range(self.num_arms):
                # Re-derive the probabilities (though we already have them in self.transitions[i],
                # we keep this for compatibility with your existing compute_whittle usage)
                features = self.features[i]
                r_prob, d_prob = self.adjust_probabilities(features)
                s = self.state[i]
                index = compute_whittle([r_prob, d_prob], s, self.discount_factor, self.subsidy)
                whittle_indices.append((index, i))

        # Sort by descending Whittle index
        whittle_indices.sort(reverse=True, key=lambda x: x[0])

        # Choose top arms (or groups)
        action = np.zeros(self.num_arms, dtype=int)
        if self.grouping:
            for _, group_key in whittle_indices[:self.budget]:
                for i in self.groups[group_key]:
                    action[i] = 1
        else:
            for _, arm_idx in whittle_indices[:self.budget]:
                action[arm_idx] = 1

        return action

    def render(self, mode='human'):
        """ Print the current state for debugging or visualization. """
        print("Current State:", self.state)

    def close(self):
        pass

    # Under Construction
    # def group_patients(self):
    #     """
    #     Group patients based on features (e.g., age & pre-existing conditions).
    #     Left as-is for now, since grouping logic is not the main focus here.
    #     """
    #     self.groups = {}
    #     for i in range(self.num_arms):
    #         age_group = ("young" if self.features[i, 0] < 0.33
    #                      else "middle-aged" if self.features[i, 0] < 0.66
    #                      else "old")
    #         pre_condition_group = ("low" if self.features[i, 3] < 0.33
    #                                else "moderate" if self.features[i, 3] < 0.66
    #                                else "high")
    #         group_key = (age_group, pre_condition_group)
    #         if group_key not in self.groups:
    #             self.groups[group_key] = []
    #         self.groups[group_key].append(i)

    # def adjust_group_probabilities(self, group_key):
    #     """
    #     Compute average (recover_prob, deteriorate_prob) for a group of arms.
    #     """
    #     group = self.groups[group_key]
    #     total_recover = 0
    #     total_deter = 0
    #     for i in group:
    #         r, d = self.adjust_probabilities(self.features[i])
    #         total_recover += r
    #         total_deter += d
    #     avg_r = total_recover / len(group)
    #     avg_d = total_deter / len(group)
    #     return avg_r, avg_d

if __name__ == "__main__":
    # Quick test
    env = RMABSimulator(num_arms=5, budget=2, grouping=False)
    state = env.reset()
    print("Initial State:", state)

    action = np.array([1,0,1,0,0])
    next_state, reward, hp, done, info = env.step(action)
    print("Next State:", next_state, "Reward:", reward, "Healthy %:", hp)
