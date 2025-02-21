import gym
import numpy as np
from gym import spaces
from compute_whittle import compute_whittle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


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
             # 1) Possibly scale the features
            feats_scaled = StandardScaler().fit_transform(self.features)
            # 2) K-means
            kmeans = KMeans(n_clusters=5, random_state=0).fit(feats_scaled)
            # 3) Build self.groups
            self.groups = {}
            for i in range(self.num_arms):
                c = kmeans.labels_[i]
                if c not in self.groups:
                    self.groups[c] = []
                self.groups[c].append(i)
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
        A step that counts each healthy arm based on the *current* (old) state,
        then transitions to the next state afterwards.

        Returns:
            next_state,
            reward (sum of old state's healthy arms),
            healthy_percentage (of the new state),
            done,
            info
        """
        # Validate the action
        assert self.action_space.contains(action), "Invalid action!"
        assert np.sum(action) <= self.budget, f"Exceeded budget {self.budget}"

        # 1) Record old state for reward
        old_state = self.state.copy()
        # Reward based on how many arms were healthy in old_state
        reward = np.sum(old_state)

        # 2) Compute next_state
        next_state = np.zeros_like(self.state)
        for i in range(self.num_arms):
            s = self.state[i]           # current state
            a = action[i]               # 0 or 1
            p = self.transitions[i, a, s, :]
            s_next = np.random.choice([0, 1], p=p)
            next_state[i] = s_next

        # 3) Update self.state to the newly sampled next_state
        self.state = next_state

        # 4) Healthy percentage is based on the *new* state
        healthy_percentage = np.mean(self.state) * 100
        done = False
        info = {}

        return next_state, reward, healthy_percentage, done, info



    def compute_whittle_actions(self):
        """
        Compute the formal Whittle index for each arm using the 2x2x2 transition matrix.
        Select the top arms within the budget.
        """
        # If you have grouping, you can handle it similarly by computing
        # a group-level transition matrix or merging transitions, but here's
        # the per-arm logic:

        whittle_indices = []

        # Group Logic - Under Construction
        if self.grouping:
            whittle_indices = []
            for group_id in self.groups:
                # pick a 'representative' state
                # e.g. we just choose the first arm's state in this group
                rep_arm = self.groups[group_id][0]
                s = self.state[rep_arm]
                
                # or average the states if you want, but typically
                # you'd just pick 1 for whittle calculation

                # build average transitions
                cluster_trans = self.build_cluster_transition(group_id)
                
                # compute whittle index, ignoring the fact that multiple states exist
                w_index = compute_whittle(cluster_trans, s, self.discount_factor)
                whittle_indices.append((w_index, group_id))

            # sort by descending whittle
            whittle_indices.sort(reverse=True, key=lambda x: x[0])

            # pick top budget groups
            chosen = [group for _, group in whittle_indices[:self.budget]]

            # final action array
            action = np.zeros(self.num_arms, dtype=int)
            for group_id in chosen:
                for arm_idx in self.groups[group_id]:
                    action[arm_idx] = 1

            return action
        else:
            # No grouping => compute an index for each arm individually
            for i in range(self.num_arms):
                s = self.state[i]
                # self.transitions[i] is shape (2,2,2) for arm i
                w_index = compute_whittle(self.transitions[i], s, self.discount_factor)
                whittle_indices.append((w_index, i))

            # Sort by descending Whittle index
            whittle_indices.sort(reverse=True, key=lambda x: x[0])

            # Pick the top arms
            action = np.zeros(self.num_arms, dtype=int)
            for _, arm_idx in whittle_indices[:self.budget]:
                action[arm_idx] = 1

            return action
    
    # Suppose self.groups is {0: [arm_i1, arm_i2, ...], 1: [...], ...}
    # We'll build an "average" 2x2x2 matrix for each group 
    #  by summing up transitions among that group's arms.

    def build_cluster_transition(self, cluster_id):
        members = self.groups[cluster_id]
        # initialize an empty 2x2x2
        cluster_trans = np.zeros((2,2,2))
        for idx in members:
            cluster_trans += self.transitions[idx]
        # then average by dividing
        cluster_trans /= len(members)
        return cluster_trans


    def render(self, mode='human'):
        """ Print the current state for debugging or visualization. """
        print("Current State:", self.state)

    def close(self):
        pass

if __name__ == "__main__":
    # Quick test
    env = RMABSimulator(num_arms=5, budget=2, grouping=False)
    state = env.reset()
    print("Initial State:", state)

    action = np.array([1,0,1,0,0])
    next_state, reward, hp, done, info = env.step(action)
    print("Next State:", next_state, "Reward:", reward, "Healthy %:", hp)
