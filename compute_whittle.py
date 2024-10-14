def compute_whittle(transitions, state, discount_factor, subsidy):
    """
    Compute the Whittle index for a given arm (patient) in a specific state.
    
    arguments:
    - transitions: A list with two elements [recover_prob, deteriorate_prob] representing 
                   the probabilities of recovering and deteriorating.
    - state: Current state of the patient (0 or 1).
    - discount_factor: The discount factor for future rewards (gamma).
    - subsidy: The threshold or subsidy under which we balance treatment decisions.
    
    Returns:
    - whittle_index: The computed Whittle index for the patient in the given state.
    """
    recover_prob, deteriorate_prob = transitions

    # Compute the value of treating and not treating the patient
    value_treat = compute_value_treatment(recover_prob, state, discount_factor)
    value_no_treat = compute_value_no_treatment(deteriorate_prob, state, discount_factor)
    
    # Calculate opportunity cost
    opportunity_cost = value_treat - value_no_treat
    
    # Compute the Whittle index by solving the subsidy
    whittle_index = subsidy + opportunity_cost
    
    return whittle_index


def compute_value_treatment(recover_prob, state, discount_factor):
    """
    Compute the expected value of treating a patient in a given state.
    Adjusted to use the dynamic recover probability instead of a fixed transition matrix.
    """
    # If the patient is unhealthy (state = 0)
    if state == 0:
        value_treat = recover_prob * (1)  # Reward for becoming healthy
    else:
        value_treat = 1  # Patient is already healthy, reward is 1 (stay healthy)
    
    # Discount future rewards
    discounted_value_treat = discount_factor * value_treat
    return discounted_value_treat


def compute_value_no_treatment(deteriorate_prob, state, discount_factor):
    """
    Compute the value of not treating a patient based on the current state.
    Adjusted to use the dynamic deteriorate probability instead of a fixed transition matrix.
    """
    # Immediate reward
    immediate_reward = 0 if state == 1 else -1  # Penalize if the patient is unhealthy and no treatment is given

    # Calculate future expected value based on deterioration probability
    future_value = deteriorate_prob * 0 + (1 - deteriorate_prob) * 1  # Probability of staying healthy

    # Apply discount factor
    value_no_treat = immediate_reward + discount_factor * future_value

    return value_no_treat
