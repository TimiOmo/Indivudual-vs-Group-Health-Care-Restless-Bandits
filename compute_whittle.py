def compute_whittle(transitions, state, discount_factor, subsidy, reward_healthy=1, reward_unhealthy=-1):
    """
    Compute the Whittle index for a given arm (patient) in a specific state.
    
    Arguments:
    - transitions: [recover_prob, deteriorate_prob], probabilities of recovery and deterioration.
    - state: Current state of the patient (0: unhealthy, 1: healthy).
    - discount_factor: Discount factor (gamma) for future rewards.
    - subsidy: Baseline subsidy to balance treatment decisions.
    - reward_healthy: Reward for a healthy state.
    - reward_unhealthy: Reward (or penalty) for an unhealthy state.
    
    Returns:
    - whittle_index: The computed Whittle index for the given state and probabilities.
    """
    recover_prob, deteriorate_prob = transitions

    # Validate probabilities
    if not (0 <= recover_prob <= 1 and 0 <= deteriorate_prob <= 1):
        raise ValueError("Probabilities must be between 0 and 1.")

    # Calculate values for treating and not treating
    value_treat = compute_value_treatment(
        recover_prob, state, discount_factor, reward_healthy, reward_unhealthy
    )
    value_no_treat = compute_value_no_treatment(
        deteriorate_prob, state, discount_factor, reward_healthy, reward_unhealthy
    )

    # Opportunity cost of treating
    opportunity_cost = value_treat - value_no_treat

    # Compute Whittle index
    whittle_index = subsidy + opportunity_cost
    return whittle_index


def compute_value_treatment(recover_prob, state, discount_factor, reward_healthy, reward_unhealthy):
    """
    Compute the expected value of treating a patient in a given state.
    """
    if state == 0:  # Unhealthy state
        immediate_reward = recover_prob * reward_healthy + (1 - recover_prob) * reward_unhealthy
    else:  # Healthy state
        immediate_reward = reward_healthy

    # Future rewards (infinite horizon assumption)
    future_reward = (
        discount_factor
        * (recover_prob * reward_healthy + (1 - recover_prob) * reward_unhealthy)
        / (1 - discount_factor)
    )

    return immediate_reward + future_reward


def compute_value_no_treatment(deteriorate_prob, state, discount_factor, reward_healthy, reward_unhealthy):
    """
    Compute the expected value of not treating a patient based on the current state.
    """
    if state == 1:  # Healthy state
        immediate_reward = reward_healthy
    else:  # Unhealthy state
        immediate_reward = reward_unhealthy

    # Future rewards (infinite horizon assumption)
    future_reward = (
        discount_factor
        * (deteriorate_prob * reward_unhealthy + (1 - deteriorate_prob) * reward_healthy)
        / (1 - discount_factor)
    )

    return immediate_reward + future_reward
