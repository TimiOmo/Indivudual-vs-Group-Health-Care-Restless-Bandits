def compute_whittle(transitions, state, discount_factor, subsidy):
    """
    Compute the Whittle index for a given arm (patient) in a specific state.
    
    arguments:
    - transitions: Transition matrix for the patient (probabilities of moving between states).
    - state: Current state of the patient (0 or 1).
    - discount_factor: The discount factor for future rewards (gamma).
    - subsidy: The threshold or subsidy under which we balance treatment decisions.
    
    Returns:
    - whittle_index: The computed Whittle index for the patient in the given state.
    """
    # Compute the value of treating and not treating the patient
    value_treat = compute_value_treatment(transitions, state, discount_factor)
    value_no_treat = compute_value_no_treatment(transitions, state, discount_factor)
    
    # Calculate opportunity cost
    opportunity_cost = value_treat - value_no_treat
    
    # Compute the Whittle index by solving the subsidy
    whittle_index = subsidy + opportunity_cost
    
    return whittle_index


def compute_value_treatment(transitions, state, discount_factor):
    """
    Compute the expected value of treating a patient in a given state.
    
    arguments:
    - transitions: Transition matrix for the patient (probabilities of moving between states).
    - state: Current state of the patient (0 or 1).
    - discount_factor: The discount factor for future rewards (gamma).
    
    Returns:
    - value_treat: The expected value of providing treatment.
    """
    # Transitions is a matrix like: [[P(stay unhealthy), P(become healthy)], [P(become unhealthy), P(stay healthy)]]
    
    # If the patient is unhealthy (state = 0)
    if state == 0:
        # Probability of staying unhealthy and becoming healthy when treated
        p_stay_unhealthy, p_become_healthy = transitions[state]
        
        # The reward for becoming healthy is 1 (success), and staying unhealthy is 0 (failure)
        value_treat = p_stay_unhealthy * (0) + p_become_healthy * (1)
        
    # If the patient is healthy (state = 1)
    else:
        # Probability of becoming unhealthy and staying healthy when treated
        p_become_unhealthy, p_stay_healthy = transitions[state]
        
        # The reward for staying healthy is 1 (success), and becoming unhealthy is 0 (failure)
        value_treat = p_become_unhealthy * (0) + p_stay_healthy * (1)
    
    # Consider future rewards with the discount factor
    # We assume that after treating this round, future rounds are valued with the discount factor
    discounted_future_value = discount_factor * value_treat
    
    return discounted_future_value



def compute_value_no_treatment(transitions, state, discount_factor):
    """
    Compute the value of not treating a patient based on the current state and transitions.

    Arguments:
    - transitions: Transition matrix for the patient (probabilities of moving between states without treatment).
    - state: Current state of the patient (0 or 1).
    - discount_factor: The discount factor for future rewards.

    Returns:
    - value_no_treat: The expected value of not treating the patient.
    """
    # Extract the transition probabilities for no treatment (assumed from transitions matrix)
    no_treatment_probs = transitions[state]

    # Calculate the immediate reward (you may or may not want to include immediate reward here)
    # If the patient is already healthy (state 1), no immediate reward (or penalty)
    immediate_reward = 0 if state == 1 else -1  # Penalize if the patient is unhealthy and no treatment is given


    # Calculate future expected value using the transition probabilities and discount factor
    # This assumes you value the future state based on the probability of being in state 1 (healthy)
    future_value = (
        no_treatment_probs[1] * 1  # Probability of transitioning to healthy state * reward for being healthy
        + no_treatment_probs[0] * 0  # Probability of staying in unhealthy state * reward for staying unhealthy
    )

    # Apply discount factor
    value_no_treat = immediate_reward + discount_factor * future_value

    return value_no_treat

