# File: compute_whittle.py

import numpy as np

def compute_whittle(transitions,
                    current_state,
                    discount_factor,
                    lambda_low=0.0,
                    lambda_high=2.0,
                    max_iter=50,
                    tol=1e-6):
    """
    Compute the Whittle index for a 2-state arm in 'current_state'
    assuming reward = s (if s=1 => +1, s=0 => +0) minus lambda if action=1.
    The environment's actual reward does *not* subtract cost, but
    lambda appears here as a Lagrange multiplier from the budget constraint.

    transitions: shape (2,2,2): transitions[a,s,s_next].
    current_state: 0 or 1.
    discount_factor: float in (0,1).
    lambda_low, lambda_high: search bounds for lambda.
    max_iter: how many bisection steps and Q-iterations to do.
    tol: tolerance for concluding Q(passive) ~ Q(active).
    """

    def compute_qvalues_for_lambda(lambda_val,
                                   transitions,
                                   discount_factor,
                                   max_iter=50,
                                   tol=1e-6):
        """
        Solve the 2-state Bellman equations for the given lambda:
          r_lambda(s,a) = s - lambda*(a==1).
        """
        Q = np.zeros((2,2))  # Q[s,a] for s=0..1, a=0..1

        for _ in range(max_iter):
            Q_prev = Q.copy()
            for s in (0,1):
                for a in (0,1):
                    # Immediate reward: s minus lambda if action=1
                    r_sa = float(s) - (lambda_val if a == 1 else 0.0)

                    # Value of next states
                    exp_val = 0.0
                    for s_next in (0,1):
                        # next state's value = max of Q[s_next, any a']
                        v_next = max(Q_prev[s_next,0], Q_prev[s_next,1])
                        exp_val += transitions[a, s, s_next] * v_next

                    Q[s,a] = r_sa + discount_factor * exp_val

            # check for convergence
            if np.max(np.abs(Q - Q_prev)) < tol:
                break

        return Q

    def q_diff(lambda_val):
        """
        Difference Q_lambda(current_state, 0) - Q_lambda(current_state, 1).
        If difference=0, you're indifferent => that lambda is the index.
        """
        Q = compute_qvalues_for_lambda(lambda_val,
                                       transitions,
                                       discount_factor,
                                       max_iter=max_iter,
                                       tol=tol)
        return Q[current_state, 0] - Q[current_state, 1]

    # Bisection search to find the lambda where Q(passive)=Q(active).
    left, right = lambda_low, lambda_high
    left_diff  = q_diff(left)
    right_diff = q_diff(right)

    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        mid_diff = q_diff(mid)

        if abs(mid_diff) < tol:
            return mid  # near exact root

        if np.sign(mid_diff) == np.sign(left_diff):
            left = mid
            left_diff = mid_diff
        else:
            right = mid

    return 0.5*(left+right)  # best guess if no exact crossing found


def compute_whittle_for_both_states(transitions, discount_factor):
    """
    Convenience: get the Whittle index for state=0 and state=1
    in a single call. Returns a dict with 'W0' and 'W1'.
    """
    w0 = compute_whittle(transitions, 0, discount_factor)
    w1 = compute_whittle(transitions, 1, discount_factor)
    return {'W0': w0, 'W1': w1}
