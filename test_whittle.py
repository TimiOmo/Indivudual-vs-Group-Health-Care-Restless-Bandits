import unittest
from compute_whittle import compute_whittle

class TestComputeWhittle(unittest.TestCase):
    def test_healthy_patient(self):
        transitions = [0.8, 0.2]  # High recovery prob, low deterioration prob
        state = 1  # Healthy state
        discount_factor = 0.9
        subsidy = 0.1
        
        result = compute_whittle(transitions, state, discount_factor, subsidy)
        self.assertGreater(result, 0, "Whittle index should be positive for a healthy patient")

    def test_unhealthy_patient(self):
        transitions = [0.6, 0.4]  # Medium recovery prob, medium deterioration prob
        state = 0  # Unhealthy state
        discount_factor = 0.9
        subsidy = 0.1
        
        result = compute_whittle(transitions, state, discount_factor, subsidy)
        self.assertGreater(result, 0, "Whittle index should be positive for an unhealthy patient")

    def test_extreme_subsidy(self):
        transitions = [0.7, 0.3]  # Balanced transition probabilities
        state = 0  # Unhealthy state
        discount_factor = 0.9
        subsidy = 10.0  # High subsidy
        
        result = compute_whittle(transitions, state, discount_factor, subsidy)
        self.assertGreater(result, 10, "Whittle index should be large when the subsidy is large")

if __name__ == "__main__":
    unittest.main()
