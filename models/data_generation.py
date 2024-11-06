import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples=1000):
    """
    Generate synthetic data with demographic and health-related features for training a neural network model.

    Parameters:
    - num_samples: int, number of samples to generate.

    Returns:
    - pandas DataFrame containing the synthetic data.
    """
    data = {
        # Age: Skewed distribution towards older individuals
        "age": np.random.beta(2, 5, num_samples),  # Older individuals more likely

        # Sex: Binary, with equal probability
        "sex": np.random.choice([0, 1], num_samples),  # 0 = female, 1 = male

        # Race: Categorical, with specified distribution for a hypothetical population
        "race": np.random.choice(
            ["White", "Black", "Asian", "Hispanic", "Other/Mixed"], 
            num_samples,
            p=[0.6, 0.13, 0.06, 0.18, 0.03]
        ),

        # Pre-existing Conditions: Higher probability with age
        "pre_existing_condition": np.random.binomial(
            1, 0.1 + 0.8 * np.random.beta(2, 5, num_samples)  # More likely for older
        )
    }

    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    return df

# Example usage:
synthetic_data = generate_synthetic_data(1000)
print(synthetic_data.head())
