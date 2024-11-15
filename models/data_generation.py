import numpy as np
import json

def generate_synthetic_data(batch_size, output_file="synthetic_data.json"):
    """
    Generate synthetic data with transition probabilities based on feature weights and save to a JSON file.

    Parameters:
    - batch_size: Number of data samples to generate.
    - output_file: Name of the JSON file to save the synthetic data.
    """
    def adjust_probabilities(features, base_recover_prob=0.8, base_deteriorate_prob=0.2, noise_level=0.05):
        """
        Adjust transition probabilities based on feature weights.
        """
        age_weight = -0.2 + np.random.uniform(-noise_level, noise_level)
        sex_weight = 0.05 + np.random.uniform(-noise_level, noise_level)
        race_weights = [
            0.0 + np.random.uniform(-noise_level, noise_level), 
            -0.05 + np.random.uniform(-noise_level, noise_level), 
            -0.1 + np.random.uniform(-noise_level, noise_level), 
            -0.15 + np.random.uniform(-noise_level, noise_level), 
            -0.2 + np.random.uniform(-noise_level, noise_level)
        ]
        pre_existing_weight = -0.3 + np.random.uniform(-noise_level, noise_level)

        age, sex, race, pre_existing_condition = features
        recover_prob = base_recover_prob + age_weight * age + sex_weight * sex + race_weights[int(race)] + pre_existing_weight * pre_existing_condition
        deteriorate_prob = base_deteriorate_prob + age_weight * age + sex_weight * sex + race_weights[int(race)] + pre_existing_weight * pre_existing_condition
        
        # Normalize probabilities
        total_prob = recover_prob + deteriorate_prob
        recover_prob = np.clip(recover_prob / total_prob, 0, 1)
        deteriorate_prob = np.clip(deteriorate_prob / total_prob, 0, 1)

        return recover_prob, deteriorate_prob

    # Generate features
    age = np.random.beta(2, 5, size=(batch_size, 1))
    sex = np.random.choice([0, 1], size=(batch_size, 1))
    race = np.random.choice([0, 1, 2, 3, 4], p=[0.6, 0.13, 0.06, 0.18, 0.03], size=(batch_size, 1))
    pre_existing = (0.1 + 0.8 * age).round()

    features = np.hstack([age, sex, race, pre_existing])

    target_probs = np.zeros((batch_size, 2, 2))
    for i in range(batch_size):
        recover_prob, deteriorate_prob = adjust_probabilities(features[i])
        target_probs[i, 0, :] = [1 - recover_prob, recover_prob]
        target_probs[i, 1, :] = [deteriorate_prob, 1 - deteriorate_prob]

    # Save to JSON
    data = {
        "features": features.tolist(),
        "target_probs": target_probs.tolist()
    }
    with open(output_file, "w") as f:
        json.dump(data, f)

    print(f"Synthetic data saved to {output_file}")

# Example usage
if __name__ == "__main__":
    generate_synthetic_data(batch_size=1000)
