import numpy as np
import json

def generate_weights(batch_size, noise_level=0.05):
    """
    Generate consistent weights for features.
    """
    weights = {
        "age_weight": -0.2 + np.random.uniform(-noise_level, noise_level, size=(batch_size,)),
        "sex_weight": 0.05 + np.random.uniform(-noise_level, noise_level, size=(batch_size,)),
        "race_weights": np.array([
            0.0 + np.random.uniform(-noise_level, noise_level, size=(batch_size,)),
            -0.05 + np.random.uniform(-noise_level, noise_level, size=(batch_size,)),
            -0.1 + np.random.uniform(-noise_level, noise_level, size=(batch_size,)),
            -0.15 + np.random.uniform(-noise_level, noise_level, size=(batch_size,)),
            -0.2 + np.random.uniform(-noise_level, noise_level, size=(batch_size,))
        ]),
        "pre_existing_weight": -0.3 + np.random.uniform(-noise_level, noise_level, size=(batch_size,))
    }
    return weights

def adjust_probabilities(features, weights, base_recover_prob=0.8, base_deteriorate_prob=0.2):
    """
    Adjust transition probabilities based on features and weights.
    """
    age, sex, race, pre_existing_condition = features
    index = int(race)

    logits_recover = (
        base_recover_prob +
        weights["age_weight"] * age +
        weights["sex_weight"] * sex +
        weights["race_weights"][index] +
        weights["pre_existing_weight"] * pre_existing_condition
    )
    logits_deteriorate = (
        base_deteriorate_prob +
        weights["age_weight"] * age +
        weights["sex_weight"] * sex +
        weights["race_weights"][index] +
        weights["pre_existing_weight"] * pre_existing_condition
    )

    recover_prob = 1 / (1 + np.exp(-logits_recover))  # Sigmoid
    deteriorate_prob = 1 / (1 + np.exp(-logits_deteriorate))
    total_prob = recover_prob + deteriorate_prob

    return np.clip(recover_prob / total_prob, 0, 1), np.clip(deteriorate_prob / total_prob, 0, 1)

def generate_synthetic_data(batch_size, output_file, seed=None):
    """
    Generate synthetic data with transition probabilities and save as JSON.

    Arguments:
    - batch_size: Number of data points to generate.
    - output_file: File to save the generated data.
    - seed: Random seed for reproducibility (optional).
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate features
    age = np.random.beta(2, 5, size=(batch_size, 1))
    sex = np.random.choice([0, 1], size=(batch_size, 1))
    race = np.random.choice([0, 1, 2, 3, 4], p=[0.6, 0.13, 0.06, 0.18, 0.03], size=(batch_size, 1))
    pre_existing = (0.1 + 0.8 * age).round()
    features = np.hstack([age, sex, race, pre_existing])

    # Generate weights
    weights = generate_weights(batch_size)

    # Initialize arrays for probabilities
    target_probs = np.zeros((batch_size, 2, 2))
    for i in range(batch_size):
        recover_prob, deteriorate_prob = adjust_probabilities(
            features[i],
            {
                key: val[i] if key != "race_weights" else weights["race_weights"][:, i]
                for key, val in weights.items()
            }
        )
        target_probs[i, 0, :] = [1 - recover_prob, recover_prob]
        target_probs[i, 1, :] = [deteriorate_prob, 1 - deteriorate_prob]

    # Save ground truth
    ground_truth = []
    for i in range(batch_size):
        ground_truth.append({
            "features": {
                "age": features[i][0],
                "sex": features[i][1],
                "race": int(features[i][2]),
                "pre_existing": features[i][3]
            },
            "weights": {
                key: val[i].item() if key != "race_weights" else val[:, i].tolist()
                for key, val in weights.items()
            },
            "transition_probabilities": {
                "healthy": target_probs[i, 0].tolist(),
                "unhealthy": target_probs[i, 1].tolist()
            }
        })

    with open(output_file, "w") as f:
        json.dump(ground_truth, f, indent=4)

    print(f"Ground truth data saved to {output_file}")


# Example usage
if __name__ == "__main__":
    # Generate separate datasets with different random seeds
    generate_synthetic_data(batch_size=1000, output_file="training_data.json", seed=42)
    generate_synthetic_data(batch_size=500, output_file="validation_data.json", seed=123)
    generate_synthetic_data(batch_size=1000, output_file="simulation_data.json", seed=456)
