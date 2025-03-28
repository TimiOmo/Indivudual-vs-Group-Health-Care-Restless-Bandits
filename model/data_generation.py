import numpy as np
import json

def generate_weights(batch_size, noise_level=0.05):
    """
    Generate random feature weights for a batch of arms/patients.
    This is used to create variability in the logistic equations.
    """
    weights = {
        "age_weight_treat": -0.2 + np.random.uniform(-noise_level, noise_level, size=(batch_size,)),
        "sex_weight_treat": 0.05 + np.random.uniform(-noise_level, noise_level, size=(batch_size,)),
        "race_weights_treat": np.array([
            0.0 + np.random.uniform(-noise_level, noise_level, size=(batch_size,)),
            -0.05 + np.random.uniform(-noise_level, noise_level, size=(batch_size,)),
            -0.1 + np.random.uniform(-noise_level, noise_level, size=(batch_size,)),
            -0.15 + np.random.uniform(-noise_level, noise_level, size=(batch_size,)),
            -0.2 + np.random.uniform(-noise_level, noise_level, size=(batch_size,))
        ]),
        "pre_existing_weight_treat": -0.3 + np.random.uniform(-noise_level, noise_level, size=(batch_size,)),

        # For no-treat scenario, we can define separate offsets if we like
        "age_weight_notreat": -0.1 + np.random.uniform(-noise_level, noise_level, size=(batch_size,)),
        "sex_weight_notreat": 0.02 + np.random.uniform(-noise_level, noise_level, size=(batch_size,)),
        "race_weights_notreat": np.array([
            0.0 + np.random.uniform(-noise_level, noise_level, size=(batch_size,)),
            -0.02 + np.random.uniform(-noise_level, noise_level, size=(batch_size,)),
            -0.05 + np.random.uniform(-noise_level, noise_level, size=(batch_size,)),
            -0.07 + np.random.uniform(-noise_level, noise_level, size=(batch_size,)),
            -0.09 + np.random.uniform(-noise_level, noise_level, size=(batch_size,))
        ]),
        "pre_existing_weight_notreat": -0.15 + np.random.uniform(-noise_level, noise_level, size=(batch_size,))
    }
    return weights

def logistic(x):
    """ Sigmoid helper function. """
    return 1.0 / (1.0 + np.exp(-x))

def build_2x2x2_transition(features, w_dict,
                           base_recover_treat=0.8,
                           base_recover_notreat=0.05,
                           base_deteriorate_treat=0.1,
                           base_deteriorate_notreat=0.2):
    """
    For each arm, build a full 2x2x2 transition matrix:
      transitions[a, old_state, new_state]

    We'll define:
    - (s=0, a=1): recover prob ~ logistic( base_recover_treat + [feature weights] )
    - (s=0, a=0): recover prob ~ logistic( base_recover_notreat + [feature weights] )
    - (s=1, a=1): deteriorate prob ~ logistic( base_deteriorate_treat + [feature weights] )
    - (s=1, a=0): deteriorate prob ~ logistic( base_deteriorate_notreat + [feature weights] )
    """
    age, sex, race, precon = features
    # Convert race to int for indexing
    race_idx = int(race)

    # Weighted logits for treat (s=0 => recover)
    logit_recover_treat = (
        base_recover_treat
        + w_dict["age_weight_treat"]    * age
        + w_dict["sex_weight_treat"]    * sex
        + w_dict["race_weights_treat"][race_idx]
        + w_dict["pre_existing_weight_treat"] * precon
    )
    # Weighted logits for not treat (s=0 => spontaneous recovery)
    logit_recover_notreat = (
        base_recover_notreat
        + w_dict["age_weight_notreat"]    * age
        + w_dict["sex_weight_notreat"]    * sex
        + w_dict["race_weights_notreat"][race_idx]
        + w_dict["pre_existing_weight_notreat"] * precon
    )

    # Weighted logits for treat (s=1 => deteriorate)
    logit_deter_treat = (
        base_deteriorate_treat
        + w_dict["age_weight_treat"]    * age
        + w_dict["sex_weight_treat"]    * sex
        + w_dict["race_weights_treat"][race_idx]
        + w_dict["pre_existing_weight_treat"] * precon
    )
    # Weighted logits for not treat (s=1 => deteriorate)
    logit_deter_notreat = (
        base_deteriorate_notreat
        + w_dict["age_weight_notreat"]    * age
        + w_dict["sex_weight_notreat"]    * sex
        + w_dict["race_weights_notreat"][race_idx]
        + w_dict["pre_existing_weight_notreat"] * precon
    )

    # Now convert logits to probabilities
    # s=0, a=1 => recover
    rec_0_1 = logistic(logit_recover_treat)
    # s=0, a=0 => recover
    rec_0_0 = logistic(logit_recover_notreat)
    # s=1, a=1 => deteriorate
    det_1_1 = logistic(logit_deter_treat)
    # s=1, a=0 => deteriorate
    det_1_0 = logistic(logit_deter_notreat)

    # Build the 2x2x2 matrix
    trans = np.zeros((2,2,2))

    # Action=0 => no treat
    # s=0 => next=1 with prob rec_0_0, next=0 with prob 1-rec_0_0
    trans[0, 0, 0] = 1 - rec_0_0
    trans[0, 0, 1] = rec_0_0

    # s=1 => next=0 with prob det_1_0, next=1 with prob 1-det_1_0
    trans[0, 1, 0] = det_1_0
    trans[0, 1, 1] = 1 - det_1_0

    # Action=1 => treat
    # s=0 => next=1 with prob rec_0_1, next=0 with prob 1-rec_0_1
    trans[1, 0, 0] = 1 - rec_0_1
    trans[1, 0, 1] = rec_0_1

    # s=1 => next=0 with prob det_1_1, next=1 with prob 1-det_1_1
    trans[1, 1, 0] = det_1_1
    trans[1, 1, 1] = 1 - det_1_1

    return trans

def generate_synthetic_data(batch_size=10, output_file="synthetic_2x2x2.json", seed=None):
    """
    Generate features for 'batch_size' arms, create 2x2x2 transitions for each,
    and dump the results to a single JSON file. Also store ground-truth weights for reference.

    We store:
      {
          "data_obj": {
              "features": list_of_feature_vectors,
              "transitions": list_of_2x2x2_matrices
          },
          "ground_truth": [
              {
                "features": {...},
                "weights": {...},
                "transition_matrix": [...]
              },
              ...
          ]
      }
    """
    if seed is not None:
        np.random.seed(seed)

    # 1) Generate features
    age = np.random.beta(2, 5, size=(batch_size, 1))
    sex = np.random.choice([0, 1], size=(batch_size, 1))
    race = np.random.choice([0, 1, 2, 3, 4], p=[0.6, 0.13, 0.06, 0.18, 0.03], size=(batch_size, 1))
    pre_existing = (0.1 + 0.8 * age).round()
    features = np.hstack([age, sex, race, pre_existing])

    # 2) Generate random weights
    w = generate_weights(batch_size)

    # 3) Build 2x2x2 transitions for each arm
    transitions_array = np.zeros((batch_size, 2, 2, 2))
    for i in range(batch_size):
        f_i = features[i]  # shape (4,)
        w_i = {
            "age_weight_treat": w["age_weight_treat"][i],
            "sex_weight_treat": w["sex_weight_treat"][i],
            "race_weights_treat": w["race_weights_treat"][:, i],
            "pre_existing_weight_treat": w["pre_existing_weight_treat"][i],
            "age_weight_notreat": w["age_weight_notreat"][i],
            "sex_weight_notreat": w["sex_weight_notreat"][i],
            "race_weights_notreat": w["race_weights_notreat"][:, i],
            "pre_existing_weight_notreat": w["pre_existing_weight_notreat"][i]
        }
        trans_2x2x2 = build_2x2x2_transition(f_i, w_i)
        transitions_array[i] = trans_2x2x2

    data_obj = {
        "features": features.tolist(),
        "transitions": transitions_array.tolist()
    }

    ground_truth = []
    for i in range(batch_size):
        gt_dict = {
            "features": {
                "age": features[i][0],
                "sex": features[i][1],
                "race": int(features[i][2]),
                "pre_existing": features[i][3]
            },
            "weights": {
                "age_weight_treat": w["age_weight_treat"][i].item(),
                "sex_weight_treat": w["sex_weight_treat"][i].item(),
                "race_weights_treat": w["race_weights_treat"][:, i].tolist(),
                "pre_existing_weight_treat": w["pre_existing_weight_treat"][i].item(),
                "age_weight_notreat": w["age_weight_notreat"][i].item(),
                "sex_weight_notreat": w["sex_weight_notreat"][i].item(),
                "race_weights_notreat": w["race_weights_notreat"][:, i].tolist(),
                "pre_existing_weight_notreat": w["pre_existing_weight_notreat"][i].item()
            },
            "transition_matrix": transitions_array[i].tolist()
        }
        ground_truth.append(gt_dict)

    # Combine everything into one JSON structure
    combined_output = {
        "data_obj": data_obj,
        "ground_truth": ground_truth
    }

    # 4) Write the combined dictionary to a single JSON file
    with open(output_file, "w") as f:
        json.dump(combined_output, f, indent=4)

    print(f"[Done] Synthetic 2x2x2 data (batch={batch_size}) saved to {output_file}")

if __name__ == "__main__":
    generate_synthetic_data(batch_size=10, output_file="synthetic_2x2x2.json", seed=123)
    generate_synthetic_data(batch_size=50, output_file="training_data_2x2x2.json", seed=42)
    generate_synthetic_data(batch_size=20, output_file="test_data_2x2x2.json", seed=999)
