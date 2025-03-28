#!/usr/bin/env python
"""
test_transition_model.py

Loads a trained transition model and runs inference on a test JSON file.
Computes error metrics (MSE) if ground truth labels are provided.

Usage:
  python test_transition_model.py --test_file test_data_2x2x2.json --model_file trained_model.pth
"""

import json
import argparse
import numpy as np
import torch
import torch.nn as nn

# Define the same TransitionNN architecture as used during training.
class TransitionNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=4):
        super(TransitionNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Ensures outputs in [0, 1]
        )

    def forward(self, x):
        return self.net(x)

def load_test_data(file_path):
    """
    Loads test data from JSON.
    Expects a top-level object with either:
       { "features": [...], "transitions": [...] }
    or
       { "data_obj": { "features": [...], "transitions": [...] }, ... }
       
    Returns:
       test_features: Tensor of shape [num_arms, 4]
       test_labels:   Tensor of shape [num_arms, 4] corresponding to:
                     [ p(0,0,1), p(0,1,1), p(1,0,1), p(1,1,1) ]
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    # Check if the data is nested under "data_obj"
    if "data_obj" in data:
        data = data["data_obj"]

    features = np.array(data["features"], dtype=np.float32)
    transitions = np.array(data["transitions"], dtype=np.float32)

    labels = []
    for i in range(len(features)):
        p_00 = transitions[i, 0, 0, 1]
        p_01 = transitions[i, 0, 1, 1]
        p_10 = transitions[i, 1, 0, 1]
        p_11 = transitions[i, 1, 1, 1]
        labels.append([p_00, p_01, p_10, p_11])

    test_features = torch.tensor(features, dtype=torch.float32)
    test_labels = torch.tensor(labels, dtype=torch.float32)
    return test_features, test_labels

def main():
    parser = argparse.ArgumentParser(description="Test a trained transition model on a test JSON file.")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to test JSON file with {features, transitions, ground_truth} structure.")
    parser.add_argument("--model_file", type=str, required=True,
                        help="Path to the trained model weights file (e.g., trained_model.pth)")
    parser.add_argument("--hidden_dim", type=int, default=16,
                        help="Hidden dimension of the model (should match training configuration)")
    args = parser.parse_args()

    # Load test data
    test_features, test_labels = load_test_data(args.test_file)
    input_dim = test_features.shape[1]  # typically 4
    output_dim = 4  # four probabilities corresponding to each (action, old_state) combo

    # Instantiate the model and load weights
    model = TransitionNN(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(args.model_file, map_location=torch.device('cpu')))
    model.eval()

    # Run inference on test data
    with torch.no_grad():
        predictions = model(test_features)

    # Compute error metrics (Mean Squared Error)
    mse_loss = nn.MSELoss()
    mse_value = mse_loss(predictions, test_labels)
    print(f"Test MSE Loss: {mse_value.item():.4f}")

    # Print a few sample predictions alongside the ground truth
    print("\nSample predictions vs ground truth:")
    for i in range(min(5, test_features.shape[0])):
        pred = predictions[i].cpu().numpy().tolist()
        true = test_labels[i].cpu().numpy().tolist()
        print(f"Sample {i}: Prediction: {pred}, Ground Truth: {true}")

if __name__ == "__main__":
    main()
