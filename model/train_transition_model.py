#!/usr/bin/env python
"""
train_transition_model.py

Trains a neural network to predict the probabilities of next_state=1
for each combination of (old_state, action) => four outputs total:
  p(s'=1|s=0,a=0), p(s'=1|s=0,a=1),
  p(s'=1|s=1,a=0), p(s'=1|s=1,a=1).

Usage example:
  python train_transition_model.py \
    --train_file training_data_2x2x2.json \
    --val_file validation_data_2x2x2.json \
    --num_epochs 1000 \
    --learning_rate 0.01 \
    --output_model trained_model.pth

Saves the model weights to 'trained_model.pth'.
"""

import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    # If features and transitions are nested under "data_obj", extract them.
    if "data_obj" in data:
        data = data["data_obj"]

    features = np.array(data["features"], dtype=np.float32)
    transitions = np.array(data["transitions"], dtype=np.float32)

    # Build the 4 probabilities for next_state=1:
    labels = []
    for i in range(len(features)):
        p_00 = transitions[i, 0, 0, 1]
        p_01 = transitions[i, 0, 1, 1]
        p_10 = transitions[i, 1, 0, 1]
        p_11 = transitions[i, 1, 1, 1]
        labels.append([p_00, p_01, p_10, p_11])

    features_torch = torch.tensor(features, dtype=torch.float32)
    labels_torch   = torch.tensor(labels,   dtype=torch.float32)

    return features_torch, labels_torch



class TransitionNN(nn.Module):
    """
    Simple feedforward neural network:
      input_dim -> hidden_dim -> hidden_dim -> output_dim=4
    Each output is a probability in [0,1], so we apply a final Sigmoid.
    """
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # outputs in [0,1]
        )

    def forward(self, x):
        return self.net(x)  # shape [batch_size,4]


def train_model(model,
                train_features,
                train_labels,
                val_features=None,
                val_labels=None,
                num_epochs=1000,
                learning_rate=0.01):
    """
    Train the model using MSELoss. Optionally do validation every 100 epochs.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs+1):
        model.train()
        outputs = model(train_features)  # [batch_size,4]
        loss = criterion(outputs, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            msg = f"Epoch {epoch}/{num_epochs}, Train Loss={loss.item():.4f}"
            if val_features is not None and val_labels is not None:
                model.eval()
                with torch.no_grad():
                    val_out = model(val_features)
                    val_loss = criterion(val_out, val_labels)
                msg += f", Val Loss={val_loss.item():.4f}"
            print(msg)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network to predict 2x2x2 transitions.")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training JSON with {features, transitions, ground_truth}.")
    parser.add_argument("--val_file",   type=str, default=None,
                        help="Optional path to validation JSON (same format).")
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--output_model", type=str, default="trained_model.pth",
                        help="File to save model weights.")
    args = parser.parse_args()

    # 1) Load training data
    train_features, train_labels = load_data(args.train_file)

    # 2) Optionally load validation data
    if args.val_file:
        val_features, val_labels = load_data(args.val_file)
    else:
        val_features, val_labels = None, None

    input_dim = train_features.shape[1]  # typically 4
    output_dim = 4                      # p(s'=1) for each combination (a=0|1, s=0|1)

    # 3) Build the neural network
    model = TransitionNN(input_dim=input_dim,
                         hidden_dim=args.hidden_dim,
                         output_dim=output_dim)
    print(model)

    # 4) Train
    model = train_model(model,
                        train_features, train_labels,
                        val_features, val_labels,
                        num_epochs=args.num_epochs,
                        learning_rate=args.learning_rate)

    # 5) Save the model weights
    torch.save(model.state_dict(), args.output_model)
    print(f"Model saved to {args.output_model}")
