import torch
import torch.nn as nn
import torch.optim as optim
import json
import argparse

# Define the neural network model
class TransitionNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TransitionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.view(-1, 2, 2)  # Reshape output to a 2x2 matrix
        return nn.functional.softmax(x, dim=-1)

# Load synthetic data
def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    # Flatten the features dictionary into a list of numerical values
    features = torch.tensor(
        [[d["features"]["age"], d["features"]["sex"], d["features"]["race"], d["features"]["pre_existing"]] for d in data],
        dtype=torch.float32
    )

    # Flatten the target probabilities
    target_probs = torch.tensor(
        [d["transition_probabilities"]["healthy"] + d["transition_probabilities"]["unhealthy"] for d in data],
        dtype=torch.float32
    ).view(-1, 2, 2)  # Reshape into [batch_size, 2, 2]

    return features, target_probs


# Training the model
def train_model(model, train_features, train_target_probs, val_features=None, val_target_probs=None, num_epochs=1000, learning_rate=0.01):
    criterion = nn.MSELoss()  # Mean Squared Error for probabilities
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        # Forward pass for training data
        model.train()
        train_outputs = model(train_features)
        train_loss = criterion(train_outputs, train_target_probs)

        # Backward pass and optimization
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Validation loss computation
        if val_features is not None and val_target_probs is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(val_features)
                val_loss = criterion(val_outputs, val_target_probs)

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            if val_features is not None:
                print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")
            else:
                print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss.item():.4f}")

    return model

# Test the model with a new sample
def test_model(model, test_sample):
    test_sample = torch.tensor(test_sample, dtype=torch.float32)
    output = model(test_sample)
    return output.detach().numpy()

# Main workflow
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test the Transition Neural Network.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data file (JSON).")
    parser.add_argument("--val_file", type=str, help="Path to the validation data file (JSON).")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for training.")
    parser.add_argument("--hidden_dim", type=int, default=16, help="Number of hidden dimensions in the neural network.")
    parser.add_argument("--test_sample", type=str, help="Comma-separated test sample features (e.g., 0.1,0.3,0.5,0.2).")

    args = parser.parse_args()

    # Load training data
    train_features, train_target_probs = load_data(args.train_file)

    # Load validation data (optional)
    if args.val_file:
        val_features, val_target_probs = load_data(args.val_file)
    else:
        val_features, val_target_probs = None, None

    # Define hyperparameters
    input_dim = train_features.shape[1]
    hidden_dim = args.hidden_dim
    output_dim = 4  # Corresponds to a 2x2 matrix

    # Initialize the model
    model = TransitionNN(input_dim, hidden_dim, output_dim)

    # Train the model
    print("Training the model...")
    model = train_model(
        model,
        train_features,
        train_target_probs,
        val_features=val_features,
        val_target_probs=val_target_probs,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )

    # Test the model
    if args.test_sample:
        test_sample = [[float(x) for x in args.test_sample.split(",")]]
        print("\nTest Sample (Unnormalized):", test_sample)
        output = test_model(model, test_sample)
        print("\nTransition Matrix (Healthy/Unhealthy):")
        print("Healthy:", output[0][0])
        print("Unhealthy:", output[0][1])

    # Save the trained model
    torch.save(model.state_dict(), "trained_model.pth")
    print("Model saved to trained_model.pth")

