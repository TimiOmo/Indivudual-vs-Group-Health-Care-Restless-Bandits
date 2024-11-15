import torch
import torch.nn as nn
import torch.optim as optim
import json

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
    features = torch.tensor(data["features"], dtype=torch.float32)
    target_probs = torch.tensor(data["target_probs"], dtype=torch.float32)

    # Reduce target_probs to a [batch_size, 2, 2] shape
    reduced_target_probs = target_probs[:, :, :, 1]  # Extract "probability to improve" for healthy and unhealthy
    return features, reduced_target_probs

# Training the model
def train_model(model, features, target_probs, num_epochs=1000, learning_rate=0.01):
    criterion = nn.MSELoss()  # Mean Squared Error for probabilities
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, target_probs)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

    return model

# Test the model with a new sample
def test_model(model, test_sample):
    test_sample = torch.tensor(test_sample, dtype=torch.float32)
    output = model(test_sample)
    return output.detach().numpy()

# Main workflow
if __name__ == "__main__":
    # Load data
    features, target_probs = load_data("synthetic_data.json")

    # Define hyperparameters
    input_dim = features.shape[1]
    hidden_dim = 16
    output_dim = 4  # Corresponds to a 2x2 matrix

    # Initialize the model
    model = TransitionNN(input_dim, hidden_dim, output_dim)

    # Train the model
    print("Training the model...")
    model = train_model(model, features, target_probs)

    # Test the model
    test_sample = [[0.1, 0.3, 0.5, 0.2]]  # Example test sample
    print("\nTest Sample (Unnormalized):", test_sample)
    output = test_model(model, test_sample)
    print("\nTransition Matrix (Healthy/Unhealthy):")
    print("Healthy:", output[0][0])
    print("Unhealthy:", output[0][1])
