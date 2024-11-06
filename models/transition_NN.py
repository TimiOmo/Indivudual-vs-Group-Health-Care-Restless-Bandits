import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Neural Network for Transition Probability Prediction
class TransitionProbabilityNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=8):
        super(TransitionProbabilityNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Input layer to hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Hidden layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # Output layer
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Output probabilities between 0 and 1

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # Output layer with sigmoid activation
        
        # Reshape to 2x2x2 matrix to represent transition probabilities
        x = x.view(-1, 2, 2, 2)
        return x

# Synthetic data generation function
def generate_synthetic_data(batch_size):
    # Generate random features: age, sex, race, pre-existing conditions
    age = np.random.beta(2, 5, size=(batch_size, 1))  # Skewed towards older age
    sex = np.random.choice([0, 1], size=(batch_size, 1))  # 0 for female, 1 for male
    race = np.random.choice([0, 1, 2, 3, 4], p=[0.6, 0.13, 0.06, 0.18, 0.03], size=(batch_size, 1))
    pre_existing = (0.1 + 0.8 * age).round()  # Higher probability with age
    
    # Stack features into a (batch_size, 4) array
    features = np.hstack([age, sex, race, pre_existing])
    
    # Generate target transition probabilities as dummy values for training
    target_probs = np.random.rand(batch_size, 2, 2, 2)
    
    return torch.tensor(features, dtype=torch.float32), torch.tensor(target_probs, dtype=torch.float32)

# Training function
def train_transition_probability_nn(model, num_epochs=1000, batch_size=32, learning_rate=0.001):
    criterion = nn.MSELoss()  # Mean Squared Error loss for probability prediction
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        # Generate synthetic batch
        features, target_probs = generate_synthetic_data(batch_size)
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, target_probs)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Main function to initialize and train the model
def main():
    model = TransitionProbabilityNN(input_dim=4, hidden_dim=32, output_dim=8)
    train_transition_probability_nn(model, num_epochs=1000, batch_size=32, learning_rate=0.001)
    
    # Testing the model on a sample feature
    sample_features = torch.rand((1, 4))  # Random sample feature with 4 dimensions
    output = model(sample_features)
    print("Output Transition Probabilities (2x2x2 matrix):\n", output)

if __name__ == "__main__":
    main()
