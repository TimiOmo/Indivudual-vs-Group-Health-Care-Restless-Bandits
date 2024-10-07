import torch 
import torch.nn as nn
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
import random

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the environment (simplified)
class HealthcareEnv:
    def __init__(self, num_arms, grouping=False):
        self.num_arms = num_arms
        self.grouping = grouping
        self.state = torch.zeros(num_arms)  # Represents the health state of each arm
        self.reset()
        
    def reset(self):
        self.state = torch.rand(self.num_arms)  # Random initial health for each arm
        logging.info(f"Environment reset with state: {self.state}")
        return self.state
    
    def step(self, action):
        reward = self.allocate_resources(action)
        self.state[action] = self.state[action] + reward  # Avoid in-place modification
        logging.info(f"Action taken: {action}, Reward: {reward}, New state: {self.state}")
        return self.state, reward
    
    def allocate_resources(self, action):
        # Simulated reward for improving health state
        reward = torch.rand(1).item()
        logging.info(f"Resources allocated to arm {action}, Reward generated: {reward}")
        return reward

# Define the agent (simplified)
class Agent(nn.Module):
    def __init__(self, num_arms):
        super(Agent, self).__init__()
        self.fc = nn.Linear(num_arms, num_arms)
        
    def forward(self, state):
        # Clone the state tensor to avoid in-place modifications
        state_cloned = state.clone()  # Clone the state to ensure no in-place ops
        return torch.softmax(self.fc(state_cloned), dim=-1)

# Train function with epsilon-greedy exploration
def train(env, agent, optimizer, episodes=1000, epsilon=0.1):
    losses = []
    rewards = []
    
    for episode in range(episodes):
        state = env.reset()

        # Detach the state from computation graph to avoid accidental in-place operations
        state = state.detach()

        action_probabilities = agent(state)

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            # Explore: take a random action
            action = random.randint(0, env.num_arms - 1)
        else:
            # Exploit: take the action with the highest probability
            action = torch.argmax(action_probabilities).item()

        # Step in the environment
        next_state, reward = env.step(action)

        # Calculate loss (negative log likelihood of the chosen action * reward)
        loss = -torch.log(action_probabilities[action].clone()) * reward

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Append loss and reward for tracking
        losses.append(loss.item())
        rewards.append(reward)
        
    return losses, rewards


# Comparison experiment setup
def run_experiment():
    num_arms = 10  # Can represent individuals or groups
    
    # Experiment with individuals
    env_individuals = HealthcareEnv(num_arms)
    agent_individuals = Agent(num_arms)
    optimizer_individuals = optim.Adam(agent_individuals.parameters(), lr=0.01)
    
    logging.info("Starting training with individuals as arms")
    losses_individuals, rewards_individuals = train(env_individuals, agent_individuals, optimizer_individuals, epsilon=0.1)
    
    # Experiment with groups
    env_groups = HealthcareEnv(num_arms, grouping=True)
    agent_groups = Agent(num_arms)
    optimizer_groups = optim.Adam(agent_groups.parameters(), lr=0.01)
    
    logging.info("Starting training with groups as arms")
    losses_groups, rewards_groups = train(env_groups, agent_groups, optimizer_groups, epsilon=0.1)
    
    # Plotting the results
    fig, axs = plt.subplots(2, 2, figsize=(15, 7))
    
    # Plot losses for individuals
    axs[0, 0].plot(losses_individuals, label="Loss (Individuals)")
    axs[0, 0].set_title("Loss over Episodes (Individuals)")
    axs[0, 0].set_xlabel("Episodes")
    axs[0, 0].set_ylabel("Loss")
    
    # Plot rewards for individuals
    axs[0, 1].plot(rewards_individuals, color="green", label="Reward (Individuals)")
    axs[0, 1].set_title("Reward over Episodes (Individuals)")
    axs[0, 1].set_xlabel("Episodes")
    axs[0, 1].set_ylabel("Reward")
    
    # Plot losses for groups
    axs[1, 0].plot(losses_groups, label="Loss (Groups)")
    axs[1, 0].set_title("Loss over Episodes (Groups)")
    axs[1, 0].set_xlabel("Episodes")
    axs[1, 0].set_ylabel("Loss")
    
    # Plot rewards for groups
    axs[1, 1].plot(rewards_groups, color="green", label="Reward (Groups)")
    axs[1, 1].set_title("Reward over Episodes (Groups)")
    axs[1, 1].set_xlabel("Episodes")
    axs[1, 1].set_ylabel("Reward")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()


# reward set as state
# continuos
# reward function of state
# loss function is how likely to change state