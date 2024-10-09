import argparse
import numpy as np
import matplotlib.pyplot as plt
from simulator import RMABSimulator

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def run_experiment(num_arms=10, budget=3, episodes=1000):
    env = RMABSimulator(num_arms=num_arms, budget=budget)

    total_rewards = []
    healthy_percentages = []
    
    for episode in range(episodes):
        state = env.reset()

        # Run the optimal policy (compute Whittle actions)
        action = env.compute_whittle_actions()
        
        next_state, reward, healthy_percentage, done, _ = env.step(action)
        total_rewards.append(reward)
        healthy_percentages.append(healthy_percentage)

    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {episodes} episodes: {avg_reward}")
    
    return total_rewards, healthy_percentages

def main():
    parser = argparse.ArgumentParser(description="Run RMAB Simulation")
    parser.add_argument("--num_arms", type=int, default=10, help="Number of arms (patients)")
    parser.add_argument("--budget", type=int, default=3, help="Budget (number of treatments available)")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to run")

    args = parser.parse_args()

    # Run the experiment
    total_rewards, healthy_percentages = run_experiment(args.num_arms, args.budget, args.episodes)

    # Smooth the data
    smoothed_rewards = moving_average(total_rewards, window_size=10)
    smoothed_healthy_percentages = moving_average(healthy_percentages, window_size=10)

    # Plot the rewards
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(smoothed_rewards, linewidth=2)
    plt.title("Smoothed Rewards over time")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(smoothed_healthy_percentages, linewidth=2)
    plt.title("Smoothed Percentage of Healthy Individuals")
    plt.xlabel("Episode")
    plt.ylabel("Percentage of Healthy Individuals")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
