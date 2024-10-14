import argparse
import numpy as np
import matplotlib.pyplot as plt
from simulator import RMABSimulator
from algorithms import random_policy, whittle_policy

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def run_experiment(policy_name="whittle", num_arms=10, budget=3, episodes=1000):
    env = RMABSimulator(num_arms=num_arms, budget=budget)
    
    total_rewards = []
    healthy_percentages = []

    for episode in range(episodes):
        state = env.reset()

        # Apply the chosen policy
        if policy_name == "whittle":
            action = env.compute_whittle_actions()
        elif policy_name == "random":
            action = random_policy(env.num_arms, env.budget)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        next_state, reward, healthy_percentage, done, _ = env.step(action)
        total_rewards.append(reward)
        healthy_percentages.append(healthy_percentage)

    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {episodes} episodes: {avg_reward}")
    
    return total_rewards, healthy_percentages

def main():
    parser = argparse.ArgumentParser(description="Run RMAB Simulation")
    parser.add_argument("--policy", type=str, default="whittle", help="Policy to use (whittle or random)")
    parser.add_argument("--num_arms", type=int, default=10, help="Number of arms (patients)")
    parser.add_argument("--budget", type=int, default=3, help="Budget (number of treatments available)")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to run")

    args = parser.parse_args()

    # Run the experiment with the selected policy
    total_rewards, healthy_percentages = run_experiment(args.policy, args.num_arms, args.budget, args.episodes)

    # Smooth the data
    smoothed_rewards = moving_average(total_rewards, window_size=500)
    smoothed_healthy_percentages = moving_average(healthy_percentages, window_size=500)

    # Plot the rewards
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(smoothed_rewards, linewidth=2)
    plt.title(f"Smoothed Rewards over time ({args.policy} policy)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(smoothed_healthy_percentages, linewidth=2)
    plt.title(f"Smoothed Percentage of Healthy Individuals ({args.policy} policy)")
    plt.xlabel("Episode")
    plt.ylabel("Percentage of Healthy Individuals")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
