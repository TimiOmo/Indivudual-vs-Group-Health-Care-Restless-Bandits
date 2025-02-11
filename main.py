import argparse
import numpy as np
import matplotlib.pyplot as plt
from simulator import RMABSimulator
from algorithms import random_policy, whittle_policy

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def run_experiment(policy_name="whittle", num_arms=10, budget=3, episodes=1000, grouping=False, subsidy=0.4, discount_factor=0.75):
    env = RMABSimulator(num_arms=num_arms, budget=budget, grouping=grouping, subsidy=subsidy, discount_factor=discount_factor)
    
    total_rewards = []
    healthy_percentages = []

    for episode in range(episodes):
        state = env.reset()

        # Apply the chosen policy
        if policy_name == "whittle":
            action = env.compute_whittle_actions()
        elif policy_name == "random":
            action = random_policy(env)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        next_state, reward, healthy_percentage, done, _ = env.step(action)
        total_rewards.append(reward)
        healthy_percentages.append(healthy_percentage)

    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {episodes} episodes: {avg_reward}")

    # Calculate cumulative rewards
    cumulative_rewards = np.cumsum(total_rewards)

    # Compute average probabilities
    avg_recover_prob, avg_deteriorate_prob = env.calculate_average_probabilities()
    print(f"Average Recover Probability: {avg_recover_prob}, Average Deteriorate Probability: {avg_deteriorate_prob}")

    return total_rewards, healthy_percentages, cumulative_rewards

def run_multiple_experiments(num_runs, **kwargs):
    all_rewards = []
    all_healthy_percentages = []

    for run in range(num_runs):
        print(f"\nRunning experiment {run + 1} out of {num_runs}")
        total_rewards, healthy_percentages, _ = run_experiment(**kwargs)
        all_rewards.append(total_rewards)
        all_healthy_percentages.append(healthy_percentages)

    avg_rewards = np.mean(all_rewards, axis=0)
    avg_healthy_percentages = np.mean(all_healthy_percentages, axis=0)

    return avg_rewards, avg_healthy_percentages

def main():
    parser = argparse.ArgumentParser(description="Run RMAB Simulation")
    parser.add_argument("--policy", type=str, default="whittle", help="Policy to use (whittle or random)")
    parser.add_argument("--num_arms", type=int, default=10, help="Number of arms (patients)")
    parser.add_argument("--budget", type=int, default=3, help="Budget (number of treatments available)")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to run")
    parser.add_argument("--grouping", action="store_true", help="Group patients by characteristics")
    parser.add_argument("--subsidy", type=float, default=0.4, help="Subsidy for Whittle index computation")
    parser.add_argument("--discount_factor", type=float, default=0.75, help="Discount factor for future rewards")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs for multiple experiments")

    args = parser.parse_args()

    if args.num_runs > 1:
        avg_rewards, avg_healthy_percentages = run_multiple_experiments(
            num_runs=args.num_runs, policy_name=args.policy, num_arms=args.num_arms, 
            budget=args.budget, episodes=args.episodes, grouping=args.grouping, 
            subsidy=args.subsidy, discount_factor=args.discount_factor
        )
        cumulative_rewards = np.cumsum(avg_rewards)
    else:
        avg_rewards, avg_healthy_percentages, cumulative_rewards = run_experiment(
            policy_name=args.policy, num_arms=args.num_arms, budget=args.budget, 
            episodes=args.episodes, grouping=args.grouping, subsidy=args.subsidy, 
            discount_factor=args.discount_factor
        )

        

    # Smooth data with dynamic window size based on number of episodes
    window_size = min(50, len(avg_rewards) // 10)  # Adjust this to suit your data length
    smoothed_rewards = moving_average(avg_rewards, window_size=window_size)
    smoothed_healthy_percentages = moving_average(avg_healthy_percentages, window_size=window_size)


    # Plot cumulative rewards and smoothed data
    plt.figure(figsize=(18, 6))

    # Cumulative reward plot
    plt.subplot(1, 3, 1)
    plt.plot(cumulative_rewards, linewidth=2, label="Cumulative Reward")
    plt.title(f"Cumulative Reward over Episodes ({args.policy} policy)")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True)

    # Smoothed reward plot
    plt.subplot(1, 3, 2)
    plt.plot(smoothed_rewards, linewidth=2)
    plt.title(f"Smoothed Rewards over time ({args.policy} policy) - Grouping: {args.grouping}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)

    # Smoothed healthy percentage plot
    plt.subplot(1, 3, 3)
    plt.plot(smoothed_healthy_percentages, linewidth=2)
    plt.title(f"Smoothed Percentage of Healthy Individuals ({args.policy} policy) - Grouping: {args.grouping}")
    plt.xlabel("Episode")
    plt.ylabel("Percentage of Healthy Individuals")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
