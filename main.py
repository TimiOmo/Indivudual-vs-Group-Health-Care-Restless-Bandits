# File: main.py

import argparse
import numpy as np
import matplotlib.pyplot as plt
from simulator import RMABSimulator
from algorithms import random_policy, whittle_policy

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def run_experiment(policy_name="whittle",
                   num_arms=10,
                   budget=3,
                   episodes=100,
                   steps_per_episode=10,
                   grouping=False,
                   subsidy=0.4,
                   discount_factor=0.75):
    """
    Run multiple episodes. In each episode, we reset the environment
    and then run 'steps_per_episode' steps of action selection (policy)
    and environment transitions.

    :param policy_name: "whittle" or "random"
    :param num_arms: number of arms (patients)
    :param budget: how many arms can be activated each step
    :param episodes: how many episodes to run
    :param steps_per_episode: how many steps per episode
    :param grouping: whether arms are grouped
    :param subsidy: for whittle calculations (passed to environment)
    :param discount_factor: for whittle calculations
    :return: 
        total_rewards: list of episode rewards (length=episodes)
        healthy_percentages: list of final-step healthy percentages per episode
        cumulative_rewards: cumulative sum of total_rewards
    """

    env = RMABSimulator(num_arms=num_arms,
                        budget=budget,
                        grouping=grouping,
                        subsidy=subsidy,
                        discount_factor=discount_factor)
    
    total_rewards = []
    healthy_percentages = []

    for ep in range(episodes):
        # Reset environment at the start of each episode
        env.reset()
        episode_reward = 0
        final_healthy_pct = 0.0

        for step in range(steps_per_episode):
            # Apply the chosen policy
            if policy_name == "whittle":
                action = env.compute_whittle_actions()
            elif policy_name == "random":
                action = random_policy(env)
            else:
                raise ValueError(f"Unknown policy: {policy_name}")

            # Step the environment
            next_state, reward, healthy_percentage, done, _ = env.step(action)
            episode_reward += reward
            final_healthy_pct = healthy_percentage

            # If we wanted a termination condition, we could break here:
            if done:
                break

        total_rewards.append(episode_reward)
        healthy_percentages.append(final_healthy_pct)

    # Summaries
    avg_reward = np.mean(total_rewards)
    print(f"\nPolicy: {policy_name}, Episodes: {episodes}, Steps/Episode: {steps_per_episode}")
    print(f"Average reward per episode: {avg_reward:.3f}")

    # Calculate cumulative rewards across episodes
    cumulative_rewards = np.cumsum(total_rewards)

    # avg_recover_prob, avg_deteriorate_prob = env.calculate_average_probabilities()
    # print(f"Average Recover Probability: {avg_recover_prob:.3f}, "
    #       f"Average Deteriorate Probability: {avg_deteriorate_prob:.3f}")

    return total_rewards, healthy_percentages, cumulative_rewards

def run_multiple_experiments(num_runs=1, **kwargs):
    """
    Run the experiment multiple times (e.g. for repeated trials)
    and average the results.
    """
    all_rewards = []
    all_healthy_percentages = []

    for run_idx in range(num_runs):
        print(f"\nRunning experiment {run_idx + 1} out of {num_runs}")
        total_rewards, healthy_percentages, _ = run_experiment(**kwargs)
        all_rewards.append(total_rewards)
        all_healthy_percentages.append(healthy_percentages)

    # Average across runs
    avg_rewards = np.mean(all_rewards, axis=0)
    avg_healthy_percentages = np.mean(all_healthy_percentages, axis=0)

    return avg_rewards, avg_healthy_percentages

def main():
    parser = argparse.ArgumentParser(description="Run RMAB Simulation")
    parser.add_argument("--policy", type=str, default="whittle", help="Policy to use (whittle or random)")
    parser.add_argument("--num_arms", type=int, default=10, help="Number of arms (patients)")
    parser.add_argument("--budget", type=int, default=3, help="Budget (number of treatments available)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run")
    parser.add_argument("--steps_per_episode", type=int, default=10, help="Steps (time horizon) within each episode")
    parser.add_argument("--grouping", action="store_true", help="Group patients by characteristics")
    parser.add_argument("--subsidy", type=float, default=0.4, help="Subsidy for Whittle index computation")
    parser.add_argument("--discount_factor", type=float, default=0.75, help="Discount factor for future rewards")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs for multiple experiments")

    args = parser.parse_args()

    if args.num_runs > 1:
        avg_rewards, avg_healthy_percentages = run_multiple_experiments(
            num_runs=args.num_runs,
            policy_name=args.policy,
            num_arms=args.num_arms,
            budget=args.budget,
            episodes=args.episodes,
            steps_per_episode=args.steps_per_episode,
            grouping=args.grouping,
            subsidy=args.subsidy,
            discount_factor=args.discount_factor
        )
        cumulative_rewards = np.cumsum(avg_rewards)
    else:
        avg_rewards, avg_healthy_percentages, cumulative_rewards = run_experiment(
            policy_name=args.policy,
            num_arms=args.num_arms,
            budget=args.budget,
            episodes=args.episodes,
            steps_per_episode=args.steps_per_episode,
            grouping=args.grouping,
            subsidy=args.subsidy,
            discount_factor=args.discount_factor
        )

    # Smooth data
    window_size = min(50, len(avg_rewards) // 10) if len(avg_rewards) > 10 else 1
    smoothed_rewards = moving_average(avg_rewards, window_size=window_size)
    smoothed_healthy = moving_average(avg_healthy_percentages, window_size=window_size)

    # Plotting
    plt.figure(figsize=(18, 6))

    # 1) Cumulative reward plot
    plt.subplot(1, 3, 1)
    plt.plot(cumulative_rewards, linewidth=2, label="Cumulative Reward")
    plt.title(f"Cumulative Reward over Episodes ({args.policy} policy)")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True)

    # 2) Smoothed reward plot
    plt.subplot(1, 3, 2)
    plt.plot(smoothed_rewards, linewidth=2)
    plt.title(f"Smoothed Rewards over time\n(Policy={args.policy}, Grouping={args.grouping})")
    plt.xlabel("Episode")
    plt.ylabel("Reward (smoothed)")
    plt.grid(True)

    # 3) Smoothed healthy plot
    plt.subplot(1, 3, 3)
    plt.plot(smoothed_healthy, linewidth=2)
    plt.title(f"Smoothed % Healthy\n(Policy={args.policy}, Grouping={args.grouping})")
    plt.xlabel("Episode")
    plt.ylabel("Percentage Healthy (smoothed)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
