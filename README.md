# Indivudual-vs-Group-Health-Care-Restless-Bandits

inspired by: https://github.com/lily-x/online-rmab/tree/main/src


## Files
- `main.py` - runs simulator with visulization
- `algorithms.py` - in progress
- `compute_whittle.py` - computes whittle index
- `simulator.py` - simulator using OpenAI's gym

## Running
To run in CLI use:  

```python
python main.py --num_arms [arms] --budget [budget] --episodes [episodes] --policy [whittle/random] --subsidy [subsidy] --discount_factor [discount_factor] --grouping --num_runs [runs]

N - number of arms
(--num_arms) Specify the number of arms (patients).
Example: --num_arms 10

T - number of episodes
(--episodes) Total number of episodes to run.
Example: --episodes 1000

B - budget
(--budget) Total number of treatments/resources to spend.
Example: --budget 3

P - policy to use
(--policy) Choose the policy: whittle or random.
Example: --policy whittle

S - subsidy for Whittle index computation
(--subsidy) Specify the subsidy for Whittle index.
Example: --subsidy 0.5

D - discount factor
(--discount_factor) Set the discount factor for future rewards.
Example: --discount_factor 0.9

G - grouping option
(--grouping) Toggle grouping of patients by characteristics. Default is False.
Example: --grouping

R - number of runs
(--num_runs) Set the number of times to repeat the experiment.
Example: --num_runs 5
