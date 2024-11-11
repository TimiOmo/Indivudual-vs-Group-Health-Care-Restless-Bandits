# Individual vs Group Health Care Restless Bandits

Inspired by: [online-rmab](https://github.com/lily-x/online-rmab)

## Files

- `main.py` - Runs the simulator with visualization.
- `algorithms.py` - Contains algorithms (in progress).
- `compute_whittle.py` - Computes the Whittle index.
- `simulator.py` - Simulator built using OpenAI's gym.

## Running

To run in the CLI, use:

```bash
python main.py --num_arms [arms] --budget [budget] --episodes [episodes] --policy [whittle/random] --subsidy [subsidy] --discount_factor [discount_factor] --grouping --num_runs [runs]

### Parameters

- **N - number of arms**
  - `--num_arms`: Specify the number of arms (patients).
  - Example: `--num_arms 10`

- **T - number of episodes**
  - `--episodes`: Total number of episodes to run.
  - Example: `--episodes 1000`

- **B - budget**
  - `--budget`: Total number of treatments/resources to spend.
  - Example: `--budget 3`

- **P - policy to use**
  - `--policy`: Choose the policy: whittle or random.
  - Example: `--policy whittle`

- **S - subsidy for Whittle index computation**
  - `--subsidy`: Specify the subsidy for Whittle index.
  - Example: `--subsidy 0.5`

- **D - discount factor**
  - `--discount_factor`: Set the discount factor for future rewards.
  - Example: `--discount_factor 0.9`

- **G - grouping option**
  - `--grouping`: Toggle grouping of patients by characteristics. Default is False.
  - Example: `--grouping`

- **R - number of runs**
  - `--num_runs`: Set the number of times to repeat the experiment.
  - Example: `--num_runs 5`
