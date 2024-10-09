# Indivudual-vs-Group-Health-Care-Restless-Bandits

inspired by: https://github.com/lily-x/online-rmab/tree/main/src


## Files
- `main.py` - main driver\
- `algorithms.py` - optimal (as of right now)\
- `compute_whittle.py` - computes whittle index\
- `simulator.py` - simulator using OpenAI's gym

## Running
To run in CLI use:  

```python
python main.py --num_arms [arms] --budget 3 --episodes [episodes]