# Data

This folder includes the data for all experiments used in the analysis directory:

| Dataset Name    |                 Description                 | 
|:----------------|:-------------------------------------------:|
| methods_main    |       Experiment with no added costs        |
| irl_validation  | Validation experiment with added time costs |

# How to run

For an experiment, run:
```
python src/download_exp.py -e <experiment name in yamls/experiments>
python src/preprocess_human_runs.py -e <experiment name in yamls/experiments>
```
