# Data

This folder includes the data for all experiments used in the analysis directory:

| Dataset Name   |                                 Description                                  | Source                                                                                                                                                                                                                                                                                                                | 
|:---------------|:----------------------------------------------------------------------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| methods_main   |                        Experiment with no added costs                        |                                                                                                                                                                                                                                                                                                                       |
| irl_validation |                 Validation experiment with added time costs                  |                                                                                                                                                                                                                                                                                                                       |
| quest_first    |     First run of questionnaire experiment (one session per participant)      |                                                                                                                                                                                                                                                                                                                       |
| quest_second   |    Second run of questionnaire experiment (two sessions per participant)     |                                                                                                                                                                                                                                                                                                                       |
| c1.1           | Data provided for an experiment where variance does not depend on node depth | Ruiqi He; From: He, R., Jain, Y. R., & Lieder, F. (2021). Measuring and modelling how people learn how to plan and how people adapt their planning strategies the to structure of the environment. In International Conference on Cognitive Modeling. Retrieved from https://is.mpg.de/uploads_file/attachment/attachment/671/20210720_ICCM_submission_final.pdf. |
| c2.1           |   Data provided for an experiment where variance decreases with node depth   | Ruiqi He; From: He, R., Jain, Y. R., & Lieder, F. (2021). Measuring and modelling how people learn how to plan and how people adapt their planning strategies the to structure of the environment. In International Conference on Cognitive Modeling. Retrieved from https://is.mpg.de/uploads_file/attachment/attachment/671/20210720_ICCM_submission_final.pdf. |

# How to run

For an experiment, run:
```
python src/download_exp.py -e <experiment name in yamls/experiments>
python src/preprocess_human_runs.py -e <experiment name in yamls/experiments>
```
