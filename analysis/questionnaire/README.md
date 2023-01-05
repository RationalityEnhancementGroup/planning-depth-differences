# About

This directory contains notebooks that are associated with investigating the relationship between questionnaire measures and the output of our IRL method.

## Set-up

### Generating inputs

0. Create log folder in this directory.
1. If not already done, generate the solution file from the questionnaire file used in the experiment:

```
cd <path to planning-depth-differences>/data
python src/construct_questionnaires_and_key.py
```

2. Download/process data according to the instructions in `data/README.md`.

3. Get Computational Microscope strategies (~2 hours):
   ```
   for experiment in quest_main quest_first quest_second;
       do condor_submit_bid 2 submission_scripts/MPI-IS/M_00_Get_CM_Strategies.sub experiment=$experiment
   done;
   ```
   
4. Get full inference for main questionnaire experiment:
   ```
   cd <path to irl-project>/cluster
   condor_submit_bid 2 submission_scripts/MPI-IS/04_Infer_Params.sub param_file=params_full_four0 experiment=quest_first cost_function=dist_depth_eff_forw
   condor_submit_bid 2 submission_scripts/MPI-IS/04_Infer_Params.sub param_file=params_full_four1 experiment=quest_first cost_function=dist_depth_eff_forw
   condor_submit_bid 2 submission_scripts/MPI-IS/04_Infer_Params.sub param_file=params_full_four0 experiment=quest_second cost_function=dist_depth_eff_forw
   condor_submit_bid 2 submission_scripts/MPI-IS/04_Infer_Params.sub param_file=params_full_four1 experiment=quest_second cost_function=dist_depth_eff_forw
   condor_submit_bid 2 submission_scripts/MPI-IS/04_Infer_Params.sub param_file=params_full_four0 experiment=quest_main cost_function=dist_depth_eff_forw
   condor_submit_bid 2 submission_scripts/MPI-IS/04_Infer_Params.sub param_file=params_full_four1 experiment=quest_main cost_function=dist_depth_eff_forw
   condor_submit_bid 2 submission_scripts/MPI-IS/04_Infer_Params.sub param_file=params_full_four0 experiment=quest_main cost_function=back_dist_depth_eff_forw
   condor_submit_bid 2 submission_scripts/MPI-IS/04_Infer_Params.sub param_file=params_full_four1 experiment=quest_main cost_function=back_dist_depth_eff_forw
   ```
5. Calculate factor scores:
   ```
   python /home/vfelso/github/planning-depth-differences/analysis/questionnaire/src/calculate_factor_scores.py -e QuestMain 
   ```

   

| Section | Subsection |                                                              Heading / Description                                                              | How to replicate (locally)                                           |                                                  How to replicate (cluster)                                                   |
|--------:|:----------:|:-----------------------------------------------------------------------------------------------------------------------------------------------:|----------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------:|
|     ??  |    ??      |                                                             Participant strategies                                                              | `python find_stable_point_with_cm.py -e QuestMain -s questionnaires` | `condor_submit_bid 2 submission_scripts/MPI-IS/M_03_Report_CM_Strategies.sub experiment=QuestMain subdirectory=questionnaire` |
