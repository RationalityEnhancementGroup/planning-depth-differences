# About

This directory contains scripts & notebooks that are associated with investigating the relationship between questionnaire measures and the output of our IRL method.

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
   
4. Get full inference for  questionnaire experiment (might need to run each submission one at a time, ~2 hours):

   Pilot:
     ```
     cd <path to irl-project>/cluster
     condor_submit_bid 1 submission_scripts/MPI-IS/04_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=dist_depth_eff_forw experiment=quest_first participants=quest_first output_string=quest_first
     condor_submit_bid 1 submission_scripts/MPI-IS/04_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=dist_depth_eff_forw experiment=quest_second participants=quest_second1 output_string=quest_second1
     condor_submit_bid 1 submission_scripts/MPI-IS/04_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=dist_depth_eff_forw experiment=quest_second participants=quest_second2 output_string=quest_second2
     ```
   Main:
     ```
     cd <path to irl-project>/cluster
     condor_submit_bid 1 submission_scripts/MPI-IS/04_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=dist_depth_eff_forw experiment=quest_main participants=quest_main1 output_string=quest_main1
     condor_submit_bid 1 submission_scripts/MPI-IS/04_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=dist_depth_eff_forw experiment=quest_main participants=quest_main2 output_string=quest_main2
     condor_submit_bid 1 submission_scripts/MPI-IS/04_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=dist_depth_eff_forw experiment=quest_main participants=quest_main3 output_string=quest_main3
     condor_submit_bid 1 submission_scripts/MPI-IS/04_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=dist_depth_eff_forw experiment=quest_main participants=quest_main4 output_string=quest_main4
     condor_submit_bid 1 submission_scripts/MPI-IS/04_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=dist_depth_eff_forw experiment=quest_main participants=quest_main5 output_string=quest_main5
     condor_submit_bid 1 submission_scripts/MPI-IS/04_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=dist_depth_eff_forw experiment=quest_main participants=quest_main6 output_string=quest_main6
     ```
5. Combine inferences by pid:

   Pilot:
     ```
     cd <path to irl-project>/cluster
     condor_submit_bid 2 submission_scripts/MPI-IS/05_Combine_Human.sub cost_function=dist_depth_eff_forw experiment=quest_first participant_file=quest_first output_string=quest_first simulated_cost_function=back_dist_depth_eff_forw
     for file_idx in {1..6};
        do condor_submit_bid 2 submission_scripts/MPI-IS/05_Combine_Human.sub cost_function=dist_depth_eff_forw experiment=quest_second participant_file=quest_second$file_idx output_string=quest_second$file_idx simulated_cost_function=back_dist_depth_eff_forw
     done;
     ```
   Main:
     ```
     cd <path to irl-project>/cluster
     for file_idx in {1..2};
       do condor_submit_bid 2 submission_scripts/MPI-IS/05_Combine_Human.sub cost_function=dist_depth_eff_forw experiment=quest_main participant_file=quest_main$file_idx output_string=quest_main$file_idx simulated_cost_function=back_dist_depth_eff_forw;
     done;
     ```
5. Get MAP file:
6. Calculate factor scores:
   ```
   python /home/vfelso/github/planning-depth-differences/analysis/questionnaire/src/calculate_factor_scores.py -e QuestMain 
   ```

   

| Section | Subsection |                                                              Heading / Description                                                              | How to replicate (locally)                                           |                                                  How to replicate (cluster)                                                   |
|--------:|:----------:|:-----------------------------------------------------------------------------------------------------------------------------------------------:|----------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------:|
|     ??  |    ??      |                                                             Participant strategies                                                              | `python find_stable_point_with_cm.py -e QuestMain -s questionnaires` | `condor_submit_bid 2 submission_scripts/MPI-IS/M_03_Report_CM_Strategies.sub experiment=QuestMain subdirectory=questionnaire` |
