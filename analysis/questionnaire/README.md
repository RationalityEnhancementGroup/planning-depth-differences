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
       do condor_submit_bid 2 submission_scripts/MPI-IS/M_01_Get_CM_Strategies.sub experiment=$experiment
   done;
   ```
   
4. Get full inference for  questionnaire experiment (might need to run each submission one at a time, ~2 hours):

   Pilot:
     ```
     cd <path to irl-project>/cluster
     condor_submit_bid 1 submission_scripts/MPI-IS/01_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=back_dist_depth_eff_forw experiment=quest_first participants=quest_first output_string=quest_first
     condor_submit_bid 1 submission_scripts/MPI-IS/01_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=back_dist_depth_eff_forw experiment=quest_second participants=quest_second1 output_string=quest_second1
     condor_submit_bid 1 submission_scripts/MPI-IS/01_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=back_dist_depth_eff_forw experiment=quest_second participants=quest_second2 output_string=quest_second2
     ```
   Main:
    ```
     cd <path to irl-project>/cluster
     for file_idx in {1..7};
        do condor_submit_bid 1 submission_scripts/MPI-IS/01_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=back_dist_depth_eff_forw experiment=quest_main participants=quest_main$file_idx save_path=/fast/vfelso;
     done
     ```
5. Combine inferences by pid:

   Pilot:
     ```
     cd <path to irl-project>/cluster
     condor_submit_bid 2 submission_scripts/MPI-IS/02_Combine_Inferences.sub cost_function=back_dist_depth_eff_forw experiment=quest_first participant_file=quest_first output_string=quest_first base_cost_function=back_dist_depth_eff_forw
     for file_idx in {1..2};
        do condor_submit_bid 2 submission_scripts/MPI-IS/02_Combine_Inferences.sub cost_function=back_dist_depth_eff_forw experiment=quest_second participant_file=quest_second$file_idx output_string=quest_second$file_idx base_cost_function=back_dist_depth_eff_forw
     done;
     ```
   Main:
     ```
     cd <path to irl-project>/cluster
     for file_idx in {1..7};
       do condor_submit_bid 2 submission_scripts/MPI-IS/02_Combine_Inferences.sub cost_function=back_dist_depth_eff_forw experiment=quest_main participant_file=quest_main$file_idx output_string=quest_main$file_idx base_cost_function=back_dist_depth_eff_forw save_path=/fast/vfelso;
     done;
     ```
6. Get MAP file:
   Pilot:
   ```
   condor_submit_bid 1 submission_scripts/MPI-IS/05_Get_Best_Parameters.sub experiment=quest_first base_cost_function=back_dist_depth_eff_forw cost_function=back_dist_depth_eff_forw participant_file=quest_first save_path=/fast/vfelso;
   condor_submit_bid 1 submission_scripts/MPI-IS/05_Get_Best_Parameters.sub experiment=quest_second base_cost_function=back_dist_depth_eff_forw cost_function=back_dist_depth_eff_forw participant_file=quest_second save_path=/fast/vfelso;
   ```
   Main:
   ```
   for file_idx in {1..7};
    do condor_submit_bid 1 submission_scripts/MPI-IS/05_Get_Best_Parameters.sub experiment=quest_main base_cost_function=back_dist_depth_eff_forw cost_function=back_dist_depth_eff_forw participant_file=quest_main$file_idx save_path=/fast/vfelso;
   done;
   ```
8. Calculate factor scores:
   ```
   python /home/vfelso/github/planning-depth-differences/analysis/questionnaire/src/calculate_factor_scores.py -e QuestMain 
   ```
   
Questionnaire main

|  Section |                  Subsection                  |                                             Heading / Description                                              | How to replicate (locally)                                                              |                                                  How to replicate (cluster)                                                   |
|---------:|:--------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|-----------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------:|
|  Methods |               Cronbach's alpha               |                                                   Materials                                                    | python src/compute_cronbachs_alpha.py -e QuestMain > log/cronbachs_alpha.txt            |                                                              NA                                                               |
|  Methods |       Motivation, effort, demographics       |                                                  Participants                                                  | python src/get_descriptive_statistics.py -e QuestMain > log/descriptive_statistics.txt  |                                                              NA                                                               |
|  Results |                     BIC                      |                   Participant data was best fit by the cost model with all five cost weights                   |                                                                                         |                                                                                                                               |
|  Results |           Model parameter analysis           |             Inferred model parameters could not explain the variance in the questionnaire measures             |                                                                                         |                                                                                                                               |
|  Results |         Model parameter analysis table       |                                   Variance explained by the model parameters                                   |                                                                                         |                                                                                                                               |
|  Results |                Click analysis                | Simpler behavioral measures \& participant strategy can explain the variance in the Cognitive Reflection Test. |                                                                                         |                                                                                                                               |
|  Results |      Computational microscope analysis       |                               Forward search is associated with increased age.                                 |                                                                                         |                                                                                                                               |
| Appendix | Computational microscope descriptive results |                                             Participant strategies                                             | `python find_stable_point_with_cm.py -e QuestMain -s questionnaires`                    | `condor_submit_bid 2 submission_scripts/MPI-IS/M_03_Report_CM_Strategies.sub experiment=QuestMain subdirectory=questionnaire` |
| Appendix |       Variance inflation factor Table        |                                                   VIF Table                                                    | `python generate_vif_table.py`                                                          |                                                                                                                               |
