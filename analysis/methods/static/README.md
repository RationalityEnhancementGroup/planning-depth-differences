# About

This directory contains notebooks that are associated with developing the IRL method itself.

# Cluster Computations

## Human data computations on cluster

0. Before running, make a log file in the `cluster` directory as well as the `analysis/methods/static` directory: `mkdir log`
1. Create a virtual environment as outlined in the top project folder both on your local machine and the cluster
2. Download and preprocess the participant data, outside the cluster and then transfer to cluster (if needed, see instructions for downloading data in the `data` subfolder):
   ```
   cd <path to project>
   source env/bin/activate
   cd <path to project>/data
   for experiment in methods_main irl_validation;
      do python src/download_exp.py -e $experiment
      python src/preprocess_human_runs.py -e $experiment
   done;
   ```
   Then, transfer by adding the subfolder in `data/processed/<experiment>` to git or via rsync/scp.
3. Get the computational microscope strategies (~2 hours, depends on number of participants):
   ```
   for experiment in methods_main irl_validation;
       do condor_submit_bid 1 submission_scripts/MPI-IS/M_00_Get_CM_Strategies.sub experiment=$experiment save_path=/fast/vfelso
   done;
   ```
   If needed, move the files to your local computer.
5. Infer parameters for a participant file (~2 hours, will need to run each by itself):
   ```
   cd <path to irl-project>/cluster
   condor_submit_bid 2 submission_scripts/MPI-IS/04_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=back_dist_depth_eff_forw experiment=methods_main participants=methods_main save_path=/fast/vfelso;

   condor_submit_bid 2 submission_scripts/MPI-IS/04_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=dist_depth_eff_forw experiment=irl_validation participants=irl_validation1 save_path=/fast/vfelso
   condor_submit_bid 2 submission_scripts/MPI-IS/04_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=dist_depth_eff_forw experiment=irl_validation participants=irl_validation2 save_path=/fast/vfelso
   condor_submit_bid 2 submission_scripts/MPI-IS/04_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=dist_depth_eff_forw experiment=irl_validation participants=irl_validation3 save_path=/fast/vfelso
   condor_submit_bid 2 submission_scripts/MPI-IS/04_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=dist_depth_eff_forw experiment=irl_validation participants=irl_validation1 block=fairy save_path=/fast/vfelso
   condor_submit_bid 2 submission_scripts/MPI-IS/04_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=dist_depth_eff_forw experiment=irl_validation participants=irl_validation2 block=fairy save_path=/fast/vfelso
   condor_submit_bid 2 submission_scripts/MPI-IS/04_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=dist_depth_eff_forw experiment=irl_validation participants=irl_validation3 block=fairy save_path=/fast/vfelso
   ```
6. Combine the human inferences (2-3 hours):
   ```
   cd <path to irl-project>/cluster
   condor_submit_bid 2 submission_scripts/MPI-IS/05_Combine_Human.sub cost_function=back_dist_depth_eff_forw experiment=methods_main participant_file=methods_main simulated_cost_function=back_dist_depth_eff_forw save_path=/fast/vfelso;
   condor_submit_bid 2 submission_scripts/MPI-IS/05_Combine_Human.sub cost_function=dist_depth_eff_forw experiment=methods_main participant_file=methods_main simulated_cost_function=back_dist_depth_eff_forw save_path=/fast/vfelso;
   
   for file_idx in {1..3};
        do condor_submit_bid 2 submission_scripts/MPI-IS/05_Combine_Human.sub cost_function=dist_depth_eff_forw experiment=irl_validation participant_file=irl_validation$file_idx simulated_cost_function=back_dist_depth_eff_forw save_path=/fast/vfelso;
        condor_submit_bid 2 submission_scripts/MPI-IS/05_Combine_Human.sub cost_function=dist_depth_eff_forw experiment=irl_validation participant_file=irl_validation$file_idx simulated_cost_function=back_dist_depth_eff_forw block=fairy save_path=/fast/vfelso;
   done;
   ```
7. Once the inference is done for the participants, get the best parameters by running (~30 minutes):
   ```
   condor_submit_bid 1 submission_scripts/MPI-IS/M_01_Get_MAP_File_by_PID.sub experiment=methods_main simulated_cost_function=back_dist_depth_eff_forw cost_function=back_dist_depth_eff_forw participant_file=methods_main save_path=/fast/vfelso;
   condor_submit_bid 1 submission_scripts/MPI-IS/M_01_Get_MAP_File_by_PID.sub experiment=irl_validation simulated_cost_function=back_dist_depth_eff_forw cost_function=dist_depth_eff_forw participant_file=irl_validation save_path=/fast/vfelso;
   ```
    
8. Extract marginal and MLEs for human trajectories (~1 hour):
   ```
   condor_submit_bid 2 submission_scripts/MPI-IS/10_Extract_Marginal_and_MLEs_Human.sub experiment=ValidationExperiment
   condor_submit_bid 2 submission_scripts/MPI-IS/10_Extract_Marginal_and_MLEs_Human.sub experiment=MainExperiment
   ```
9. Pre-calculate trial-by-trial likelihoods (~30 minutes):
   ```
   condor_submit_bid 2 submission_scripts/MPI-IS/11_Calculate_Trial_By_Trial_Likelihood.sub experiment=TrialByTrial
   ```
   
## Simulated data computations on cluster

1. At the same time, you can simulate new trajectories on the cluster (~90 minutes):
   ```
   cd <path to irl-project>/cluster
   condor_submit_bid 2 submission_scripts/MPI-IS/06_Simulate_Optimal.sub param_file=params_full_four0
   condor_submit_bid 2 submission_scripts/MPI-IS/06_Simulate_Optimal.sub param_file=params_full_four1
   condor_submit_bid 2 submission_scripts/MPI-IS/06_Simulate_Random.sub
   condor_submit_bid 2 submission_scripts/MPI-IS/06_Simulate_Softmax.sub param_file=params_full_four0
   condor_submit_bid 2 submission_scripts/MPI-IS/06_Simulate_Softmax.sub param_file=params_full_four1
   ```
2. Once the simulation jobs are done, on the cluster, start the jobs to infer parameters for the simulations (~30 minutes):
   > **_Note_**: you should not run any other jobs at the same time, and should run each job in this step one at a time given the maximum number of jobs per user.
   ```
   cd <path to irl-project>/cluster
   condor_submit_bid 2 submission_scripts/MPI-IS/07_Submit_Inferrences.sub policy=OptimalQ
   condor_submit_bid 2 submission_scripts/MPI-IS/07_Submit_Inferrences.sub policy=SoftmaxPolicy
   condor_submit_bid 1 submission_scripts/MPI-IS/07_Infer_Simulated_Params_start.sub policy=SoftmaxPolicy param_file=params_full_four0 simulated_reward_line=0

   ```
   This runs quite a few jobs, so depending on activity in the cluster could take more time. 
3. Once the inference for the simulations are finished, combine the output files in the cluster (~12+ hours, 16 hours when tested):
    ```
    cd <path to irl-project>/cluster
    condor_submit_bid 2 submission_scripts/MPI-IS/08_Combine_Inferred.sub policy=OptimalQ
    condor_submit_bid 2 submission_scripts/MPI-IS/08_Combine_Inferred_by_Temp.sub policy=SoftmaxPolicy
    ```
4. Next:
    ```
    cd <path to irl-project>/cluster
    condor_submit_bid 2 submission_scripts/MPI-IS/M_01_Get_MAP_Simulated_by_Param.sub policy=OptimalQ
    cat parameters/temperatures/partial.txt | while read line 
    do
       condor_submit_bid 2 submission_scripts/MPI-IS/M_01_Get_MAP_Simulated_by_Temp.sub temp=$line policy=SoftmaxPolicy;
    done;
    ```
5. Calculate Optimal BIC data for later analyses:
   ```
   condor_submit_bid 2 submission_scripts/MPI-IS/M_02_Get_Optimal_BIC.sub`
   ```
6. Combine trajectories for later analyses:
   ```
   condor_submit_bid 2 submission_scripts/MPI-IS/09_Combine_Trajectories.sub policy=SoftmaxPolicy -append request_memory=90000
   condor_submit_bid 2 submission_scripts/MPI-IS/09_Combine_Trajectories.sub policy=RandomPolicy
   condor_submit_bid 2 submission_scripts/MPI-IS/09_Combine_Trajectories.sub
   ```
7. Extract marginal and MLEs for Optimal and Softmax trajectories:
   ```
   condor_submit_bid 2 submission_scripts/MPI-IS/10_Extract_Marginal_and_MLEs_Softmax.sub
   condor_submit_bid 2 submission_scripts/MPI-IS/10_Extract_Marginal_and_MLEs_Optimal.sub
   ```

# Analyses

## Prerequisites

### SPM12 for BMS

1. Download SPM12 for Bayesian model selection and place it in `irl-project/spm12`: [https://www.fil.ion.ucl.ac.uk/spm/software/spm12/](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)
2. Get the Matlab API for python working on your computer for Bayesian model selection:
    - Locally:
      - Open MATLAB and get your MATLAB path by running `matlabroot`
      - Open your terminal, activate your virtual environment and then run:
        ```
        cd <matlab path>/extern/engines/python
        python setup.py install
        ```
      - If you have any problems with this step, see: https://www.mathworks.com/help/matlab/matlab_external/get-started-with-matlab-engine-for-python.html (Notice before running the `setup.py` script, you should have activated the virtual environment -- the step before this)
    - On the cluster: you should specify a different build location since you don't have write acccess: python setup.py build -b <location> (source: https://de.mathworks.com/matlabcentral/answers/324834-error-installing-matlab-engine-api-for-python)
> # Cluster tip
> The path to various Matlab versions on the MPI-IS cluster is: `/is/software/matlab/linux/`.
> So the above could be `cd /is/software/matlab/linux/R2022a/extern/engines/python/`.

> # Other linux systems
> Matlab versions will usually be installed in /usr/local/MATLAB/


## Reporting

|   Section |           Subsection           |                                                              Heading / Description                                                               | How to replicate (locally)                                                                                      |                                                                                  How to replicate (cluster)                                                                                  |
|----------:|:------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------:|-----------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     Model |             Exp 1              |                                      Participant behavior was relatively stable for the last half of trials                                      | `python find_stable_point_with_cm.py -e MainExperiment`                                                         |                                                        `condor_submit_bid 2 submission_scripts/MPI-IS/M_03_Report_CM_Strategies.sub`                                                         |
|     Model |        Exp 1 (Table 2)         |          The model containing both a cost of prospection parameter and a mental effort parameter explains participants’ data best (BIC)          | `python plot_bic.py -e MainExperiment`                                                                          |                                                                 `condor_submit_bid 2 submission_scripts/MPI-IS/M_04_BIC.sub`                                                                 |
|     Model |       Exp 1 (Figure 3a)        |          The model containing both a cost of prospection parameter and a mental effort parameter explains participants’ data best (BMS)          | `python plot_bms.py -e MainExperiment`                                                                          |                                                                 `condor_submit_bid 2 submission_scripts/MPI-IS/M_05_BMS.sub`                                                                 |
|     Model |   Exp 1 (Figures 3b and 3c)    | The model containing both a cost of prospection parameter and a mental effort parameter explains participants’ data best (Trial Log Likelihoods) | `python plot_participant_average_likelihoods.py -e TrialByTrial`                                                |                                                          `condor_submit_bid 2 submission_scripts/MPI-IS/M_06_Click_Likelihoods.sub`                                                          |
|     Model |             Exp 1              |                                            Behavior of simulated agents matches participant behavior                                             | `python plot_human_info.py -e MainExperiment`                                                                   |                                                     `condor_submit_bid 2 submission_scripts/MPI-IS/M_07_Posterior_Predictive_Check.sub`                                                      |
|     Model |             Exp 1              |                                       Understanding the environment’s structure mitigates experienced cost                                       | `python investigate_structure_knowledge.py -e TrialByTrial`                                                     |                                                   `condor_submit_bid 2 submission_scripts/MPI-IS/M_08_Investigate_Structure_Knowledge.sub`                                                   |
|    Method |          Simulations           |                                            Both cost parameters can be recovered from simulated data                                             | `python simulated_parameter_recovery.py -e SoftmaxRecovery`                                                     |                                                    `condor_submit_bid 2 submission_scripts/MPI-IS/M_09_Simulated_Parameter_Recovery.sub`                                                     |
|    Method |          Simulations           |                                 Our method outputs individual parameters with high confidence for simulated data                                 | `python plot_simulated_hdi.py -e SoftmaxRecovery`                                                               |                                                            `condor_submit_bid 2 submission_scripts/MPI-IS/M_10_Simulated_HDI.sub`                                                            |
|    Method |             Exp 2              |          The manipulated cost of prospection parameter but not the mental effort parameter has an effect on planning operations chosen           | `python validation_experiment_validation.py -e ValidationCostModel`                                             |                                                        `condor_submit_bid 2 submission_scripts/MPI-IS/M_11_Experiment_Validation.sub`                                                        |
|    Method |             Exp 2              |           The cost of prospection parameter but not the mental effort parameter manipulation is reflected in block-wise MAP estimates            | `python regression_cross_validation.py -e ValidationExperiment`                                                 |                                                     `condor_submit_bid 2 submission_scripts/MPI-IS/M_12_Regression_Cross_Validation.sub`                                                     |
|    Method |             Exp 2              |                         Our method can be used to classify individuals as exhibiting a high versus low cost of planning                          | `python plot_validation_hdi.py -e ValidationExperiment`                                                         |                                                           `condor_submit_bid 2 submission_scripts/MPI-IS/M_13_Validation_HDI.sub`                                                            |
|  Appendix | Model, Exp 1 (Figures 1 and 2) |                                          The model infers certain sets of parameters better than others                                          | `python get_optimal_bic.py -e OptimalBIC && python plot_simulated_info.py`                                      |                                         Step 5 in Simulations section, `condor_submit_bid 2 submission_scripts/MPI-IS/M_14_Plot_Simulated_Info.sub`                                          |
|  Appendix |    Model, Exp 1 (Figure 3)     |                                            How does the assumption of stationarity affect  model fit?                                            | `python plot_all_vs_test.py && python plot_bic.py -e MainExperiment`                                            | `condor_submit_bid 2 submission_scripts/MPI-IS/M_15_Plot_All_vs_Test.sub && condor_submit_bid 2 submission_scripts/MPI-IS/M_04_BIC.sub experiment=AllTrials output_string=out_M4_AllTrials`  |
|  Appendix |          Model, Exp 1          |                                                     Participant strategies for Experiment #1                                                     | `python find_stable_point_with_cm.py -e MainExperiment`                                                         |                                                                                          Run above                                                                                           |
|  Appendix |         Method, Exp 2          |                                Participants with more spread out posteriors are still well explained by the model                                | `python plot_main_hdi.py -e MainExperiment`                                                                     |                                                         `condor_submit_bid 2 submission_scripts/MPI-IS/M_16_MainExperiment_HDI.sub`                                                          |
|  Appendix |      Method, Simulations       |                                      Regression results for parameter recovery from simulated trajectories                                       | `python simulated_parameter_recovery.py -e SoftmaxRecovery`                                                     |                                                                                          Run above                                                                                           |
|  Appendix |          Model, Exp 1          |                                                  Planning Operation Likelihoods for Each Model                                                   | `python plot_participant_average_likelihoods.py -e TrialByTrial`                                                |                                                                                          Run above                                                                                           |
|  Appendix |             Exp 2              |                               Both blocks in Experiment #2 are fit just as well as the test block in Experiment #1                               | `python regression_cross_validation.py -e ValidationExperiment`                                                 |                                                   `condor_submit_bid 2 submission_scripts/MPI-IS/M_12_Regression_Cross_Validation.sub`                                                      |                         
|  Appendix |             Exp 2              |                                                       Highest posterior density intervals                                                        | `python plot_validation_hdi.py -e ValidationExperiment`                                                         |                                                           `condor_submit_bid 2 submission_scripts/MPI-IS/M_13_Validation_HDI.sub`                                                           |

> # Tip for running locally / debugging
> In order to have all the files the code is expecting, run these commands:
>  ```
> cd <path to irl-project on local computer>
> rsync -aPzr <user>@login.cluster.is.localnet:<path to irl-project on cluster>/irl-project/analysis/methods/static/log/ analysis/methods/static/log
> > rsync -aPzr <user>@login.cluster.is.localnet:<path to irl-project on cluster>/irl-project/analysis/methods/static/data/ analysis/methods/static/data
> rsync -aPzr <user>@login.cluster.is.localnet:<path to irl-project on cluster>/irl-project/analysis/methods/static/figs/ analysis/methods/static/figs
> rsync -aPzr <user>@login.cluster.is.localnet:<path to irl-project on cluster>/irl-project/cluster/data/OptimalQ cluster/data/OptimalQ
> rsync -aPzr --include "*/*mle_and_map*" --include "*/" --exclude "*" <user>@login.cluster.is.localnet:<path to irl-project on cluster>/irl-project/data/processed/ data/processed
> rsync -aPzr --include "*/*.feather" --include "*/" --exclude "*" <user>@login.cluster.is.localnet:<path to irl-project on cluster>/irl-project/cluster/data/logliks/ cluster/data/logliks

> ```
>  > **Note**: you can always check how large a directory is with:
>  > ```du -sh```
>  > or the files in a directory with:
>  > ```ls -ltha```
>  > These directories can be large (many GBs) so check that you have enough space on your local computer or do more of the analysis steps on the cluster.
>  > So if you aren't sure if you have room on your machine check how large the directory/file is on the cluster first!