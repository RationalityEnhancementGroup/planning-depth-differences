# About

This directory contains notebooks that are associated with developing the IRL method itself.

# Cluster Computations

## Human data computations on cluster

- [x] Before running, make a log file in the `cluster` directory as well as the `analysis/methods/static` directory: `mkdir log`
- [x] Create a virtual environment as outlined in the top project folder both on your local machine and the cluster
- [x] Download and preprocess the participant data, outside the cluster and then transfer to cluster (if needed, see instructions for downloading data in the `data` subfolder):
   ```
   cd <path to project>
   source env/bin/activate
   cd <path to project>/data
   for experiment in methods_main irl_validation;
      do python src/download_exp.py -e $experiment
      python src/preprocess_human_runs.py -e $experiment
   done;
   ```
   - [x] Then, transfer by adding the subfolder in `data/processed/<experiment>` to git or via rsync/scp.
- [x] Get the computational microscope strategies (~2 hours, depends on number of participants):
   ```
   cd <path to project>/cluster
   condor_submit_bid 1 submission_scripts/MPI-IS/M_01_Get_CM_Strategies.sub experiment=methods_main save_path=/fast/vfelso
   ```
- [x] Infer parameters for a participant file (~2 hours per job and ~8000 jobs so will take a while):
   ```
   cd <path to project>/cluster
   condor_submit_bid 1 submission_scripts/MPI-IS/01_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=back_dist_depth_eff_forw experiment=methods_main participants=methods_main save_path=/fast/vfelso;
   condor_submit_bid 1 submission_scripts/MPI-IS/01_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=back_dist_depth_eff_forw experiment=methods_main participants=methods_main block=training save_path=/fast/vfelso;

   condor_submit_bid 1 submission_scripts/MPI-IS/01_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=back_dist_depth_eff_forw experiment=irl_validation participants=irl_validation1 save_path=/fast/vfelso
   condor_submit_bid 1 submission_scripts/MPI-IS/01_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=back_dist_depth_eff_forw experiment=irl_validation participants=irl_validation2 save_path=/fast/vfelso
   condor_submit_bid 1 submission_scripts/MPI-IS/01_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=back_dist_depth_eff_forw experiment=irl_validation participants=irl_validation3 save_path=/fast/vfelso
   condor_submit_bid 1 submission_scripts/MPI-IS/01_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=back_dist_depth_eff_forw experiment=irl_validation participants=irl_validation1 block=fairy save_path=/fast/vfelso
   condor_submit_bid 1 submission_scripts/MPI-IS/01_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=back_dist_depth_eff_forw experiment=irl_validation participants=irl_validation2 block=fairy save_path=/fast/vfelso
   condor_submit_bid 1 submission_scripts/MPI-IS/01_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=back_dist_depth_eff_forw experiment=irl_validation participants=irl_validation3 block=fairy save_path=/fast/vfelso
   ```
- [x] Combine the human inferences (can take 2-3 hours):
   ```
   cd <path to project>/cluster
   condor_submit_bid 1 submission_scripts/MPI-IS/05_Combine_Inferences.sub cost_function=back_dist_depth_eff_forw experiment=methods_main participant_file=methods_main simulated_cost_function=back_dist_depth_eff_forw memory=256000 save_path=/fast/vfelso;
   condor_submit_bid 1 submission_scripts/MPI-IS/05_Combine_Inferences.sub cost_function=back_dist_depth_eff_forw experiment=methods_main participant_file=methods_main simulated_cost_function=back_dist_depth_eff_forw block=training memory=256000 save_path=/fast/vfelso;
   
   for file_idx in {1..3};
        do condor_submit_bid 1 submission_scripts/MPI-IS/05_Combine_Inferences.sub cost_function=back_dist_depth_eff_forw experiment=irl_validation participant_file=irl_validation$file_idx simulated_cost_function=back_dist_depth_eff_forw memory=256000 save_path=/fast/vfelso;
        condor_submit_bid 1 submission_scripts/MPI-IS/05_Combine_Inferences.sub cost_function=back_dist_depth_eff_forw experiment=irl_validation participant_file=irl_validation$file_idx simulated_cost_function=back_dist_depth_eff_forw block=fairy memory=256000 save_path=/fast/vfelso;
   done;
   ```
- [x] Extract marginal and HDIs for human trajectories (~15 minutes):
   ```
   condor_submit_bid 1 submission_scripts/MPI-IS/04_Extract_Marginal_and_HDIs.sub experiment=methods_main cost_function=back_dist_depth_eff_forw participant_file=methods_main save_path=/fast/vfelso;
   condor_submit_bid 1 submission_scripts/MPI-IS/04_Extract_Marginal_and_HDIs.sub experiment=irl_validation cost_function=back_dist_depth_eff_forw participant_file=irl_validation save_path=/fast/vfelso;
   condor_submit_bid 1 submission_scripts/MPI-IS/04_Extract_Marginal_and_HDIs.sub experiment=irl_validation cost_function=back_dist_depth_eff_forw participant_file=irl_validation block=fairy save_path=/fast/vfelso;
   ```
- [x] Once the inference is done for the participants, get the best parameters by running (~30 minutes):
   ```
   condor_submit_bid 1 submission_scripts/MPI-IS/05_Get_Best_Parameters.sub experiment=methods_main base_cost_function=back_dist_depth_eff_forw cost_function=back_dist_depth_eff_forw participant_file=methods_main save_path=/fast/vfelso;
   condor_submit_bid 1 submission_scripts/MPI-IS/05_Get_Best_Parameters.sub experiment=methods_main base_cost_function=back_dist_depth_eff_forw cost_function=back_dist_depth_eff_forw participant_file=methods_main block=training,test save_path=/fast/vfelso;
   
   condor_submit_bid 1 submission_scripts/MPI-IS/05_Get_Best_Parameters.sub experiment=irl_validation base_cost_function=back_dist_depth_eff_forw cost_function=back_dist_depth_eff_forw participant_file=irl_validation save_path=/fast/vfelso;
   condor_submit_bid 1 submission_scripts/MPI-IS/05_Get_Best_Parameters.sub experiment=irl_validation base_cost_function=back_dist_depth_eff_forw cost_function=back_dist_depth_eff_forw participant_file=irl_validation block=fairy save_path=/fast/vfelso;
   ```
- [x] Pre-calculate trial-by-trial likelihoods (~30 minutes):
   ```
   condor_submit_bid 1 submission_scripts/MPI-IS/06_Calculate_Trial_By_Trial_Likelihood.sub experiment=MainExperiment subdirectory=methods/static save_path=/fast/vfelso;
   ```
   
## Simulated data computations on cluster

- [x] Prepare the parameters for simulating trajectories on the cluster (~15 minutes):
   ```
   cd <path to project>/cluster
   condor_submit_bid 5 -i -a request_memory=10000
   source ../env/bin/activate
   python src/get_parameters_for_simulations.py
   python src/get_parameters_for_optimal_simulation.py
   ```
   Once methods main best parameters are found:
   ```
   python src/get_human_parameters_for_simulation.py
   python src/get_human_parameters_for_simulation.py -e MainExperimentFull
   ```
- [x]  At the same time, you can simulate new trajectories on the cluster (~30 minutes):
   ```
   cd <path to project>/cluster
   condor_submit_bid 1 submission_scripts/MPI-IS/03_Simulate_Trajectories.sub param_file=reduced save_path=/fast/vfelso;
   condor_submit_bid 1 submission_scripts/MPI-IS/03_Simulate_Trajectories.sub param_file=no_added_cost policy=OptimalQ save_path=/fast/vfelso;
   ```
   Once methods main best parameters are found:
   ``` 
   condor_submit_bid 1 submission_scripts/MPI-IS/03_Simulate_Trajectories.sub param_file=participants_gamma,kappa save_path=/fast/vfelso;
   condor_submit_bid 1 submission_scripts/MPI-IS/03_Simulate_Trajectories.sub param_file=participants_back_added_cost,gamma,kappa save_path=/fast/vfelso;
   condor_submit_bid 1 submission_scripts/MPI-IS/03_Simulate_Trajectories.sub param_file=participants_gamma,given_cost,kappa save_path=/fast/vfelso;
   
   condor_submit_bid 1 submission_scripts/MPI-IS/03_Simulate_Trajectories.sub param_file=participants save_path=/fast/vfelso;
   condor_submit_bid 1 submission_scripts/MPI-IS/03_Simulate_Trajectories.sub param_file=participants_back_added_cost save_path=/fast/vfelso;
   condor_submit_bid 1 submission_scripts/MPI-IS/03_Simulate_Trajectories.sub param_file=participants_kappa save_path=/fast/vfelso;
   ```
- [x] Move trajectories over to archive them:
   ```
   mkdir ../data/processed/simulated/high_increasing/SoftmaxPolicy/participants_gamma,kappa_simulated_agents_back_dist_depth_eff_forw/
   cp data/trajectories/high_increasing/SoftmaxPolicy/participants_gamma,kappa/simulated_agents_back_dist_depth_eff_forw.csv ../data/processed/simulated/high_increasing/SoftmaxPolicy/participants_gamma,kappa_simulated_agents_back_dist_depth_eff_forw/mouselab-mdp.csv
   mkdir ../data/processed/simulated/high_increasing/SoftmaxPolicy/reduced_simulated_agents_back_dist_depth_eff_forw/ 
   cp data/trajectories/high_increasing/SoftmaxPolicy/reduced/simulated_agents_back_dist_depth_eff_forw.csv ../data/processed/simulated/high_increasing/SoftmaxPolicy/reduced_simulated_agents_back_dist_depth_eff_forw/mouselab-mdp.csv 
   mkdir ../data/processed/simulated/high_increasing/OptimalQ/no_added_cost_simulated_agents_back_dist_depth_eff_forw/
   cp data/trajectories/high_increasing/OptimalQ/no_added_cost/simulated_agents_back_dist_depth_eff_forw.csv ../data/processed/simulated/high_increasing/OptimalQ/no_added_cost_simulated_agents_back_dist_depth_eff_forw/mouselab-mdp.csv
   ```  
  - [x] Add these files to git to archive them.
- [x] Once the simulation jobs are done, on the cluster, start the jobs to infer parameters for the simulations (can take a while, like 24 hours):
   ```
   cd <path to project>/cluster
   for file_idx in {1..9};
      do condor_submit_bid 1 submission_scripts/MPI-IS/01_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=back_dist_depth_eff_forw experiment=high_increasing/SoftmaxPolicy/reduced/simulated_agents_back_dist_depth_eff_forw.csv participants=reduced$file_idx save_path=/fast/vfelso;
   done;
   
   condor_submit_bid 1 submission_scripts/MPI-IS/01_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=back_dist_depth_eff_forw experiment=high_increasing/SoftmaxPolicy/participants_gamma,kappa/simulated_agents_back_dist_depth_eff_forw.csv participants=participants save_path=/fast/vfelso;
   condor_submit_bid 1 submission_scripts/MPI-IS/01_Infer_Params.sub cost_function=back_dist_depth_eff_forw param_file=null experiment=high_increasing/OptimalQ/no_added_cost/simulated_agents_back_dist_depth_eff_forw.csv participants=no_added_cost save_path=/fast/vfelso;
   ```
- [x] Once the inference for the simulations are finished, combine the output files in the cluster (can take ~12+ hours):
   ```
   cd <path to project>/cluster
   for file_idx in {1..9};
      do condor_submit_bid 1 submission_scripts/MPI-IS/02_Combine_Inferences.sub cost_function=back_dist_depth_eff_forw experiment=simulated/high_increasing/SoftmaxPolicy/reduced participant_file=reduced$file_idx base_cost_function=back_dist_depth_eff_forw save_path=/fast/vfelso memory=256000;
   done;
  
   condor_submit_bid 1 submission_scripts/MPI-IS/02_Combine_Inferences.sub cost_function=back_dist_depth_eff_forw experiment=simulated/high_increasing/SoftmaxPolicy/participants_gamma,kappa participant_file=participants base_cost_function=back_dist_depth_eff_forw save_path=/fast/vfelso memory=256000;
   condor_submit_bid 1 submission_scripts/MPI-IS/02_Combine_Inferences.sub cost_function=back_dist_depth_eff_forw experiment=simulated/high_increasing/OptimalQ/no_added_cost participant_file=no_added_cost base_cost_function=back_dist_depth_eff_forw param_file=null save_path=/fast/vfelso
   ```
- [ ] Next, get the best parameters:
   ```
   cd <path to project>/cluster
   for file_idx in {1..9};
      do condor_submit_bid 1 submission_scripts/MPI-IS/05_Get_Best_Parameters.sub experiment=simulated/high_increasing/SoftmaxPolicy/reduced_simulated_agents_back_dist_depth_eff_forw base_cost_function=back_dist_depth_eff_forw cost_function=back_dist_depth_eff_forw participant_file=reduced$file_idx save_path=/fast/vfelso;
   done;
  
   condor_submit_bid 1 submission_scripts/MPI-IS/05_Get_Best_Parameters.sub experiment=simulated/high_increasing/SoftmaxPolicy/participants_gamma,kappa_simulated_agents_back_dist_depth_eff_forw base_cost_function=back_dist_depth_eff_forw cost_function=back_dist_depth_eff_forw participant_file=participants save_path=/fast/vfelso;
   condor_submit_bid 1 submission_scripts/MPI-IS/05_Get_Best_Parameters.sub experiment=simulated/high_increasing/OptimalQ/no_added_cost_simulated_agents_back_dist_depth_eff_forw base_cost_function=back_dist_depth_eff_forw cost_function=back_dist_depth_eff_forw participant_file=no_added_cost save_path=/fast/vfelso;
  ```
- [ ] Calculate simulated BIC data for later analyses:
   ```
   condor_submit_bid 1 submission_scripts/MPI-IS/M_02_Get_Simulated_BIC.sub`
   ```
- [ ] Extract marginal and HDIs for simulated data:
   ```
   condor_submit_bid 1 submission_scripts/MPI-IS/04_Extract_Marginal_and_HDIs.sub experiment=simulated/high_increasing/SoftmaxPolicy/reduced_simulated_agents_back_dist_depth_eff_forw  cost_function=back_dist_depth_eff_forw participant_file=reduced save_path=/fast/vfelso
   condor_submit_bid 1 submission_scripts/MPI-IS/04_Extract_Marginal_and_HDIs.sub experiment=simulated/high_increasing/SoftmaxPolicy/participants_gamma,kappa_simulated_agents_back_dist_depth_eff_forw  cost_function=back_dist_depth_eff_forw participant_file=participants save_path=/fast/vfelso
   ```

# Analyses

## Prerequisites

### SPM12 for BMS

- [x] Download SPM12 for Bayesian model selection and place it in the top level folder `spm12`: [https://www.fil.ion.ucl.ac.uk/spm/software/spm12/](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)
- [x] Get the Matlab API for python working on your computer for Bayesian model selection:
    - Locally:
      - Open MATLAB and get your MATLAB path by running `matlabroot`
      - Open your terminal, activate your virtual environment and then run:
        ```
        cd <matlab path>/extern/engines/python
        python setup.py install
        ```
      - If you have any problems with this step, see: https://www.mathworks.com/help/matlab/matlab_external/get-started-with-matlab-engine-for-python.html (Notice before running the `setup.py` script, you should have activated the virtual environment -- the step before this)
    - On the cluster: you should specify a different build location since you don't have write access: python setup.py build -b <location> (source: https://de.mathworks.com/matlabcentral/answers/324834-error-installing-matlab-engine-api-for-python)
    > ### Cluster tip
    > The path to various Matlab versions on the MPI-IS cluster is: `/is/software/matlab/linux/`.
    > With Matlab 2022B/Python 3.10 with pip 23, I had some problems (see: https://github.com/pypa/setuptools/issues/3237.)
    > Therefore I recommend:
    > Downgrade pip: `python -m pip install --force-reinstall pip==21.3` 
    > Go to MATLAB API path: `cd /is/software/matlab/linux/R2022b/extern/engines/python/`.
    > You could then run, e.g. `python -m pip install --use-deprecated=out-of-tree-build .` (non-pip:  `python setup.py build -b <build location> install`)

    > ### Other linux systems
    > Matlab versions will usually be installed in /usr/local/MATLAB/


## Reporting

|        Done         |  Section |           Subsection           |                                                        Heading / Description                                                        | How to replicate (locally)                                                                                                                                                                 |                                                                                 How to replicate (cluster)                                                                                  |
|:-------------------:|---------:|:------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------:|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                     |    Model |             Exp 1              |                               Participant behavior was relatively stable for the last half of trials                                | `python src/find_stable_point_with_cm.py -e MainExperiment > log/CM_MainExperiment.stdout`                                                                                                 |                                                        `condor_submit_bid 1 submission_scripts/MPI-IS/M_03_Report_CM_Strategies.sub`                                                        |
| :heavy_check_mark:  |    Model |        Exp 1 (Table 2)         |                                     The full cost model explains participants' data best (BIC)                                      | `python src/plot_bic.py -e MainExperiment  > log/BIC_MainExperiment.stdout`                                                                                                                |                                                                `condor_submit_bid 1 submission_scripts/MPI-IS/M_04_BIC.sub`                                                                 |
|                     |    Model |       Exp 1 (Figure 3a)        |                                     The full cost model explains participants' data best (BMS)                                      | `python src/plot_bms.py -e MainExperiment > log/BMS_MainExperiment.stdout`                                                                                                                 |                                                                `condor_submit_bid 1 submission_scripts/MPI-IS/M_05_BMS.sub`                                                                 |
| :heavy_check_mark:  |    Model |   Exp 1 (Figures 3b and 3c)    |                            The full cost model explains participants’ data best (Trial Log Likelihoods)                             | `python src/plot_participant_average_likelihoods.py -e MainExperiment > log/click_MainExperiment.stdout`                                                                                   |                                                         `condor_submit_bid 1 submission_scripts/MPI-IS/M_06_Click_Likelihoods.sub`                                                          |
| :heavy_check_mark:  |    Model |             Exp 1              |                                      Behavior of simulated agents matches participant behavior                                      | `python src/posterior_predictive_check.py -e MainExperiment > log/PPC_MainExperiment.stdout`                                                                                               |                                                     `condor_submit_bid 1 submission_scripts/MPI-IS/M_07_Posterior_Predictive_Check.sub`                                                     |
| :heavy_check_mark:  |    Model |             Exp 1              |                                Understanding the environment’s structure mitigates experienced cost                                 | `python src/investigate_structure_knowledge.py -e MainExperiment > log/post_MainExperiment.stdout`                                                                                         |                                                  `condor_submit_bid 1 submission_scripts/MPI-IS/M_08_Investigate_Structure_Knowledge.sub`                                                   |
|                     |   Method |          Simulations           |                                      Both cost parameters can be recovered from simulated data                                      | `python src/simulated_parameter_recovery.py -e SoftmaxRecovery > log/parameter_recovery_SoftmaxRecovery.stdout`                                                                            |                                                    `condor_submit_bid 1 submission_scripts/MPI-IS/M_09_Simulated_Parameter_Recovery.sub`                                                    |
|                     |   Method |          Simulations           |                          Our method outputs individual parameters with high confidence for simulated data                           | `python src/plot_simulated_hdi.py -e SoftmaxRecovery > log/HDI_SoftmaxRecovery.stdout`                                                                                                     |                                                           `condor_submit_bid 1 submission_scripts/MPI-IS/M_10_Simulated_HDI.sub`                                                            |
| :heavy_check_mark:  |   Method |             Exp 2              |    The manipulated cost of prospection parameter but not the mental effort parameter has an effect on planning operations chosen    | `python src/validation_experiment_validation.py -e ValidationExperiment > log/ValidationExperiment.stdout`                                                                                 |                                                       `condor_submit_bid 1 submission_scripts/MPI-IS/M_11_Experiment_Validation.sub`                                                        |
| :heavy_check_mark:  |   Method |             Exp 2              |     The cost of prospection parameter but not the mental effort parameter manipulation is reflected in block-wise MAP estimates     | `python src/regression_cross_validation.py > log/regression_ValidationExperiment.stdout`                                                                                                   |                                                    `condor_submit_bid 1 submission_scripts/MPI-IS/M_12_Regression_Cross_Validation.sub`                                                     |
| :heavy_check_mark:  |   Method |             Exp 2              |                   Our method can be used to classify individuals as exhibiting a high versus low cost of planning                   | `python src/plot_validation_hdi.py > log/HDI_ValidationExperiment.stdout`                                                                                                                  |                                                           `condor_submit_bid 1 submission_scripts/MPI-IS/M_13_Validation_HDI.sub`                                                           |
|                     | Appendix | Model, Exp 1 (Figures 1 and 2) |                                   The model infers certain sets of parameters better than others                                    | `python src/plot_simulated_bic_vs_parameters.py > log/Simulated_BIC_vs_Parameters.stdout && python src/plot_simulated_info.py > log/Simulated_Info.stdout`                                 |                                         Step 5 in Simulations section, `condor_submit_bid 1 submission_scripts/MPI-IS/M_14_Plot_Simulated_Info.sub`                                         |
| :heavy_check_mark:  | Appendix |    Model, Exp 1 (Figure 3)     |                                     How does the assumption of stationarity affect  model fit?                                      | `python src/plot_all_vs_test.py > log/AllTrials.stdout && python src/plot_bic.py -e AllTrials > log/BIC_AllTrials.stdout`                                                                  | `condor_submit_bid 1 submission_scripts/MPI-IS/M_15_Plot_All_vs_Test.sub && condor_submit_bid 1 submission_scripts/MPI-IS/M_04_BIC.sub experiment=AllTrials output_string=out_M4_AllTrials` |
|                     | Appendix |          Model, Exp 1          |                                              Participant strategies for Experiment #1                                               | Run above (CM_MainExperiment.stdout)                                                                                                                                                       |                                                                                          Run above                                                                                          |
| :heavy_check_mark:  | Appendix |         Method, Exp 2          |                         Participants with more spread out posteriors are still well explained by the model                          | `python src/plot_main_hdi.py -e MainExperiment > log/HDI_MainExperiment.stdout`                                                                                                            |                                                         `condor_submit_bid 1 submission_scripts/MPI-IS/M_16_MainExperiment_HDI.sub`                                                         |
|                     | Appendix |      Method, Simulations       |                                Regression results for parameter recovery from simulated trajectories                                | Run above                                                                                                                                                                                  |                                                                                          Run above                                                                                          |
| :heavy_check_mark:  | Appendix |          Model, Exp 1          |                                            Planning Operation Likelihoods for Each Model                                            | Run above (click_MainExperiment.stdout)                                                                                                                                                    |                                                                                          Run above                                                                                          |
|                     | Appendix |             Exp 2              |                        Both blocks in Experiment #2 are fit just as well as the test block in Experiment #1                         | Run above                                                                                                                                                                                  |                                                                                          Run above                                                                                          |                         
|                     | Appendix |             Exp 2              |                                                 Highest posterior density intervals                                                 | Run above                                                                                                                                                                                  |                                                                                          Run above                                                                                          |
| :heavy_check_mark:  | Appendix |             Exp 1              | Are simulated data from the fitted best model more highly correlated with human clicks than those of the fitted alternative models? | Run above (PPC_MainExperiment.stdout)                                                                                                                                                      |                                                                                          Run above                                                                                          |
|                     | Appendix |             Exp 1              |     Does the addition of a discount rate and power utility function explain the data better than the planning depth parameter?      | `python src/plot_bic.py -e MainExperimentFull > log/BIC_MainExperimentFull.stdout && python src/posterior_predictive_check.py -e MainExperimentFull > log/PPC_MainExperimentFull.stdout`   |                                                          `condor_submit_bid 1 submission_scripts/MPI-IS/M_04_BIC.sub experiment=MainExperimentFull`                                                           |

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