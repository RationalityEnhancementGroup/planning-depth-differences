default:
	echo `date` > log/BIC_MainExperiment.stdout
	python src/plot_bic.py -e MainExperiment >> log/BIC_MainExperiment.stdout
	echo `date` > log/CM_MainExperiment.stdout
	python src/find_stable_point_with_cm.py -e MainExperiment >> log/CM_MainExperiment.stdout
	echo `date` > log/BIC_MainExperiment.stdout
	python src/plot_bic.py -e MainExperiment >> log/BIC_MainExperiment.stdout
	echo `date` > log/BMS_MainExperiment.stdout
	python src/plot_bms.py -e MainExperiment >> log/BMS_MainExperiment.stdout
	echo `date` > log/click_MainExperiment.stdout
	python src/plot_participant_average_likelihoods.py -e MainExperiment >> log/click_MainExperiment.stdout
	echo `date` > log/PPC_MainExperiment.stdout
	python src/posterior_predictive_check.py -e MainExperiment >> log/PPC_MainExperiment.stdout
	echo `date` > log/post_MainExperiment.stdout
	python src/investigate_structure_knowledge.py -e MainExperiment >> log/post_MainExperiment.stdout
	echo `date` > log/parameter_recovery_SoftmaxRecovery.stdout
	python src/simulated_parameter_recovery.py -e SoftmaxRecovery >> log/parameter_recovery_SoftmaxRecovery.stdout
	echo `date` > log/HDI_SoftmaxRecovery.stdout
	python src/plot_simulated_hdi.py -e SoftmaxRecovery >> log/HDI_SoftmaxRecovery.stdout
	echo `date` > log/ValidationExperiment.stdout
	python src/validation_experiment_validation.py -e ValidationExperiment >> log/ValidationExperiment.stdout
	echo `date` > log/regression_ValidationExperiment.stdout
	python src/regression_cross_validation.py >> log/regression_ValidationExperiment.stdout
	echo `date` > log/HDI_ValidationExperiment.stdout
	python src/plot_validation_hdi.py >> log/HDI_ValidationExperiment.stdout
	echo `date` > log/Simulated_BIC_vs_Parameters.stdout
	python src/plot_simulated_bic_vs_parameters.py >> log/Simulated_BIC_vs_Parameters.stdout
	echo `date` > log/Simulated_Info.stdout
	python src/plot_simulated_info.py >> log/Simulated_Info.stdout
	echo `date` > log/AllTrials.stdout
	python src/plot_all_vs_test.py >> log/AllTrials.stdout
	echo `date` > log/BIC_AllTrials.stdout
	python src/plot_bic.py -e AllTrials >> log/BIC_AllTrials.stdout
	echo `date` > log/HDI_MainExperiment.stdout
	python src/plot_main_hdi.py -e MainExperiment >> log/HDI_MainExperiment.stdout
	echo `date` > log/BIC_MainExperimentFull.stdout
	python src/plot_bic.py -e MainExperimentFull >> log/BIC_MainExperimentFull.stdout
	echo `date` > log/PPC_MainExperimentFull.stdout
	python src/posterior_predictive_check.py -e MainExperimentFull >> log/PPC_MainExperimentFull.stdout
no-matlab:
	echo `date` > log/BIC_MainExperiment.stdout
	python src/plot_bic.py -e MainExperiment >> log/BIC_MainExperiment.stdout
	echo `date` > log/BIC_MainExperimentFull.stdout
	python src/plot_bic.py -e MainExperimentFull >> log/BIC_MainExperimentFull.stdout
	echo `date` > log/BIC_c1.1.stdout
	python src/plot_bic.py -e c1.1 >> log/BIC_c1.1.stdout
	echo `date` > log/BIC_c2.1.stdout
	python src/plot_bic.py -e c2.1 >> log/BIC_c2.1.stdout
	echo `date` > log/CM_MainExperiment.stdout
	python src/find_stable_point_with_cm.py -e MainExperiment >> log/CM_MainExperiment.stdout
	echo `date` > log/BIC_MainExperiment.stdout
	python src/plot_bic.py -e MainExperiment >> log/BIC_MainExperiment.stdout
	echo `date` > log/click_MainExperiment.stdout
	python src/plot_participant_average_likelihoods.py -e MainExperiment >> log/click_MainExperiment.stdout
	echo `date` > log/PPC_MainExperiment.stdout
	python src/posterior_predictive_check.py -e MainExperiment >> log/PPC_MainExperiment.stdout
	echo `date` > log/PPC_MainExperimentFull.stdout
	python src/posterior_predictive_check.py -e MainExperimentFull >> log/PPC_MainExperimentFull.stdout
	echo `date` > log/post_MainExperiment.stdout
	python src/investigate_structure_knowledge.py -e MainExperiment >> log/post_MainExperiment.stdout
	echo `date` > log/parameter_recovery_SoftmaxRecovery.stdout
	python src/simulated_parameter_recovery.py -e SoftmaxRecovery >> log/parameter_recovery_SoftmaxRecovery.stdout
	echo `date` > log/HDI_SoftmaxRecovery.stdout
	python src/plot_simulated_hdi.py -e SoftmaxRecovery >> log/HDI_SoftmaxRecovery.stdout
	echo `date` > log/ValidationExperiment.stdout
	python src/validation_experiment_validation.py -e ValidationExperiment >> log/ValidationExperiment.stdout
	echo `date` > log/regression_ValidationExperiment.stdout
	python src/regression_cross_validation.py >> log/regression_ValidationExperiment.stdout
	echo `date` > log/HDI_ValidationExperiment.stdout
	python src/plot_validation_hdi.py >> log/HDI_ValidationExperiment.stdout
	echo `date` > log/Simulated_BIC_vs_Parameters.stdout
	python src/plot_simulated_bic_vs_parameters.py >> log/Simulated_BIC_vs_Parameters.stdout
	echo `date` > log/Simulated_Info.stdout
	python src/plot_simulated_info.py >> log/Simulated_Info.stdout
	echo `date` > log/AllTrials.stdout
	python src/plot_all_vs_test.py >> log/AllTrials.stdout
	echo `date` > log/BIC_AllTrials.stdout
	python src/plot_bic.py -e AllTrials >> log/BIC_AllTrials.stdout
	echo `date` > log/HDI_MainExperiment.stdout
	python src/plot_main_hdi.py -e MainExperiment >> log/HDI_MainExperiment.stdout
	echo `date` > log/BIC_MainExperimentFull.stdout
	python src/plot_bic.py -e MainExperimentFull >> log/BIC_MainExperimentFull.stdout
	echo `date` > log/PPC_MainExperimentFull.stdout
	python src/posterior_predictive_check.py -e MainExperimentFull >> log/PPC_MainExperimentFull.stdout
	echo `date` > log/Variance_Analyses.stdout
	python src/table_parameter_estimates.py >> log/Variance_Analyses.stdout
movefigs:
	cp figs/c1.1_bic.png ../../../../thesis_draft/figures/c1.1_bic.png
	cp figs/c2.1_bic.png ../../../../thesis_draft/figures/c2.1_bic.png
	cp figs/MainExperiment_average_ll.png ../../../../thesis_draft/figures/MainExperiment_average_ll.png
	cp figs/MainExperiment_bic.png ../../../../thesis_draft/figures/MainExperiment_bic.png
	cp figs/MainExperiment_score_lik.png ../../../../thesis_draft/figures/MainExperiment_score_lik.png
	cp figs/SoftmaxRecovery_additional_parameter_recovery.png ../../../../thesis_draft/figures/SoftmaxRecovery_additional_parameter_recovery.png
	cp figs/SoftmaxRecovery_cost_parameter_recovery.png ../../../../thesis_draft/figures/SoftmaxRecovery_cost_parameter_recovery.png
	cp figs/SoftmaxRecovery_parameter_recovery_correlation.png ../../../../thesis_draft/figures/SoftmaxRecovery_parameter_recovery_correlation.png
	cp figs/AllTrials_bic.png ../../../../thesis_draft/figures/AllTrials_bic.png
	cp figs/optimal_early_sim_depth_cost_weight_sim_given_cost.png ../../../../thesis_draft/figures/optimal_early_sim_depth_cost_weight_sim_given_cost.png
	cp figs/optimal_late_sim_depth_cost_weight_sim_given_cost.png ../../../../thesis_draft/figures/optimal_late_sim_depth_cost_weight_sim_given_cost.png
	cp figs/optimal_middle_sim_depth_cost_weight_sim_given_cost.png ../../../../thesis_draft/figures/optimal_middle_sim_depth_cost_weight_sim_given_cost.png
	cp figs/optimal_clicks_sim_depth_cost_weight_sim_given_cost.png ../../../../thesis_draft/figures/optimal_clicks_sim_depth_cost_weight_sim_given_cost.png
	cp figs/SoftmaxRecovery_sim_depth_cost_weight_sim_given_costxBIC.png ../../../../thesis_draft/figures/SoftmaxRecovery_sim_depth_cost_weight_sim_given_costxBIC.png