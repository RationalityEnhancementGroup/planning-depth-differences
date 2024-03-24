default:
	python src/model_recovery.py 
	python src/difference_fixed_vs_random_effects.py
	python src/plot_parameter_values.py
	python src/simulated_parameter_recovery.py
	python src/table_parameter_estimates.py
	python src/plot_bic.py -e MainExperiment
	python src/find_stable_point_with_cm.py -e MainExperiment
	python src/plot_bic.py -e MainExperiment
	#python src/plot_bms.py -e MainExperiment
	python src/plot_participant_average_likelihoods.py -e MainExperiment
	python src/posterior_predictive_check.py -e MainExperiment
	python src/investigate_structure_knowledge.py -e MainExperiment
	python src/simulated_parameter_recovery.py -e SoftmaxRecovery
	python src/plot_simulated_hdi.py -e SoftmaxRecovery
	python src/validation_experiment_validation.py -e ValidationExperiment
	python src/regression_cross_validation.py
	python src/plot_validation_hdi.py
	python src/plot_simulated_bic_vs_parameters.py
	python src/plot_simulated_info.py
	python src/plot_all_vs_test.py
	python src/plot_bic.py -e AllTrials
	python src/plot_main_hdi.py -e MainExperiment
	python src/plot_bic.py -e MainExperimentFull
	python src/posterior_predictive_check.py -e MainExperimentFull
