default:
	python src/compute_cronbachs_alpha.py -e QuestMain
	python src/get_descriptive_statistics.py -e QuestMain
	python ../methods/static/src/plot_bic.py -e QuestMain -s questionnaire
	python src/run_main_regressions.py -a main_experiment
	python src/run_main_regressions.py -a main_experiment_clicks
	python src/investigate_cm_relation.py
	python src/age_vs_planning.py
	python ../methods/static/src/find_stable_point_with_cm.py -e QuestMain -s questionnaire
	python src/generate_vif_table.py >> log/vif_table.stdout
