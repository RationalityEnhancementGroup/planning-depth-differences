default:
	echo `date` > log/cronbachs_alpha.stdout
	python src/compute_cronbachs_alpha.py -e QuestMain >> log/cronbachs_alpha.stdout
	echo `date` > log/descriptive_statistics.stdout
	python src/get_descriptive_statistics.py -e QuestMain >> log/descriptive_statistics.stdout
	echo `date` > log/BIC_QuestMain.stdout
	python ../methods/static/src/plot_bic.py -e QuestMain -s questionnaire >> log/BIC_QuestMain.stdout
	echo `date` > log/cost_analyses.stdout
	python src/run_main_regressions.py -a main_experiment >> log/cost_analyses.stdout
	echo `date` > log/click_analyses.stdout
	python src/run_main_regressions.py -a main_experiment_clicks >> log/click_analyses.stdout
	echo `date` > log/cm_analyses.stdout
	python src/investigate_cm_relation.py >> log/cm_analyses.stdout
	echo `date` > log/age_vs_planning.stdout
	python src/age_vs_planning.py >> log/age_vs_planning.stdout
	echo `date` > log/CM_QuestMain.stdout
	python ../methods/static/src/find_stable_point_with_cm.py -e QuestMain -s questionnaire >> log/CM_QuestMain.stdout
	echo `date` > log/vif_table.stdout
	python src/generate_vif_table.py >> log/vif_table.stdout