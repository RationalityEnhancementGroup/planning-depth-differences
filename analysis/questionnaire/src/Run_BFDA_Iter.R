devtools::load_all(paste0(getwd(), "/IRLQuestR"))

command_line_args <- commandArgs(TRUE)
yaml_file <- command_line_args[1]
num_job <- command_line_args[2]
args<-yaml::read_yaml(paste0(getwd(), "../inputs/power_analysis/", yaml_file, ".yaml"))

# res <- IRLQuestR::BFDA_Inclusion(args$design_prior, args$analysis_prior, args$coefficients, num_participants = args$num_participants, binned = args$binned, depth_vector =  c(0,1,2,5,10), num_repetitions=1, args$factors, args$base_factors, args$dep_factor, fit_path = NA, match_models = FALSE)
res <- IRLQuestR::BFDA(args$design_prior, args$analysis_prior, args$coefficients, num_participants = args$num_participants, binned = args$binned, depth_vector =  c(0,1,2,5,10), num_repetitions=1, args$factors, args$base_factors, args$dep_factor, null_region=c(-.1, .1))

dir.create("../data/bfda/")
write.csv(x=res$pos, file=paste0(getwd(), "../data/bfda/pos_", yaml_file, "_", num_job, ".csv"))
write.csv(x=res$neg, file=paste0(getwd(), "../data/bfda/neg_", yaml_file, "_", num_job, ".csv"))

