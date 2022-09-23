# Title     : BFDA Utils
# Objective : Helper functions for Bayesian Factor Design Analysis
# Created by: Valkyrie Felso
# Created on: 30.09.21

#' @export
generate_fake_data <- function(coefficients, priors, num_participants = 250, binned = FALSE, depth_vector =  c(0,1,2,5,10)){
  # initiate empty samples list
  samples <- vector(mode="list", length = length(priors))
  names(samples) <- names(priors)

  regression_terms <- vector(mode="character",length(priors))
  for (prior_index in seq_along(priors)){
    prior_name <- names(priors)[prior_index]
    prior_list <- priors[[prior_name]]

    samples[[prior_name]] <- draws_from_prior(prior_list, num_samples = num_participants)

    regression_terms[prior_index] <- paste0(coefficients[prior_name],"*samples$",prior_name)
  }
  # turn samples into a dataframe
  samples <- data.frame(samples)

  regression_eqn <- paste(regression_terms, collapse=" + ")
  samples$latent_variable <- eval(parse(text=regression_eqn))
  if (binned==TRUE){
    samples$latent_variable <- bin_latent_variable(samples$latent_variable, depth_vector = depth_vector)
  }
  return(samples)
}


#' Bins latent variable data
#'
#' This is needed since our inferred data is binned, so for power analyses
#' we would also like to have less resolution.
#'
#' @param latent_variable_vector Vector of (not-yet-binned) latent variable values
#' @param depth_vector Depths to consider (e.g the depths we are using in inferrence code)
#' @return A numeric vector of binned data for the latent variable
#' @export
bin_latent_variable <- function(latent_variable_vector, depth_vector = c(0,1,2,5,10)){
  break_points <- c(-Inf, depth_vector[-1], Inf)
  binned_data <- cut(latent_variable_vector, break_points, labels=depth_vector, right=FALSE)
  return(depth_vector[binned_data])
}

#' @export
BFDA_Inclusion <- function(design_prior, analysis_prior, coefficients, num_participants = 250, binned = TRUE, depth_vector =  c(0,1,2,5,10), num_repetitions=10000, factors, base_factors, dep_factor, fit_path = "./", match_models = FALSE){
  considered_num_subjects <- seq.int(50, num_participants, 50)

  cols = c(factors, base_factors, 'considered_num')
  results <- data.frame(matrix(ncol = length(cols), nrow = num_repetitions * length(considered_num_subjects)))
  colnames(results) <- cols
  curr_idx = 1
  for (rep_num in seq_len(num_repetitions)){
    # Step 2 of BFDA (Step 1 is done by the inputs)
    generated_data <- generate_fake_data(coefficients, design_prior, num_participants = num_participants, binned = binned, depth_vector =  depth_vector)

    # Step 3 of BFDA
    for (considered_num in considered_num_subjects){
      bfs <- get_inclusion_bayes_factors(generated_data[1:considered_num,], analysis_prior, factors, base_factors, dep_factor, fit_path = fit_path, match_models = match_models)

      results[curr_idx,names(bfs)] <- bfs
      results[curr_idx,"considered_num"] <- considered_num

      curr_idx = curr_idx + 1
    }
  }
  return(results)}

#' @export
BFDA <- function(design_prior, analysis_prior, coefficients, num_participants = 250, binned = TRUE, depth_vector =  c(0,1,2,5,10), num_repetitions=10000, factors, base_factors, dep_factor, null_region=c(-.1, .1)){
  considered_num_subjects <- seq.int(50, num_participants, 50)
  model_params <- get_model_and_priors(analysis_prior, factors , base_factors, dep_factor)

  cols = c(factors, base_factors, 'considered_num')
  pos_results <- data.frame(matrix(ncol = length(cols), nrow = num_repetitions * length(considered_num_subjects)))
  neg_results <- data.frame(matrix(ncol = length(cols), nrow = num_repetitions * length(considered_num_subjects)))
  colnames(pos_results) <- cols
  colnames(neg_results) <- cols
  curr_idx = 1
  for (rep_num in seq_len(num_repetitions)){
    # Step 2 of BFDA (Step 1 is done by the inputs)
    generated_data <- generate_fake_data(coefficients, design_prior, num_participants = num_participants, binned = binned, depth_vector =  depth_vector)

    # Step 3 of BFDA
    for (considered_num in considered_num_subjects){
      curr_model <- brms::brm(formula = model_params$full_model, family = "poisson", chains=10, data = generated_data[1:considered_num,], save_pars = brms::save_pars(all = TRUE), prior = model_params$curr_priors)

      pos_bfs <- bayestestR::bayesfactor_parameters(curr_model, null = null_region, direction = ">")
      pos_results[curr_idx,names(pos_bfs)] <- pos_bfs
      pos_results[curr_idx,"considered_num"] <- considered_num

      neg_bfs <- bayestestR::bayesfactor_parameters(curr_model, null = null_region, direction = "<")
      neg_bfs[curr_idx,names(neg_bfs)] <- neg_bfs
      neg_bfs[curr_idx,"considered_num"] <- considered_num

      curr_idx = curr_idx + 1
    }
  }
  return(list(pos=pos_results, neg=neg_results))}
#