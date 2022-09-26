# Title     : Regression Utils
# Objective : Helper functions for Bayesian Regression
# Created by: Valkyrie Felso
# Created on: 29.06.21

#' @export
generate_models <- function(factors, base_factors, dep_factor) {
  combos <- c()
  base_factors <- append(base_factors, 1)
  for (num_factors in seq_len(length(factors))){
    curr_combos <- arrangements::combinations(factors, num_factors, layout="list")
    combos <- append(combos, curr_combos)
  }
  combos <- lapply(combos, function(x){append(x, base_factors)})
  combos <- append(combos, list(base_factors))

  formulas <- lapply(combos,  function(x) {paste0(dep_factor, " ~ ", stringr::str_c(x, collapse = " + "))})
  return(list(formulas = formulas, factors = combos))
}

get_priors_for_model <- function(priors, factors){
  subset_priors = intersect(names(priors), unlist(factors))
  curr_priors<-c()
  for (prior_factor in subset_priors){
    curr_priors <- append(curr_priors, get_brms_prior(prior_factor, priors[[prior_factor]]))
  }
  return(curr_priors)
}

#' @export
fit_all_formulas <- function(models, data, priors, fit_path = "./") {
  fits <- list()
  for (formula_idx in seq_along(models$factors)){
    factors <- models$factors[formula_idx]
    formula <- models$formulas[formula_idx]

    curr_priors <- get_priors_for_model(priors, factors)

    if (is.na(fit_path)){
      curr_fit <- brms::brm(formula = unlist(formula), data = data, save_pars = brms::save_pars(all = TRUE), prior = curr_priors)
    } else {
      curr_fit <- brms::brm(formula = unlist(formula), data = data, save_pars = brms::save_pars(all = TRUE), prior = curr_priors, file = paste(fit_path, formula))
    }

    fits <- c(fits, list(curr_fit))
  }
  return(fits)
}

#' Calculates Inclusion Bayes Factors given Formulas, Data
#'
#' This function calculates Inclusion Bayes Factors for each variable.
#'
#' @param fitted_models Fitted models
#' @param match_models see bayestestR documentation
#' @return A list with model Bayes factors as 'model_bfs' and
#'                      inclusion Bayes factors as 'inclusion_bfs'
#' @export
get_inclusion_bayes_factors_for_models <- function(fitted_models, match_models = FALSE) {
  # bayes factors with fitted models
  model_bfs <- do.call(bayestestR::bayesfactor_models, fitted_models)
  inclusion_bfs <- bayestestR::bayesfactor_inclusion(model_bfs, match_models)

  return(list(model_bfs = model_bfs, inclusion_bfs = inclusion_bfs))
}

#' @export
get_inclusion_bayes_factors <- function(data, priors, factors, base_factors, dep_factor, fit_path = "./", match_models = FALSE){
  models <- generate_models(factors, base_factors, dep_factor)
  fitted_models <- fit_all_formulas(models, data, priors, fit_path = fit_path)
  bfs <- get_inclusion_bayes_factors_for_models(fitted_models, match_models = match_models)

  incl_bf <- list()
  for (factor in c(factors, base_factors)){
    incl_bf[[factor]] <- bfs$inclusion_bfs[factor,"log_BF"]
  }
  return(incl_bf)
}

#' @export
get_model_and_priors <- function(priors, factors, base_factors, dep_factor){#, family="gaussian", fit_path = "./"){
  models <- generate_models(factors, base_factors, dep_factor)
  full_model <- models$formulas[[length(models$formulas) - 1]]

  curr_priors = get_priors_for_model(priors, append(factors, base_factors))
  return(list(full_model=full_model, curr_priors=curr_priors))}
