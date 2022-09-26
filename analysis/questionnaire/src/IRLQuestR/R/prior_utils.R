# Title     : Prior Utils
# Objective : Helper functions for Priors
# Created by: Valkyrie Felso
# Created on: 30.09.21

#' Provides mapping of brms/stan -> plotting distributions
#'
#' This could be added as data, but for now it is added as a function.
#' To extend functionality, add more distributions in the list.
#'
#' @return list mapping brms/stan distribution name to a distribution
#'                               for plotting or drawing from (as text)
#' @export
plotting_dist_dict <- function(){
  return(c("normal" =  "stats::rnorm", "inv_gamma"= "MCMCpack::rinvgamma"))
}

#' Draws from list that contains dist and params
#'
#' @param prior_sub_list, contains mapping both a list of dist and params
#' e.g. list("dist" = "normal", "params" = c(0,1))
#' @param num_samples, optional number of samples to draw
#' @return Samples drawn from distributoin
#' @export
draws_from_prior <- function(prior_sub_list, num_samples=10000){
  inputs <- list(num_samples, prior_sub_list[["params"]])
  dist <- plotting_dist_dict()[prior_sub_list[["dist"]]]
  samples <- do.call(eval(parse(text=dist)), inputs)
  return(samples)
}

#' Plots a distribution to comply with WAMBS guidelines
#'
#' @param prior_sub_list, contains mapping both a list of dist and params
#' e.g. list("dist" = "normal", "params" = c(0,1))
#' @param num_samples, optional number of samples to draw
#' @param save_path, where to save png
#' @return Nothing, saves plot to save_path
#' @export
plot_dist <- function(prior_sub_list, num_samples=10000, save_path="./test.png"){
  samples <- draws_from_prior(prior_sub_list, num_samples)
  png(save_path)
  plot(density(samples))
  dev.off()
}

#' Loads in prior list
#'
#' @param prior_sub_list, contains mapping both a list of dist and params
#' e.g. list("dist" = "normal", "params" = c(0,1))
#' @return Text prior for brms
#' @export
get_text_prior <- function(prior_sub_list){
  inputs <- paste(unlist(prior_sub_list[["params"]]), collapse=",")
  return(paste0(prior_sub_list[["dist"]],"(",inputs,")"))
}

#' @export
get_brms_prior <- function(var, prior_sub_list){
  text_prior <- get_text_prior(prior_sub_list)
  return(brms::prior_string(text_prior, class = "b", coef=var))
}