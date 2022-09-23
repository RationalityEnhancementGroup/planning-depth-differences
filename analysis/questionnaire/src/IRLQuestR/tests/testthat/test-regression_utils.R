context("Test the Regression Utils")

test_that("Test ability to get generate model formulas", {
  # Data needed for tests
  base_factors = "age + gender + IQ"
  factors <- c("AD","CIT","SW")
  dep_factor = "dep_var"
  results = c("dep_var ~ AD + age + gender + IQ + 1",
              "dep_var ~ SW + age + gender + IQ + 1",
              "dep_var ~ CIT + age + gender + IQ + 1",
              "dep_var ~ AD + SW + age + gender + IQ + 1",
              "dep_var ~ AD + CIT + age + gender + IQ + 1",
              "dep_var ~ CIT + SW + age + gender + IQ + 1",
              "dep_var ~ AD + CIT + SW + age + gender + IQ + 1",
              "dep_var ~ age + gender + IQ + 1")
  expect_equal(setequal(unlist(generate_models(factors, base_factors, dep_factor)$formulas), results), TRUE)
})

test_that("We can run at least one iteration of the analysis pipeline", {
  base_factors = c()
  factors <- c("AD","CIT","SW")
  dep_factor = "latent_variable"
  models <- generate_models(factors, base_factors, dep_factor)
  priors <- list("AD" = list("dist" = "normal", "params" = c(0,1)), "SW" = list("dist" = "inv_gamma", "params" = c(.5,.5)), "CIT" = list("dist" = "normal", "params" = c(0,1)))

  data <- generate_fake_data(list("AD"=0.5, "SW"=0.25, "CIT"=0.33), priors, num_participants = 250, binned = TRUE)

  ibf <- IRLQuestR::get_inclusion_bayes_factors(data, priors, factors, base_factors, dep_factor, fit_path = NA, match_models = FALSE)
  expect_equal(length(ibf),3)
})