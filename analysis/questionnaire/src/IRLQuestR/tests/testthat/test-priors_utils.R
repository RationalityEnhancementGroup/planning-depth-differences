context("Test the Prior Utils")

test_that("Test prior works as expected", {
  # Data needed for tests
  priors_depth <- list("AD" = list("dist" = "normal", "params" = c(0,1)), "SW" = list("dist" = "inv_gamma", "params" = c(.5,.5)), "CIT" = list("dist" = "normal", "params" = c(0,1)))

  expect_equal(get_text_prior(priors_depth$AD), "normal(0,1)")
  expect_equal(get_text_prior(priors_depth$SW), "inv_gamma(0.5,0.5)")
  expect_equal(get_text_prior(priors_depth$CIT), "normal(0,1)")
})