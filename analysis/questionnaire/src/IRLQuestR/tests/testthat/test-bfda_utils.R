context("Test the BFDA Utils")

test_that("Binning works as expected", {
  expect_equal(bin_latent_variable(c(0, .4, 1.5, 4.5, 7.5, 40)), c(0, 0, 1, 2, 5, 10))
  expect_equal(bin_latent_variable(c(5,6,7,15,16,17), c(6, 16)), c(6,6,6,6,16,16))
  expect_equal(bin_latent_variable(c(-1, 0, 1, 2, 5, 10)), c(0, 0, 1, 2, 5, 10))
})

test_that("We can generate data", {
  priors_depth <- list("AD" = list("dist" = "normal", "params" = c(0,1)), "SW" = list("dist" = "inv_gamma", "params" = c(.5,.5)), "CIT" = list("dist" = "normal", "params" = c(0,1)))
  depth_vectors <- list(c(0,1,2,5,10), c(1,5,10), c(6,16))
  coefficient_vectors <- list(list("AD"=1, "SW"=1, "CIT"=1), list("AD"=0.5, "SW"=0.25, "CIT"=0.33))
  for (depth_vector in depth_vectors){
    for (coefficient_vector in coefficient_vectors){
      samples <- generate_fake_data(coefficient_vector, priors_depth, num_participants = 250, binned = FALSE, depth_vector =  depth_vector)
      samples_binned <- generate_fake_data(coefficient_vector, priors_depth, num_participants = 250, binned = TRUE, depth_vector =  depth_vector)

      expect_equal(samples$latent_variable, coefficient_vector$AD*samples$AD +coefficient_vector$SW*samples$SW + coefficient_vector$CIT*samples$CIT)
      expect_equal(setequal(unique(samples_binned$latent_variable), depth_vector), TRUE)
    }
  }
})