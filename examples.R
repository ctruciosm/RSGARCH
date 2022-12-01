##################################################
###     How the functions should be used      ####
##################################################
library(rugarch)
source("gray_dgp.R")
source("gray_ml.R")


omega <- c(0.18, 0.01)
alpha <- c(0.4, 0.1)
beta <- c(0.2, 0.7)
P <- matrix(c(0.90, 0.1, 0.03, 0.97), byrow = TRUE, ncol = 2)
distribution <-  "std"
dados <- simulate_gray(n = 5000, distribution, omega, alpha, beta, time_varying = FALSE, P = P)
r <- dados$r
k <- 2
par_ini = c(omega + runif(2, 0, 0.03), alpha + runif(2, 0, 0.03), beta + runif(2, 0, 0.03), 0.8, 0.96, 5)
gray_likelihood(par_ini, r, distribution, k)

theta_hat1 <- fit_gray(r, distribution, 2, par_ini)
theta_hat2 <- fit_gray(r, distribution, 2)

