##################################################
###     How the functions should be used      ####
##################################################
library(rugarch)
source("R/gray_dgp.R")
source("R/gray_ml.R")




distribution <- "std"
returns <- read.csv("returns.csv", head = FALSE)
theta_hat2 <- fit_gray(returns$V1, distribution, 2)

 e <- rdist("std", 5000, mu = 0, sigma = sqrt(0.6), shape = 5)

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



library(MSGARCH)

returns <- read.csv("msgarch_julia_norm.csv", head = FALSE)

MSGARC_Spec = CreateSpec(variance.spec = list(model = c("sGARCH","sGARCH")),
                         switch.spec = list(do.mix = FALSE),
                         distribution.spec = list(distribution = c("norm", "norm")))

MSGARCH_fit = FitML(MSGARC_Spec,returns$V1)
MSGARCH_fit

# simulation from specification
spec <- CreateSpec()
par <- c(0.1, 0.05, 0.9, 0.2, 0.1, 0.8, 0.99, 0.01)
set.seed(1234)
sim <- simulate(object = spec, nsim = 1L, nahead = 10000, nburn = 500L, par = par)

fit <- FitML(spec = spec, data = sim$draw[,1])

write.csv(as.numeric(sim$draw[,1]), "msgarch_r_norm.csv")


fit2 <- FitMCMC(spec = spec, data = sim$draw[,1])