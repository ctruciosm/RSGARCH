##################################################
###  RSGARCH Estim: Estimate RSGARCH Models   ####
##################################################
# Common comments 
## Doubts
##################################################
library(rugarch)
source("gray_dgp.R")

#### Gray (1996)

gray_likelihood_time_varying <- function(par, r, distribution) {
    # par = matrix with collumns omega, alpha, beta, C, D
    k <- nrow(par)
    n <- length(r)
    h <- matrix(NA, ncol = k + 1, nrow = n)
    Pt <- rep(NA, n)

    p <- pnorm(par[1, 4] + par[1, 5]*0)
    q <- pnorm(par[2, 4] + par[2, 5]*0)
    Pt[1]  <- (1 - q) / (2 - p - q)

    h[1, 1:k] = var(r)
    h[1, k + 1] <- Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2]

    for (i in 2:n) {
        p <- pnorm(par[1, 4] + par[1, 5] * r[i - 1])
        q <- pnorm(par[2, 4] + par[2, 5] * r[i - 1])
        h[i, 1:k] = par[, 1] + par[, 2] * r[i - 1]^2 + par[, 3] * h[i - 1, k + 1]
        numA <- (1 - q) * rugarch::ddist(distribution, r[i - 1], mu = 0, sigma = sqrt(h[i - 1, 2])) * (1 - Pt[i - 1])
        numB <- p * rugarch::ddist(distribution, r[i - 1], mu = 0, sigma = sqrt(h[i - 1, 1])) * Pt[i - 1]
        deno <- rugarch::ddist(distribution, r[i - 1], mu = 0, sigma = sqrt(h[i - 1, 1])) * Pt[i - 1] + 
                rugarch::ddist(distribution, r[i - 1], mu = 0, sigma = sqrt(h[i - 1, 2])) * (1 - Pt[i - 1])
        Pt[i]  <-  numA/deno + numB/deno
        h[i, k + 1] <- Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2]
        log_lik[i - 1] <- log(rugarch::ddist(distribution, r[i - 1], mu = 0, sigma = sqrt(h[i - 1, 1])) * Pt[i - 1] + 
                              rugarch::ddist(distribution, r[i - 1], mu = 0, sigma = sqrt(h[i - 1, 2])) * (1 - Pt[i - 1]))
    }
    return(-mean(log_lik))
}



gray_likelihood <- function(par, r, distribution, k) {
    # par = numeric vector: omega, alpha, beta, p11, p22
    n <- length(r)
    h <- matrix(NA, ncol = k + 1, nrow = n)
    Pt <- rep(NA, n)
    log_lik <- rep(NA, n - 1)

    omega <- par[1:k]
    alpha <- par[(k + 1):(2 * k)]
    beta <- par[(2 * k + 1):(3 * k)]
    p <- par[3 * k + 1]
    q <- par[4 * k]
    nu <- ifelse(distribution == "std", par[4 * k + 1], 0)

    Pt[1]  <- (1 - q) / (2 - p - q)     ## Pi = P(St = 1) - Pag 683 Hamilton (1994)
    h[1, 1:k] = var(r)                  ## See Fig 2 in Gray (1996)
    h[1, k + 1] <- Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2]

    for (i in 2:n) {
        numA <- (1 - q) * ddist(distribution, r[i - 1], sigma = sqrt(h[i - 1, 2]), shape = nu) * (1 - Pt[i - 1])
        numB <- p * ddist(distribution, r[i - 1], sigma = sqrt(h[i - 1, 1]), shape = nu) * Pt[i - 1]
        deno <- ddist(distribution, r[i - 1], sigma = sqrt(h[i - 1, 1]), shape = nu) * Pt[i - 1] + 
                ddist(distribution, r[i - 1], sigma = sqrt(h[i - 1, 2]), shape = nu) * (1 - Pt[i - 1])
        Pt[i]  <-  numA/deno + numB/deno

        h[i, 1:k] = omega + alpha * r[i - 1]^2 + beta * h[i - 1, k + 1]
        h[i, k + 1] <- Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2]

        log_lik[i - 1] <- log(ddist(distribution, r[i], sigma = sqrt(h[i, 1]), shape = nu) * Pt[i] + 
                              ddist(distribution, r[i], sigma = sqrt(h[i, 2]), shape = nu) * (1 - Pt[i]))
    }
    return(-mean(log_lik))
}


fit_gray <- function(r, distribution = "norm", k, par_ini = NULL) { 
      if (is.null(par_ini)) {
            # Grid
      }
      # Constraints
      if (distribution == "norm") {
        ui <- rbind(diag(4 * k), matrix(0, ncol = 4 * k, nrow = 2))
        ui[4 * k + 1, 4 * k - 1] <- -1
        ui[4 * k + 2, 4 * k] <- -1
        ci <- c(rep(1e-06, 4 * k), -1, -1)
        parameters <- constrOptim(theta = par_ini, f = gray_likelihood, grad = NULL, ui = ui, ci = ci, r = r, distribution = distribution, k = k)
      }
      if (distribution == "std") {
        ui <- rbind(diag(4 * k + 1), matrix(0, ncol = 4 * k + 1, nrow = 2))
        ui[4 * k + 2, 4 * k - 1] <- -1
        ui[4 * k + 3, 4 * k] <- -1
        ci <- c(rep(1e-06, 4 * k), 4, -1, -1)
        parameters <- constrOptim(theta = par_ini, f = gray_likelihood, grad = NULL, ui = ui, ci = ci, r = r, distribution = distribution, k = k)
      }
  return(parameters)
}




omega <- c(0.18, 0.01)
alpha <- c(0.4, 0.1)
beta <- c(0.2, 0.7)
P <- matrix(c(0.90, 0.1, 0.03, 0.97), byrow = TRUE, ncol = 2)
distribution <-  "norm"
dados <- simulate_gray(n = 5000, distribution, omega, alpha, beta, time_varying = FALSE, P = P)
r <- dados$r
k <- 2
par_ini = c(omega + runif(2, 0, 0.03), alpha + runif(2, 0, 0.03), beta + runif(2, 0, 0.03), 0.8, 0.96)
gray_likelihood(par_ini, r, distribution, k)

library(microbenchmark)
microbenchmark(AAA = fit_gray(r, distribution, 2, par_ini), times = 1)