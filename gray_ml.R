##################################################
###  RSGARCH Estim: Estimate RSGARCH Models   ####
##################################################
# Common comments 
## Doubts
##################################################
library(rugarch)

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
        if (distribution == "std") {
          par_ini <- c(0.05, 0.15, 0.3, 0.1, 0.6, 0.2, 0.85, 0.92, 5)
        } else {
          par_ini <- c(0.05, 0.15, 0.3, 0.1, 0.6, 0.2, 0.85, 0.92)
        }
        ll = gray_likelihood(par_ini, r, distribution, k)
        for (i in 1:1000){
          # GRID is hard becausa we have 8/9 parameters
          omegas <- runif(2, 0.01, 0.3)
          alphas <- runif(2, 0.05, 0.5)
          betas <- c(runif(1, 0.4, max(0.41, 1 - alphas[1])), runif(1, 0.4, max(0.41, 1 - alphas[2])))
          p <- runif(2, 0.8, 0.99)
          if (distribution == "std") {
            par_random <- c(omegas, alphas, betas, p, runif(1, 4, 8))
          } else {
            par_random <- c(omegas, alphas, betas, p)
          }
          if (gray_likelihood(par_random, r, distribution, k) < ll){
            ll <- gray_likelihood(par_random, r, distribution, k)
            par_ini <- par_random
          }
        }                                 
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

