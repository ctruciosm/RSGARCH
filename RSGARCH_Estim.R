##################################################
###  RSGARCH Estim: Estimate RSGARCH Models   ####
##################################################

#### Gray (1996)
omega <- c(0.01, 0.05)
alpha <- c(0.1, 0.07)
beta <- c(0.6, 0.88)
C <- rnorm(2)
D <- rnorm(2)
par = as.matrix(data.frame(omega, alpha, beta, C, D))

gray_likelihood_time_varying <- function(par, r, distribution) {
    # par = matrix with collumns omega, alpha, beta, C, D
    k <- nrow(par)
    n <- length(r)
    h <- matrix(NA, ncol = k + 1, nrow = n)
    Pt <- rep(NA, n)

    p <- qnorm(par[1, 4] + par[1, 5]*0)
    q <- qnorm(par[2, 4] + par[2, 5]*0)
    Pt[1]  <- p
    h[1, 1:k] = par[, 1]/(1 - par[, 2] - par[, 3])
    h[1, k + 1] <- Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2]

    for (i in 2:n) {
        p <- qnorm(par[1, 4] + par[1, 5] * r[i - 1])
        q <- qnorm(par[2, 4] + par[2, 5] * r[i - 1])
        h[i, 1:k] = par[,1] + par[,2]*r[i - 1]^2 + par[, 3]*h[i - 1, k + 1]
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
