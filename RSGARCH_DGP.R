#################################################
###  RSGARCH GPD: Simulate RSGARCH Models    ####
#################################################
library(rugarch)

#### Gray (1996)
simulate_gray <- function(n = 1000, distribution = "std", omega, alpha, beta, time_varying = TRUE, P, C = NULL, D = NULL, burnin = 500) {
    
    if (!time_varying & is.null(P)) stop("Transition matrix P should be provided")
    if (time_varying & (is.null(C) | is.null(D))) stop("Vectors C and D should be provided")

    ntot <- n + burnin
    k <- length(omega)
    e <- rdist(distribution, ntot)
    h <- matrix(NA, ncol = k + 1, nrow = ntot)
    r <- rep(NA, ntot)
    Pt <- rep(NA, ntot)
    h[1, 1:k] <- 1

    if (!time_varying) {
        p <- P[1, 1]
        q <- P[2, 2]
        Pt[1] <- (1 - q) / (2 - p - q)       ## P(St = 1) - Pag 683 Hamilon (1994)
        h[1, k + 1] <- Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2]
        r[1] <- e[1] * sqrt(h[1, k + 1])
        for (i in 2:ntot) {
            h[i, 1:k] <- omega + alpha * r[i - 1]^2 + beta * h[i - 1, k + 1]
            numA <- (1 - q) * ddist(distribution, r[i - 1], sigma = sqrt(h[i - 1, 2])) * (1 - Pt[i - 1])
            numB <- p * ddist(distribution, r[i - 1], sigma = sqrt(h[i - 1, 1])) * Pt[i - 1]
            deno <- ddist(distribution, r[i - 1], sigma = sqrt(h[i - 1, 1])) * Pt[i - 1] + 
                    ddist(distribution, r[i - 1], sigma = sqrt(h[i - 1, 2])) * (1 - Pt[i - 1])
            Pt[i]  <-  numA/deno + numB/deno
            h[i, k + 1] <- Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2]
            r[i] <- e[i] * sqrt(h[i, k + 1])
        }
    } else {
        p <- pnorm(C[1] + D[1]*0)             ## 0 é a melhor opção?
        q <- pnorm(C[2] + D[2]*0)             ## 0 é a melhor opção?
        Pt[1]  <- (1 - q) / (2 - p - q)       ## Unconditional Probability: P(St = 1) - Pag 683 Hamilon (1994)
        h[1, k + 1] <- Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2]
        r[1] <- e[1] * sqrt(h[1, k + 1])
        for (i in 2:ntot) {
            p <- pnorm(C[1] + D[1] * r[i - 1])
            q <- pnorm(C[2] + D[2] * r[i - 1])
            h[i, 1:k] <- omega + alpha * r[i - 1]^2 + beta * h[i - 1, k + 1]
            numA <- (1 - q) * ddist(distribution, r[i - 1], sigma = sqrt(h[i - 1, 2])) * (1 - Pt[i - 1])
            numB <- p * ddist(distribution, r[i - 1], sigma = sqrt(h[i - 1, 1])) * Pt[i - 1]
            deno <- ddist(distribution, r[i - 1], sigma = sqrt(h[i - 1, 1])) * Pt[i - 1] + 
                    ddist(distribution, r[i - 1], sigma = sqrt(h[i - 1, 2])) * (1 - Pt[i - 1])
            Pt[i]  <-  numA/deno + numB/deno
            h[i, k + 1] <- Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2]
            r[i] <- e[i] * sqrt(h[i, k + 1])
        }
    }
    return(list(r = r[(burnin + 1):ntot], h = h[(burnin + 1):ntot, ], Pt = Pt[(burnin + 1):ntot]))
}


