#################################################
###  RSGARCH GPD: Simulate RSGARCH Models    ####
#################################################

#### Gray (1996)
omega <- c(0.01, 0.05)
alpha <- c(0.1, 0.07)
beta <- c(0.6, 0.88)
P <- matrix(c(0.7, 0.3, 0.3, 0.7), byrow = TRUE, ncol = 2)

simulate_gray <- function(n = 1000, distribution = "std", omega, alpha, beta, time_varying = TRUE, 
                          P, C = NULL, D = NULL, burnin = 500) {
    
    if (!time_varying & is.null(P)) stop("Transition matrix P should be provided")
    if (time_varying & (is.null(C) | is.null(D))) stop("Vectors C and D should be provided")

    ntot <- n + burnin
    k <- length(omega)
    e <- rugarch::rdist(distribution, ntot)
    h <- matrix(NA, ncol = k + 1, nrow = ntot)
    r <- rep(NA, ntot)
    Pt <- rep(NA, ntot)
    h[1, 1 : k] <- omega/(1 - alpha - beta)

    if (!time_varying) {
        p <- P[1, 1]
        q <- P[2, 2]
        Pt[1] <- P[1, 1]   # Qual utilizar aqui?
        h[1, k + 1] <- Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2]
        r[1] <- e[1] * sqrt(h[1, k + 1])
        for (i in 2: ntot) {
            h[i, 1:k] <- omega + alpha * r[i - 1]^2 + beta * h[i - 1, k + 1]
            numA <- (1 - q) * rugarch::ddist(distribution, r[i - 1], mu = 0, sigma = sqrt(h[i - 1, 2])) * (1 - Pt[i - 1])
            numB <- p * rugarch::ddist(distribution, r[i - 1], mu = 0, sigma = sqrt(h[i - 1, 1])) * Pt[i - 1]
            deno <- rugarch::ddist(distribution, r[i - 1], mu = 0, sigma = sqrt(h[i - 1, 1])) * Pt[i - 1] + 
                    rugarch::ddist(distribution, r[i - 1], mu = 0, sigma = sqrt(h[i - 1, 2])) * (1 - Pt[i - 1])
            Pt[i]  <-  numA/deno + numB/deno
            h[i, k + 1] <- Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2]
            r[i] <- e[i] * sqrt(h[i, k + 1])
        }
    } else {
        p <- qnorm(C[1] + D[1]*0) # 0 é a melhor opção?
        q <- qnorm(C[2] + D[2]*0) # 0 é a melhor opção?
        Pt[1]  <- p   # Qual utilizar aqui?
        h[1, k + 1] <- Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2]
        r[1] <- e[1] * sqrt(h[1, k + 1])
        for (i in 2: ntot) {
            p <- qnorm(C[1] + D[1] * r[i - 1])
            q <- qnorm(C[2] + D[2] * r[i - 1])
            h[i, 1:k] <- omega + alpha * r[i - 1]^2 + beta * h[i - 1, k + 1]
            numA <- (1 - q) * rugarch::ddist(distribution, r[i - 1], mu = 0, sigma = sqrt(h[i - 1, 2])) * (1 - Pt[i - 1])
            numB <- p * rugarch::ddist(distribution, r[i - 1], mu = 0, sigma = sqrt(h[i - 1, 1])) * Pt[i - 1]
            deno <- rugarch::ddist(distribution, r[i - 1], mu = 0, sigma = sqrt(h[i - 1, 1])) * Pt[i - 1] + 
                    rugarch::ddist(distribution, r[i - 1], mu = 0, sigma = sqrt(h[i - 1, 2])) * (1 - Pt[i - 1])
            Pt[i]  <-  numA/deno + numB/deno
            h[i, k + 1] <- Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2]
            r[i] <- e[i] * sqrt(h[i, k + 1])
        }
    }
    return(list(r = r[(burnin + 1): ntot], h = h[(burnin + 1): ntot, ], Pt = Pt[(burnin + 1): ntot]))
}

dados <- simulate_gray(n = 5000, "std", omega, alpha, beta, time_varying = FALSE, P)