#################################################
###  RSGARCH GPD: Simulate RSGARCH Models    ####
#################################################

using Random, Distributions, Plots, GARCH

function simulate_gray(n, omega, alpha, beta, time_varying, P, C, D, burnin)
   
   if (!time_varying & isnothing(P)) 
    @error "Transition matrix P should be provided"
   end
   if (time_varying & (isnothing(C) || isnothing(D)))
    @error "Vectors c and d should be provided"
   end

   ntot = n + burnin;
   k = length(omega);
   e = rand(Normal(), ntot);
   h = zeros(ntot, k + 1);
   r = zeros(ntot);
   Pt = zeros(ntot);
   h[1, 1:k] .= 1.0;

   if (!time_varying)
    p = P[1, 1];
    q = P[2, 2];
    Pt[1] = (1 - q) / (2 - p - q);                  ## P(St = 1) - Pag 683 Hamilon (1994)
    h[1, k + 1] = Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2];
    r[1] = e[1] * sqrt(h[1, k + 1]);
    for i = 2:ntot
        h[i, 1:k] = omega .+ alpha.* r[i - 1]^2 + beta.* h[i - 1, k + 1];
        numA = (1 - q) * pdf(Normal(0, sqrt(h[i - 1, 2])), r[i - 1]) * (1 - Pt[i - 1]);
        numB = p * pdf(Normal(0, sqrt(h[i - 1, 1])), r[i - 1]) * Pt[i - 1];
        deno = pdf(Normal(0, sqrt(h[i - 1, 1])), r[i - 1]) * Pt[i - 1] + 
               pdf(Normal(0, sqrt(h[i - 1, 2])), r[i - 1]) * (1 - Pt[i - 1]);
        Pt[i] = numA/deno + numB/deno;
        h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
        r[i] = e[i] * sqrt(h[i, k + 1]);
    end
   else
    p = cdf(Normal(), C[1] + D[1] * 0);              ## 0 é a melhor opção?
    q = cdf(Normal(), C[2] + D[2] * 0);              ## 0 é a melhor opção?
    Pt[1] = (1 - q) / (2 - p - q);                   ## Unconditional Probability: P(St = 1) - Pag 683 Hamilon (1994)
    h[1, k + 1] = Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2];
    r[1] = e[1] * sqrt(h[1, k + 1]);
    for i = 2:ntot
        p = cdf(Normal(), C[1] + D[1] * r[i - 1]);
        q = cdf(Normal(), C[2] + D[2] * r[i - 1]);
        h[i, 1:k] = omega .+ alpha.* r[i - 1]^2 + beta.* h[i - 1, k + 1];
        numA = (1 - q) * pdf(Normal(0, sqrt(h[i - 1, 2])), r[i - 1]) * (1 - Pt[i - 1]);
        numB = p * pdf(Normal(0, sqrt(h[i - 1, 1])), r[i - 1]) * Pt[i - 1];
        deno = pdf(Normal(0, sqrt(h[i - 1, 1])), r[i - 1]) * Pt[i - 1] + 
               pdf(Normal(0, sqrt(h[i - 1, 2])), r[i - 1]) * (1 - Pt[i - 1]);
        Pt[i] = numA/deno + numB/deno;
        h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
        r[i] = e[i] * sqrt(h[i, k + 1]);
    end
   end
   return r[burnin + 1:end], h[burnin + 1:end, ], Pt[burnin + 1: end, ];
end


