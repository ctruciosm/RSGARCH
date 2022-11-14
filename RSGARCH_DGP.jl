#################################################
###  RSGARCH GPD: Simulate RSGARCH Models    ####
#################################################

using Random, Distributions, Plots, GARCH


omega = [0.05, 0.1]
alpha = [0.1, 0.05]
beta = [0.87, 0.7]
n = 1000
distri = Normal()
c = [0.4, 0.6]
d = [0,8, 0.3]
P = [0.9 0.1; 0.2 0.8];
time_varying = false;

function simulate_gray(n, omega, alpha, beta, burnin, time_varying, P, c, d)
   
   if (!time_varying & isnothing(P)) 
    @error "Transition matrix P should be provided"
   end
   if (time_varying & (isnothing(c) || isnothing(c)))
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
    Pt[1] = (1 - q) / (2 - p - q);
    h[1, k + 1] = Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2];
    r[1] = e[1] * sqrt(h[1, k + 1]);
    for i = 2:ntot
        h[i, 1:k] = omega .+ alpha.* r[i - 1]^2 + beta.* h[1, k + 1];
        numA = (1 - q) * pdf(Normal(0, sqrt(h[i - 1, 2])), r[i - 1]) * (1 - Pt[i - 1, 1]);
        numB = p * pdf(Normal(0, sqrt(h[i - 1, 1])), r[i - 1]) * Pt[i - 1, 1];
        deno = pdf(Normal(0, sqrt(h[i - 1, 1])), r[i - 1]) * Pt[i - 1, 1] + 
               pdf(Normal(0, sqrt(h[i - 1, 2])), r[i - 1]) * (1 - Pt[i - 1, 1]);
        Pt[i] = numA/deno + numB/deno;
        h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
        r[i] = e[i] * sqrt(h[i, k + 1]);
    end
   else
    p = cdf(Normal(), c[1] + d[1] * 0);
    q = cdf(Normal(), c[2] + d[2] * 0);
    Pt[1] = (1 - q) / (2 - p - q);
    h[1, k + 1] = Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2];
    r[1] = e[1] * sqrt(h[1, k + 1]);
    for i = 2:ntot
        p = cdf(Normal(), c[1] + d[1] * r[i - 1]);
        q = cdf(Normal(), c[2] + d[2] * r[i - 1]);
        h[i, 1:k] = omega .+ alpha.* r[i - 1]^2 + beta.* h[1, k + 1];
        numA = (1 - q) * pdf(Normal(0, sqrt(h[i - 1, 2])), r[i - 1]) * (1 - Pt[i - 1, 1]);
        numB = p * pdf(Normal(0, sqrt(h[i - 1, 1])), r[i - 1]) * Pt[i - 1, 1];
        deno = pdf(Normal(0, sqrt(h[i - 1, 1])), r[i - 1]) * Pt[i - 1, 1] + 
               pdf(Normal(0, sqrt(h[i - 1, 2])), r[i - 1]) * (1 - Pt[i - 1, 1]);
        Pt[i] = numA/deno + numB/deno;
        h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
        r[i] = e[i] * sqrt(h[i, k + 1]);
    end
   end
   #return r[burnin + 1:end];
   return r[burnin + 1:end], h[burnin + 1:end, ], Pt[burnin + 1: end, ];
end







# https://github.com/yonghanjung/RegimeSwitching_GARCH/blob/master/simulation/msGarchSim.m

P = [0.7 0.3; 0.4  0.6];

simulate_haas = function(n, omega, alpha, beta, P)
    burnin = 500;
    ntot = n + burnin;
    state = Vector{Int32}(undef, ntot);
    k = length(omega);
    e = rand(Normal(), ntot);
    [h[1, j] = omega[j] / (1 - alpha[j] - beta[j]) for j in 1:k];
    state[1] = 1;
    r[1] = e[1]* sqrt(h[1, state[1]]);
    for t = 2:ntot
        h[t, 1:k] = omega .+ alpha .* r[t - 1]^2 + beta .* h[t - 1, 1:k];
        state[t] = wsample([1, 2], P[state[t - 1], :], 1)[1];
        r[t] = e[t] * sqrt(h[t, state[t]]);
    end
    return r[burnin + 1:end];
end


teste_r, teste_h, teste_p = simulate_gray(1000, omega, alpha, beta, 500, true, nothing, c, d)
