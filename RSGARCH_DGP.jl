################################
###      RSGARCH MODELS     ####
################################
using Random, Distributions, Plots, GARCH

simulate_garch = function (n, omega, alpha, beta, distri)
    burnin = 500;
    ntot = n + burnin;
    e = rand(distri, ntot);
    h = zeros(ntot);
    r = zeros(ntot);
    h[1] = omega/(1 - alpha - beta)
    r[1] = e[1]*sqrt(h[1])
    for i = 2:ntot
        h[i] = omega + alpha * r[i - 1]^2 + beta * h[i - 1];
        r[i] = e[i] * sqrt(h[i]);
    end
    return r[burnin + 1:end];
end

teste = simulate_garch(1000, 0.01, 0.05, 0.87, Normal());

plot(teste)

omega = [0.05, 0.1]
alpha = [0.1, 0.05]
beta = [0.87, 0.7]
n = 1000
distri = Normal()
c = [0.4, 0.6]
d = [0,8, 0.3]

simulate_gray = function(n, omega, alpha, beta, c, d)
   burnin = 500;
   ntot = n + burnin;
   k = length(omega);
   e = rand(Normal(), ntot);
   P = zeros(ntot);
   Q = zeros(ntot);
   Pt = zeros(ntot, k);
   h = zeros(ntot, k + 1);
   r = zeros(ntot);

   P[1] = cdf(Normal(), c[1] + d[1] * 0);  # r_0 = E(r_0) = 0
   Q[1] = cdf(Normal(), c[2] + d[2] * 0);  # r_0 = E(r_0) = 0
   [h[1, j] = omega[j] / (1 - alpha[j] - beta[j]) for j in 1:k];

   Pt[1, 1] = 0.5;                         # I'm not sure.
   h[1, k + 1] = Pt[1, 1] * h[1, 1] + (1 - Pt[1, 1]) * h[1, 2];
   r[1] = e[1]*sqrt(h[1, k + 1]);

   for t = 2:ntot
    P[t] = cdf(Normal(), c[1] + d[1] * r[t- 1]);
    Q[t] = cdf(Normal(), c[2] + d[2] * r[t - 1]);
    [h[t, j] = omega[j] + alpha[j]*r[t - 1]^2 + beta[j] * h[t - 1, k + 1] for j in 1:k];
    
    num_a = P[t] * pdf(Normal(0, h[t - 1, 1]), r[t - 1]) * Pt[t - 1, 1];
    num_b = (1 - Q[t]) * pdf(Normal(0, h[t - 1, 2]), r[t - 1]) * (1 - Pt[t - 1, 1]);
    den = pdf(Normal(0, h[t - 1, 1]), r[t - 1]) * Pt[t - 1, 1] + 
          pdf(Normal(0, h[t - 1, 2]), r[t - 1]) * (1 - Pt[t - 1, 1]);
    Pt[t, 1] = num_a/den + num_b/den;

    h[t, k + 1] = Pt[t, 1] * h[t, 1] + (1 - Pt[t, 1]) * h[t, 2];
    r[t] = e[t]*sqrt(h[t, k + 1]);
   end
   return r[burnin + 1:end];
end

teste = simulate_gray(1000, omega, alpha, beta, c, d);
   


plot(teste)

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


teste = simulate_haas(1000, omega, alpha, beta, P)