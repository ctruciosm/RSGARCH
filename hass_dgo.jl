




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
