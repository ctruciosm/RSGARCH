##################################################
###  RSGARCH Estim: Estimate RSGARCH Models   ####
##################################################
using Distributions, Optim, ForwardDiff, StatsFuns, LinearAlgebra, Statistics, JuMP, BenchmarkTools

function gray_likelihood(par, r, k)
    # par = numeric vector: omega, alpha, beta, p11, p22
    n = length(r);
    h = Array{Float64}(undef, n, k + 1);
    Pt = Vector{Float64}(undef, n);
    log_lik = Vector{Float64}(undef, n - 1);

    omega = par[1:k];
    alpha = par[k + 1 : 2 * k];
    beta = par[2 * k + 1 : 3 * k];
    p = par[3 * k + 1];
    q = par[4 * k];

    Pt[1] = (1 - q) / (2 - p - q);          ## Pi = P(St = 1) - Pag 683 Hamilton (1994)
    h[1, 1:k] .= var(r);                     ## See Fig 2 in Gray (1996)
    h[1, k + 1] = Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2];

    for i = 2:n
        numA = (1 - q) * pdf(Normal(0, sqrt(h[i - 1, 2])), r[i - 1]) * (1 - Pt[i - 1]);
        numB = p * pdf(Normal(0, sqrt(h[i - 1, 1])), r[i - 1]) * Pt[i - 1];
        deno = pdf(Normal(0, sqrt(h[i - 1, 1])), r[i - 1]) * Pt[i - 1] + pdf(Normal(0, sqrt(h[i - 1, 2])), r[i - 1]) * (1 - Pt[i - 1]);
        Pt[i] = numA/deno + numB/deno;

        h[i, 1:k] = omega .+ alpha .* r[i - 1]^2 + beta .* h[i - 1, k + 1];
        h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];

        log_lik[i - 1] = log(pdf(Normal(0, sqrt(h[i, 1])), r[i]) * Pt[i] + pdf(Normal(0, sqrt(h[i, 2])), r[i]) * (1 - Pt[i]));
    end
    return -mean(log_lik);
end

function fit_gray(r, k, par_ini)
    if isnothing(par_ini)
        par_ini = [0.05, 0.15, 0.3, 0.1, 0.6, 0.2, 0.85, 0.92];     # arbitrary choice
        ll = gray_likelihood(par_ini, r, k);
        for i in 1:1000                                 
        # GRID is hard becausa we have 6/7 parameters
            omegas = rand(Uniform(0.01, 0.3), 2);
            alphas = rand(Uniform(0.05, 0.5), 2);
            betas = [rand(Uniform(0.4, max(0.41, 1 - alphas[1])), 1); rand(Uniform(0.4, max(0.41, 1 - alphas[2])), 1)];
            p = rand(Uniform(0.8, 0.99), 2);
            par_random = [omegas; alphas; betas; p];
            if gray_likelihood(par_random, r, k) < ll
                ll = gray_likelihood(par_random, r, k);
                par_ini = par_random;
            end
        end
    end
    optimum = optimize(par -> gray_likelihood(par, r, k), [0, 0, 0, 0 ,0 ,0 ,0 ,0], [Inf, Inf, Inf, Inf, Inf, Inf, 1, 1], par_ini);
    mle = optimum.minimizer;
    return mle;
end

