##################################################
###  RSGARCH Estim: Estimate RSGARCH Models   ####
##################################################
using Distributions, Optim, ForwardDiff, StatsFuns, LinearAlgebra, Statistics, JuMP, BenchmarkTools


function gray_likelihood(par, r, k, distri)
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

    if (distri == "norm")
        for i = 2:n
            numA = (1 - q) * pdf(Normal(0, sqrt(h[i - 1, 2])), r[i - 1]) * (1 - Pt[i - 1]);
            numB = p * pdf(Normal(0, sqrt(h[i - 1, 1])), r[i - 1]) * Pt[i - 1];
            deno = pdf(Normal(0, sqrt(h[i - 1, 1])), r[i - 1]) * Pt[i - 1] + pdf(Normal(0, sqrt(h[i - 1, 2])), r[i - 1]) * (1 - Pt[i - 1]);
            Pt[i] = numA/deno + numB/deno;
    
            h[i, 1:k] = omega .+ alpha .* r[i - 1]^2 + beta .* h[i - 1, k + 1];
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
    
            log_lik[i - 1] = log(pdf(Normal(0, sqrt(h[i, 1])), r[i]) * Pt[i] + pdf(Normal(0, sqrt(h[i, 2])), r[i]) * (1 - Pt[i]));
        end
    else
        nu = par[4 * k + 1];
        for i = 2:n
            numA = (1 - q) * sqrt(nu/(nu - 2)) / sqrt(h[i - 1, 2]) * pdf(TDist(nu), r[i - 1] * sqrt(nu/(nu - 2)) / sqrt(h[i - 1, 2])) * (1 - Pt[i - 1]);
            numB = p * sqrt(nu/(nu - 2)) / sqrt(h[i - 1, 1]) * pdf(TDist(nu), r[i - 1] * sqrt(nu/(nu - 2)) / sqrt(h[i - 1, 1])) * Pt[i - 1];
            deno = sqrt(nu/(nu - 2)) / sqrt(h[i - 1, 1]) * pdf(TDist(nu), r[i - 1] * sqrt(nu/(nu - 2)) / sqrt(h[i - 1, 1])) * Pt[i - 1] + 
            sqrt(nu/(nu - 2)) / sqrt(h[i - 1, 2]) * pdf(TDist(nu), r[i - 1] * sqrt(nu/(nu - 2)) / sqrt(h[i - 1, 2])) * (1 - Pt[i - 1]);
            Pt[i] = numA/deno + numB/deno;
            h[i, 1:k] = omega .+ alpha .* r[i - 1]^2 + beta .* h[i - 1, k + 1];
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
            log_lik[i - 1] = log(sqrt(nu/(nu - 2)) / sqrt(h[i, 1]) * pdf(TDist(nu), r[i] * sqrt(nu/(nu - 2)) / sqrt(h[i, 1])) * Pt[i] + 
            sqrt(nu/(nu - 2)) / sqrt(h[i, 2]) * pdf(TDist(nu), r[i] * sqrt(nu/(nu - 2)) / sqrt(h[i, 2])) * (1 - Pt[i]));
        end
    end
    return -mean(log_lik);
end



function fit_gray(r, k, par_ini, distri)
    if distri == "norm"
        if isnothing(par_ini)
            par_ini = [0.05, 0.15, 0.3, 0.1, 0.6, 0.2, 0.85, 0.92];     
            ll = gray_likelihood(par_ini, r, k, distri);
            for i in 1:1000                                 
                omegas = rand(Uniform(0.01, 0.3), 2);
                alphas = rand(Uniform(0.05, 0.5), 2);
                betas = [rand(Uniform(0.4, max(0.41, 1 - alphas[1])), 1); rand(Uniform(0.4, max(0.41, 1 - alphas[2])), 1)];
                p = rand(Uniform(0.8, 0.99), 2);
                par_random = [omegas; alphas; betas; p];
                if gray_likelihood(par_random, r, k, distri) < ll
                    ll = gray_likelihood(par_random, r, k, distri);
                    par_ini = par_random;
                end
            end
        end
        optimum = optimize(par -> gray_likelihood(par, r, k, distri), [0, 0, 0, 0 ,0 ,0 ,0 ,0], [Inf, Inf, Inf, Inf, Inf, Inf, 1, 1], par_ini);
    else
        if isnothing(par_ini)
            par_ini = [0.05, 0.15, 0.3, 0.1, 0.6, 0.2, 0.85, 0.92, 5.0];     # arbitrary choice
            ll = gray_likelihood(par_ini, r, k, distri);
            for i in 1:1000                                 
                omegas = rand(Uniform(0.01, 0.3), 2);
                alphas = rand(Uniform(0.05, 0.5), 2);
                betas = [rand(Uniform(0.4, max(0.41, 1 - alphas[1])), 1); rand(Uniform(0.4, max(0.41, 1 - alphas[2])), 1)];
                p = rand(Uniform(0.8, 0.99), 2);
                nu = rand(Uniform(4.9, 35.0), 1);
                par_random = [omegas; alphas; betas; p; nu];
                if gray_likelihood(par_random, r, k, distri) < ll
                    ll = gray_likelihood(par_random, r, k, distri);
                    par_ini = par_random;
                end
            end
        end
        optimum = optimize(par -> gray_likelihood(par, r, k, distri), [0, 0, 0, 0 ,0 ,0 ,0 ,0, 4], [Inf, Inf, Inf, Inf, Inf, Inf, 1, 1, Inf], par_ini);
    end
    mle = optimum.minimizer;
    return mle;
end
