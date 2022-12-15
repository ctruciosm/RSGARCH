# TESTES
using Distributions, Optim, Statistics, ForwardDiff, StatsFuns, NLSolversBase, JuMP, Ipopt
include("gray_dgp.jl")

n = 5000;
ω = [0.18, 0.01];
α  = [0.4, 0.1];
β = [0.2, 0.7];
P = [0.9 0.03; 0.1 0.97];
σ₁ = ω[1] / (1 - α[1] - β[1]);
σ₂ = ω[2] / (1 - α[2] - β[2]);  
par = [ω; α; β; 0.9; 0.97];
distri = "norm";
k = 2;
burnin = 500;
(r, h, Pt, s) = simulate_gray(n + 1, distri, ω, α, β, P, burnin);


function log_likelihood(par...)
    # par(numeric vector): [ω, α, β, p11, p22, 1/gl]
    n = length(r);
    h = Matrix{Float64}(undef, n, k + 1);
    Pt = Vector{Float64}(undef, n);
    log_lik = Vector{Float64}(undef, n - 1);

    ω = par[1:k];
    α = par[k + 1 : 2 * k];
    β = par[2 * k + 1 : 3 * k];
    p = par[3 * k + 1];
    q = par[4 * k];

    Pt[1] = (1 - q) / (2 - p - q);              ## Pi = P(St = 1) - Pag 683 Hamilton (1994)
    h[1, 1:k] .= var(r[1:30]);                        
    h[1, k + 1] = Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2];
    for i = 2:n
        h[i, 1] = ω[1] + α[1]* r[i - 1]^2 + β[1]* h[i - 1, k + 1];
        h[i, 2] = ω[2] + α[2]* r[i - 1]^2 + β[2]* h[i - 1, k + 1];
        Pt[i] = probability_regime_given_time_n(p, q, sqrt.(h[i - 1, :]), r[i - 1], Pt[i - 1]);
        h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
        log_lik[i - 1] = log(pdf(Normal(0, sqrt(h[i, 1])), r[i]) * Pt[i] + pdf(Normal(0, sqrt(h[i, 2])), r[i]) * (1 - Pt[i]));
    end
    return -sum(log_lik)/2;
end


par_ini = [0.05, 0.15, 0.3, 0.1, 0.6, 0.2, 0.85, 0.92]; 




log_likelihood(r, k, par_ini)


make_closures(r, k) = par -> log_likelihood(r, k, par)
nll = make_closures(r, k)
nll(par_ini)

optimize(nll, par_ini, LBFGS(), autodiff=:forward)


model = Model();
@variable(model, par[1:8] .>= 1e-6);
register(model, :nll, 8, nll; autodiff=:auto)

@variable(model, par[1:8] .>= 1e-6);
@constraint(model, par[3] + par[5] <= 0.9999999);
@constraint(model, par[4] + par[6] <= 0.9999999);
@constraint(model, par[7:8] .<= 0.9999999);
print(model)



