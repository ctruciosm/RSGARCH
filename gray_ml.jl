##################################################
###  RSGARCH Estim: Estimate RSGARCH Models   ####
##################################################
using Distributions, Optim, ForwardDiff, StatsFuns, LinearAlgebra, Statistics, JuMP, BenchmarkTools
include("gray_dgp.jl")

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
        # GRID
    end
    # Constraints
    optimum = optimize(par -> gray_likelihood(par, r, k), [0, 0, 0, 0 ,0 ,0 ,0 ,0], [Inf, Inf, Inf, Inf, Inf, Inf, 1, 1], par_ini);
    mle = optimum.minimizer;
    return mle;
end


n = 5000;
omega = [0.18, 0.01];
alpha  = [0.4, 0.1];
beta = [0.2, 0.7];
P = [0.9 0.03; 0.1 0.97];
time_varying = false;
C = 1;
D = 1;
k = 2;
burnin = 500;
(r, h, Pt, s) = simulate_gray(n, omega, alpha, beta, time_varying, P, C, D, burnin);





par_ini = [omega + rand(Uniform(0, 0.03), 2); alpha + rand(Uniform(0, 0.03), 2); beta + rand(Uniform(0, 0.03), 2); 0.8; 0.96];


mle_estim = fit_gray(r, k, par_ini)


make_closures(r, k) = par -> gray_likelihood(par, r, k)
gray_ll = make_closures(r, k)
gray_ll(par_ini)



model = Model();
@variable(model, par[1:8] .>= 0.0);
register(model, :gray_ll, 1, gray_ll; autodiff = true);
@constraint(model, par[3] + par[5] <= 0.999999);
@constraint(model, par[4] + par[6] <= 0.999999);
@constraint(model, par[7:8] .<= 1);
print(model)
@objective(model, Min, :gray_ll, par_ini)

res = optimize(gray_ll, par_ini, LBFGS(), autodiff=:forward)
res.minimizer