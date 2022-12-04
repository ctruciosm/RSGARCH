##################################################
###     How the functions should be used      ####
##################################################
using Distributions, Optim, ForwardDiff, StatsFuns, LinearAlgebra, Statistics, JuMP, BenchmarkTools, Plots

include("gray_dgp.jl")
include("gray_ml.jl")

# GRAY 1996

n = 5000;
omega = [0.18, 0.01];
alpha  = [0.4, 0.1];
beta = [0.2, 0.7];
P = [0.9 0.03; 0.1 0.97];
time_varying = false;
distri = "norm";
C = 1;
D = 1;
k = 2;
burnin = 500;
(r, h, Pt, s) = simulate_gray(n, distri, omega, alpha, beta, time_varying, P, C, D, burnin);
par_ini = [0.1, 0.01, 0.3, 0.1, 0.5, 0.3, 0.8, 0.95];
theta_hat = fit_gray(r, k, par_ini, distri)
theta_hat2 = fit_gray(r, k, nothing,distri)


distri = "std";
(r, h, Pt, s) = simulate_gray(n, distri, omega, alpha, beta, time_varying, P, C, D, burnin);

par_ini = [0.1, 0.01, 0.3, 0.1, 0.5, 0.3, 0.8, 0.95, 5];
theta_hat3 = fit_gray(r, k, par_ini, distri)
theta_hat5 = fit_gray(r, k, nothing, distri)







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