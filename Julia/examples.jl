##################################################
###     How the functions should be used      ####
##################################################
using Distributions, Optim, Statistics, ForwardDiff, StatsFuns

include("gray_dgp.jl")
include("gray_ml.jl")
include("haas_ml.jl")
include("haas_dgp.jl")


# GRAY 1996
n = 2500;
ω = [0.18, 0.01];
α  = [0.4, 0.1];
β = [0.2, 0.7];
P = [0.9 0.03; 0.1 0.97];
distri = "norm";
C = 1;
D = 1;
k = 2;
burnin = 500;
Random.seed!(1234);
(r, h, Pt, s) = simulate_gray(n, distri, ω, α, β, P, burnin);
theta_hat1 = fit_gray(r, k, nothing, distri)
distri = "std";
Random.seed!(1234);
(r, h, Pt, s) = simulate_gray(n, distri, ω, α, β, P, burnin);
theta_hat2 = fit_gray(r, k, nothing, distri)


p = 0.9
q = 0.97
(1 - q) / (2 - p - q)




0.17750871374241112 / (1 - 0.2775795649005379 - 0.3221264886593553)
0.00632090448997519 / (1 - 0.1075898505938884 - 0.6962221388739883)


0.005146022712356059 / (1 - 0.06355701709526979 - 0.7551584609809686)
0.12537823584802654 / ( 1- 0.4171454190678328 - 0.2919786807969145)


# HAAS 2004
P = [0.7 0.1; 0.3  0.9];
n = 10000;
ω = [0.18, 0.01];
α  = [0.4, 0.2];
β = [0.3, 0.7];
burnin = 500;
distri = "norm";
Random.seed!(1234);
(r, h, Pt, s) = simulate_haas(n, distri, ω, α, β, P, burnin);

k = 2;
distri = "norm";
fitted_haas = fit_haas(r, k, nothing, distri);
fitted_haas










using DelimitedFiles
r = readdlm("returns.csv", ',', Float64);
distri = "std";
k = 2;
theta_hat1 = fit_gray(r, k, nothing, distri)

theta_hat2 = fit_gray2(r, k, nothing, distri)

gray_likelihood(theta_hat1, r, k, distri)

theta = [ω; α; β; 0.9; 0.97; 1/7];
gray_likelihood(theta, r, k, distri)






using DelimitedFiles
writedlm("msgarch_julia_norm.csv", r, ',');

par = [ω; α; β; 0.7; 0.9];
k = 2;
distri = "norm";
haas_likelihood(par, r, k, distri)


par_ini = par .+ rand(Uniform(0,0.05), 8);
haas_likelihood(par_ini, r, k, distri)


using DelimitedFiles
r = readdlm("/media/ctrucios/46CE33E1CE33C847/Carlos/Research/RSGARCH/msgarch_julia_norm.csv", ',', Float64);


fitted_haas

