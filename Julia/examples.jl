##################################################
###     How the functions should be used      ####
##################################################
using Distributions, Optim, ForwardDiff, StatsFuns, LinearAlgebra, Statistics, JuMP, BenchmarkTools, Plots

include("gray_dgp.jl")
include("gray_ml.jl")

# GRAY 1996

n = 5000;
ω = [0.18, 0.01];
α  = [0.4, 0.1];
β = [0.3, 0.7];
P = [0.9 0.03; 0.1 0.97];
time_varying = false;
distri = "std";
C = 1;
D = 1;
k = 2;
burnin = 500;
(r, h, Pt, s) = simulate_gray(n, distri, ω, α, β, time_varying, P, C, D, burnin);
theta_hat1 = fit_gray(r, k, nothing, distri)



using DelimitedFiles
r = readdlm("returns.csv", ',', Float64);
distri = "std";
k = 2;
theta_hat1 = fit_gray(r, k, nothing, distri)

theta_hat2 = fit_gray2(r, k, nothing, distri)

gray_likelihood(theta_hat1, r, k, distri)

theta = [ω; α; β; 0.9; 0.97; 1/7];
gray_likelihood(theta, r, k, distri)
