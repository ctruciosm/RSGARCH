##################################################
###     How the functions should be used      ####
##################################################
using Distributions, Optim, Statistics, StatsFuns, Random, SpecialFunctions, TryCatch, DelimitedFiles

include("utils.jl")
include("DGP.jl")
include("MaximumLikelihood.jl")

#GARCH
n = 5000;
ω = 0.05;
α = 0.1;
β = 0.85;
burnin = 500
r = simulate_garch(n, ω, α, β, burnin);
θ̂ = fit_garch(r);
δ̂ = fit_garch(100*r);

# HAAS
n = 5000;
ω = [0.1, 0.05];
α = [0.2, 0.1];
β = [0.7, 0.4];
P = [0.9 0.03; 0.1 0.97];
distri = "student";
k = 2;
burnin = 500;
par_ini = nothing;
(r, h, s) = simulate_haas(n, distri, ω, α, β, P, burnin);
θ̂ = fit_haas(r, k, par_ini, distri);
δ̂ = fit_haas(100*r, k, par_ini, distri);



# GRAY 1996
n = 5000;
ω = [0.1, 0.05];
α = [0.2, 0.1];
β = [0.7, 0.4];
P = [0.9 0.03; 0.1 0.97];
distri = "student";
burnin = 500;
par_ini = nothing;
(r, h, Pt, s) = simulate_gray(n, distri, ω, α, β, P, burnin);
θ̂  = fit_gray(r, k, par_ini, distri);
δ̂  = fit_gray(100*r, k, par_ini, distri);
