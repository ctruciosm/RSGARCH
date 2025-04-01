####################################################
###                Simulations                   ###
####################################################
# Regime 1: Low Vol
# Regime 2: High Vol
####################################################

using Distributions, Optim, Statistics, StatsFuns, Random, SpecialFunctions, TryCatch, DelimitedFiles, StatsBase, QuadGK, Kronecker

include("./utils.jl")
include("./DGP.jl")
include("./MaximumLikelihood.jl")
include("./Forecast.jl")
include("./MonteCarlo.jl")

#########################################################################################
### Global Setup
#########################################################################################
MC = 1;
k = 2;
burnin = 500;
### Specific Setup
dgp = ARGS[1];
n = parse.(Int, ARGS[2]);


if (dgp == "DGP1")
    ω = [0.01, 0.18];
    α = [0.16, 0.46];
    β = [0.30, 0.20];
    P = [0.98 0.05; 0.02 0.95];
end
# Parameters Hass 2004 - Table 3
if (dgp == "DGP2")
    ω = [0.005, 0.1];
    α = [0.025, 0.25];
    β = [0.95, 0.70];
    P = [0.75 0.30; 0.25 0.70];
end
# Parameter Marcucci 2005 - Table 3
if (dgp == "DGP3")
    ω = [0.003, 0.09];
    α = [0.015, 0.07];
    β = [0.98, 0.85];
    P = [0.99 0.01; 0.01 0.99];
end
# Hotta, Trucios, Valls, Zevallos
#ω = [0.01, 0.2];  #0.05 2
#α = [0.05, 0.15];
#β = [0.9, 0.65];
#P = [0.95 0.05; 0.10 0.9];  

#########################################################################################
### Normal
#########################################################################################
distri = "norm";
parameters_gray_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_gray, n - 1);
writedlm(string("parameters_", n, "_gray_n_", dgp, "_out_", n - 1, ".csv"),  parameters_gray_n, ',')
parameters_klaassen_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_klaassen, n - 1);
writedlm(string("parameters_", n, "_klaassen_n_", dgp, "_out_", n - 1, ".csv"),  parameters_klaassen_n, ',')
parameters_haas_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_haas, n - 1);
writedlm(string("parameters_", n, "_haas_n_", dgp, "_out_", n - 1, ".csv"),  parameters_haas_n, ',');

parameters_gray_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_gray, n - 3);
writedlm(string("parameters_", n, "_gray_n_", dgp, "_out_", n - 3, ".csv"),  parameters_gray_n, ',')
parameters_klaassen_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_klaassen, n - 3);
writedlm(string("parameters_", n, "_klaassen_n_", dgp, "_out_", n - 3, ".csv"),  parameters_klaassen_n, ',')
parameters_haas_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_haas, n - 3);
writedlm(string("parameters_", n, "_haas_n_", dgp, "_out_", n - 3, ".csv"),  parameters_haas_n, ',');

parameters_gray_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_gray, n - 5);
writedlm(string("parameters_", n, "_gray_n_", dgp, "_out_", n - 5, ".csv"),  parameters_gray_n, ',')
parameters_klaassen_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_klaassen, n - 5);
writedlm(string("parameters_", n, "_klaassen_n_", dgp, "_out_", n - 5, ".csv"),  parameters_klaassen_n, ',')
parameters_haas_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_haas, n - 5);
writedlm(string("parameters_", n, "_haas_n_", dgp, "_out_", n - 5, ".csv"),  parameters_haas_n, ',');

distri = "student";
parameters_gray_t = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_gray, n - 1);
writedlm(string("parameters_", n, "_gray_t_", dgp, "_out_", n - 1, ".csv"),  parameters_gray_t, ',')
parameters_klaassen_t = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_klaassen, n - 1);
writedlm(string("parameters_", n, "_klaassen_t_", dgp, "_out_", n - 1, ".csv"),  parameters_klaassen_t, ',')
parameters_haas_t = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_haas, n - 1);
writedlm(string("parameters_", n, "_haas_t_", dgp, "_out_", n - 1, ".csv"),  parameters_haas_t, ',')

parameters_gray_t = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_gray, n - 3);
writedlm(string("parameters_", n, "_gray_t_", dgp, "_out_", n - 3, ".csv"),  parameters_gray_t, ',')
parameters_klaassen_t = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_klaassen, n - 3);
writedlm(string("parameters_", n, "_klaassen_t_", dgp, "_out_", n - 3, ".csv"),  parameters_klaassen_t, ',')
parameters_haas_t = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_haas, n - 3);
writedlm(string("parameters_", n, "_haas_t_", dgp, "_out_", n - 3, ".csv"),  parameters_haas_t, ',')

parameters_gray_t = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_gray, n - 5);
writedlm(string("parameters_", n, "_gray_t_", dgp, "_out_", n - 5, ".csv"),  parameters_gray_t, ',')
parameters_klaassen_t = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_klaassen, n - 5);
writedlm(string("parameters_", n, "_klaassen_t_", dgp, "_out_", n - 5, ".csv"),  parameters_klaassen_t, ',')
parameters_haas_t = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_haas, n - 5);
writedlm(string("parameters_", n, "_haas_t_", dgp, "_out_", n - 5, ".csv"),  parameters_haas_t, ',')


