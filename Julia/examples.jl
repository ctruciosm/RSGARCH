##################################################
###     How the functions should be used      ####
##################################################
using Distributions, Optim, Statistics, StatsFuns, Random, SpecialFunctions

include("utils.jl")
include("gray_dgp.jl")
include("gray_ml.jl")
#include("haas_ml.jl")
#include("haas_dgp.jl")


# GRAY 1996
n = 2500;
ω = [0.1, 0.05];
α = [0.2, 0.1];
β = [0.7, 0.4];
P = [0.9 0.03; 0.1 0.97];
distri = "norm";
k = 2;
burnin = 500;
Random.seed!(1234);
(r, h, Pt, s) = simulate_gray(n, distri, ω, α, β, P, burnin);
θ̂₁ = fit_gray(r, k, nothing, distri);
θ̂₂ = fit_gray_transform(r, k, nothing, distri);
θ̂₁
θ̂₂


# GRAY 1996
n = 10000;
ω = [0.001, 0.05];
α  = [0.1, 0.2];
β = [0.8, 0.7];
P = [0.99 0.03; 0.01 0.97];
distri = "norm";
k = 2;
burnin = 500;
Random.seed!(1234);
(r, h, Pt, s) = simulate_gray(n, distri, ω, α, β, P, burnin);
θ̂₁ = fit_gray(r, k, nothing, distri);
θ̂₂ = fit_gray_transform(r, k, nothing, distri);
θ̂₁
θ̂₂

# GRAY 1996
using DelimitedFiles
dados_j, head_j = readdlm("/Volumes/CTRUCIOS_SD/UNICAMP/Ongoing Research/RegimeSwitchingGARCH/RSGARCH/retornos.csv", ',', header=true);

k = 2;
distri = "norm";
θ̂₁ = fit_gray(dados_j[:,2], k, nothing, distri); 
θ̂₂ = fit_gray_transform(dados_j[:,2], k, nothing, distri); 



# P,Q,omega.1,alpha.1,beta.1,omega.2,alpha.2,beta.2
# 0.226943732 0.348192508 0.003283724 0.029516933 0.722465951 0.023203443 0.228474116 0.979609113



