##################################################
###     How the functions should be used      ####
##################################################
using Distributions, Optim, Statistics, StatsFuns, Random, SpecialFunctions, TryCatch

include("utils.jl")
include("gray_dgp.jl")
include("gray_ml.jl")
#include("haas_ml.jl")
include("haas_dgp.jl")

# sem restricao nos teta1, teta2 e teta2
# omega = exp(-teta_0)
# alfa = exp(teta_1)/[1 + exp(teta_1) + exp(teta_2)] ,     0<alfa< 1
# beta = exp(teta_2)/[1 + exp(teta_1) + exp(teta_2)],     0<beta<1 e 
# alfa + beta = [exp(teta_1) + exp(teta_2)]/[1 + exp(teta_1) + exp(teta_2)]

#GARCH
n = 5000;
ω = 0.05;
α = 0.1;
β = 0.85;
burnin = 500
r = simulate_garch(n, ω, α, β, burnin);
r100 = 100*r;
θ̂ = fit_garch(r);
δ̂ = fit_garch(r100);

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
r100 = 100*r;
θ̂ = fit_haas(r, k, par_ini, distri);
δ̂ = fit_haas(r100, k, par_ini, distri);



# GRAY 1996
n = 5000;
ω = [0.1, 0.05];
α = [0.2, 0.1];
β = [0.7, 0.4];
P = [0.9 0.03; 0.1 0.97];
distri = "norm";
k = 2;
burnin = 500;
par_ini = nothing;
(r, h, Pt, s) = simulate_gray(n, distri, ω, α, β, P, burnin);
r100 = 100*r;
θ̂₃ = fit_gray_transform(r, k, par_ini, distri);
θ̂₄ = fit_gray_transform(r100, k, par_ini, distri);

theta = fit_gray(r, k, par_ini, distri);
theta2 = fit_gray(r100, k, par_ini, distri);

gray_likelihood(r, k, distri, θ̂₁)
gray_likelihood(r, k, distri, θ̂₂)


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
gray_likelihood(r, k, distri, θ̂₁)
gray_likelihood(r, k, distri, θ̂₂)

# GRAY 1996
using DelimitedFiles
dados_j, head_j = readdlm("/Volumes/CTRUCIOS_SD/UNICAMP/Ongoing Research/RegimeSwitchingGARCH/RSGARCH/retornos.csv", ',', header=true);

k = 2;
distri = "norm";
θ̂₁ = fit_gray(dados_j[:,2], k, nothing, distri); 
θ̂₂ = fit_gray_transform(dados_j[:,2], k, nothing, distri); 

gray_likelihood(dados_j[:,2], k, distri, mle)
gray_likelihood(dados_j[:,2], k, distri, θ̂₂)

params = [0.003283724, 0.023203443, 0.029516933, 0.228474116 , 0.722465951, 0.979609113, 0.226943732, 0.348192508]
# P,Q,omega.1,alpha.1,beta.1,omega.2,alpha.2,beta.2
#   0.029516933 0.722465951 0.228474116 0.979609113

gray_likelihood(dados_j[:,2], k, distri, params)




