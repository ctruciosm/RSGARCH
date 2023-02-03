##################################################
###            RSGARCH: Forecasts             ####
##################################################
using Distributions, Optim, Statistics, StatsFuns, Random, SpecialFunctions, TryCatch, DelimitedFiles, StatsBase, QuadGK

include("utils.jl")
include("DGP.jl")
include("MaximumLikelihood.jl")
include("Forecast.jl")


prices = readdlm("/home/ctrucios/Dropbox/Research/RegimeSwitching-GARCH/RSGARCH/EUR_BRL_CHF_GBP_YEN_vs_USD.csv");
returns = 100 * log(prices / lag(prices));

InS = 2500;
Tot = size(prices)[1];
OoS = Tot - InS + 1;

VaR_1 = Matrix{Float64}(undef, OoS, 8);
VaR_5 = Matrix{Float64}(undef, OoS, 8);
ES_1  = Matrix{Float64}(undef, OoS, 8);
ES_5  = Matrix{Float64}(undef, OoS, 8);

k = 2;
for i in 1:OoS 
    println(i)
    r = returns[i:(InS + i - 1)];
    μ = mean(r);
    r = r - μ;

    θ = fit_gray(r, k, nothing, "norm");
    (h, Pt)  = fore_gray(r, k, θ, "norm");
    (VaR_1(i,1), ES_1(i,1)) = μ .+ var_es_rsgarch(0.01, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "norm");
    (VaR_5(i,1), ES_5(i,1)) = μ .+ var_es_rsgarch(0.05, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "norm");

    θ = fit_gray(r, k, nothing, "student"); 
    (h, Pt)  = fore_gray(r, k, θ, "student");
    (VaR_1(i,2), ES_1(i,2)) = μ .+ var_es_rsgarch(0.01, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "student");
    (VaR_5(i,2), ES_5(i,2)) = μ .+ var_es_rsgarch(0.05, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "student");

    θ = fit_haas(r, k, nothing, "norm");
    (h, Pt)  = fore_haas(r, k, θ, "normal");
    (VaR_1(i,3), ES_1(i,3)) = μ .+ var_es_rsgarch(0.01, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "student");
    (VaR_5(i,3), ES_5(i,3)) = μ .+ var_es_rsgarch(0.05, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "student");

    θ = fit_haas(r, k, nothing, "student");
    (h, Pt)  = fore_haas(r, k, θ, "studentl");
    (VaR_1(i,4), ES_1(i,4)) = μ .+ var_es_rsgarch(0.01, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "student");
    (VaR_5(i,4), ES_5(i,4)) = μ .+ var_es_rsgarch(0.05, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "student");
end