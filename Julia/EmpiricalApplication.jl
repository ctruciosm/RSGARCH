##################################################
###            RSGARCH: Forecasts             ####
##################################################
using Distributions, Optim, Statistics, StatsFuns, Random, SpecialFunctions, TryCatch, DelimitedFiles, StatsBase, QuadGK, StateSpaceModels
include("utils.jl")
include("DGP.jl")
include("MaximumLikelihood.jl")
include("Forecast.jl")

# Import Data
prices = DataFrame(CSV.File("/home/ctrucios/Dropbox/Research/RegimeSwitching-GARCH/RSGARCH/EUR_GBP_BRL_vs_USD.csv", header = 1, delim=","));
select!(prices, [:Date, :EUR_USD]);
rename!(prices, Symbol.(["Date","Price"]));
subset!(prices, :Date  => x -> x .< Date(2023-01-01));
dropmissing!(prices)
prices.returns = [missing; 100*(log.(prices.Price[2:end]) -log.(prices.Price[1:end-1]))];
dropmissing!(prices)

# Settings
InS = 2500;
Tot = size(prices)[1];
OoS = Tot - InS + 1;
VaR_1 = Matrix{Float64}(undef, OoS, 4);
VaR_2 = Matrix{Float64}(undef, OoS, 4);
ES_1  = Matrix{Float64}(undef, OoS, 4);
ES_2  = Matrix{Float64}(undef, OoS, 4);

# Out-of-sample VaR and ES
k = 2;
for i in 1:OoS 
    println(i)
    r = prices.returns[i:(InS + i - 1)];
    μ = mean(r);
    r = r .- μ;

    θ = fit_gray(r, k, nothing, "norm");
    (h, Pt)  = fore_gray(r, k, θ, "norm");
    (VaR_1(i,1), ES_1(i,1)) = μ .+ var_es_rsgarch(0.01, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "norm");
    (VaR_2(i,1), ES_2(i,1)) = μ .+ var_es_rsgarch(0.025, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "norm");

    θ = fit_gray(r, k, nothing, "student"); 
    (h, Pt)  = fore_gray(r, k, θ, "student");
    (VaR_1(i,2), ES_1(i,2)) = μ .+ var_es_rsgarch(0.01, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "student");
    (VaR_2(i,2), ES_2(i,2)) = μ .+ var_es_rsgarch(0.025, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "student");

    θ = fit_haas(r, k, nothing, "norm");
    (h, Pt)  = fore_haas(r, k, θ, "normal");
    (VaR_1(i,3), ES_1(i,3)) = μ .+ var_es_rsgarch(0.01, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "student");
    (VaR_2(i,3), ES_2(i,3)) = μ .+ var_es_rsgarch(0.025, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "student");

    θ = fit_haas(r, k, nothing, "student");
    (h, Pt)  = fore_haas(r, k, θ, "studentl");
    (VaR_1(i,4), ES_1(i,4)) = μ .+ var_es_rsgarch(0.01, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "student");
    (VaR_2(i,4), ES_2(i,4)) = μ .+ var_es_rsgarch(0.025, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "student");
end

writedlm("VaR1_EUR_USD.csv",  VaR_1, ',')
writedlm("VaR2_EUR_USD.csv",  VaR_2, ',')
writedlm("ES1_EUR_USD.csv",  ES_1, ',')
writedlm("ES2_EUR_USD.csv",  ES_2, ',')
