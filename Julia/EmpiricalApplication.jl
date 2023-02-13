##################################################
###            RSGARCH: Forecasts             ####
##################################################
using Distributions, Optim, Statistics, StatsFuns, Random, SpecialFunctions, TryCatch, DelimitedFiles, StatsBase, QuadGK, StateSpaceModels, CSV, DataFrames
include("utils.jl")
include("DGP.jl")
include("MaximumLikelihood.jl")
include("Forecast.jl")

# Import Data
#prices = DataFrame(CSV.File("/home/prof/ctrucios/EUR_GBP_BRL_vs_USD.csv", header = 1, delim=","));
prices = DataFrame(CSV.File("/home/ctrucios/Dropbox/Research/RegimeSwitching-GARCH/RSGARCH/EUR_GBP_BRL_vs_USD.csv", header = 1, delim=","));
select!(prices, [:Date, :EUR_USD]);
rename!(prices, Symbol.(["Date","Price"]));
dropmissing!(prices)
prices.returns = [missing; 100*(log.(prices.Price[2:end]) -log.(prices.Price[1:end-1]))];
dropmissing!(prices);

# Settings
InS = 2500;
Tot = size(prices)[1];
OoS = Tot - InS + 1;
VaR_1 = Matrix{Float64}(undef, OoS, 4);
VaR_2 = Matrix{Float64}(undef, OoS, 4);
VaR_5 = Matrix{Float64}(undef, OoS, 4);
ES_1  = Matrix{Float64}(undef, OoS, 4);
ES_2  = Matrix{Float64}(undef, OoS, 4);
ES_5  = Matrix{Float64}(undef, OoS, 4);
r_oos  = Vector{Float64}(undef, OoS - 1);



# Out-of-sample VaR and ES
k = 2;
for i in 1:OoS - 1
    println(i)
    r = prices.returns[i:(InS + i - 1)];
    r_oos[i] = prices.returns[InS + i - 1 + 1];
    μ = mean(r);
    r = r .- μ;
    #p = sum(abs.(StatsBase.autocor(r, [1, 2])) .> 2/sqrt(InS));
    #if p > 0
    #    fit_ar(p)
    #    μ = μ + fore_ar(1)
    #end
    Random.seed!(1234 + i);
    θ = fit_gray(r, k, nothing, "norm");
    (h, Pt, s)  = fore_gray(r, k, θ, "norm");
    (VaR_1[i,1], ES_1[i,1]) = μ .+ var_es_rsgarch(0.01, Pt[end], 1 - Pt[end], sqrt(h[1]), sqrt(h[2]), "norm");
    (VaR_2[i,1], ES_2[i,1]) = μ .+ var_es_rsgarch(0.025, Pt[end], 1 - Pt[end], sqrt(h[1]), sqrt(h[2]), "norm");
    (VaR_5[i,1], ES_5[i,1]) = μ .+ var_es_rsgarch(0.05, Pt[end], 1 - Pt[end], sqrt(h[1]), sqrt(h[2]), "norm");

    θ = fit_gray(r, k, nothing, "student"); 
    (h, Pt, s)  = fore_gray(r, k, θ, "student");
    (VaR_1[i,2], ES_1[i,2]) = μ .+ var_es_rsgarch(0.01, Pt[end], 1 - Pt[end], sqrt(h[1]), sqrt(h[2]), "student", θ[9]);
    (VaR_2[i,2], ES_2[i,2]) = μ .+ var_es_rsgarch(0.025, Pt[end], 1 - Pt[end], sqrt(h[1]), sqrt(h[2]), "student", θ[9]);
    (VaR_5[i,2], ES_5[i,2]) = μ .+ var_es_rsgarch(0.05, Pt[end], 1 - Pt[end], sqrt(h[1]), sqrt(h[2]), "student", θ[9]);

    θ = fit_haas(r, k, nothing, "norm");
    (h, Pt, s)  = fore_haas(r, k, θ, "norm");
    (VaR_1[i,3], ES_1[i,3]) = μ .+ var_es_rsgarch(0.01, Pt[end], 1 - Pt[end], sqrt(h[1]), sqrt(h[2]), "norm");
    (VaR_2[i,3], ES_2[i,3]) = μ .+ var_es_rsgarch(0.025, Pt[end], 1 - Pt[end], sqrt(h[1]), sqrt(h[2]), "norm");
    (VaR_5[i,3], ES_5[i,3]) = μ .+ var_es_rsgarch(0.05, Pt[end], 1 - Pt[end], sqrt(h[1]), sqrt(h[2]), "norm");

    θ = fit_haas(r, k, nothing, "student");
    (h, Pt, s)  = fore_haas(r, k, θ, "student");
    (VaR_1[i,4], ES_1[i,4]) = μ .+ var_es_rsgarch(0.01, Pt[end], 1 - Pt[end], sqrt(h[1]), sqrt(h[2]), "student", θ[9]);
    (VaR_2[i,4], ES_2[i,4]) = μ .+ var_es_rsgarch(0.025, Pt[end], 1 - Pt[end], sqrt(h[1]), sqrt(h[2]), "student", θ[9]);
    (VaR_5[i,4], ES_5[i,4]) = μ .+ var_es_rsgarch(0.05, Pt[end], 1 - Pt[end], sqrt(h[1]), sqrt(h[2]), "student", θ[9]);
end

writedlm("VaR1_EUR_USD.csv",  VaR_1, ',')
writedlm("VaR2_EUR_USD.csv",  VaR_2, ',')
writedlm("VaR5_EUR_USD.csv",  VaR_5, ',')
writedlm("ES1_EUR_USD.csv",  ES_1, ',')
writedlm("ES2_EUR_USD.csv",  ES_2, ',')
writedlm("ES5_EUR_USD.csv",  ES_5, ',')
writedlm("r_oos_EUR_USD.csv",  r_oos, ',')


