####################################################
###           Monte Carlo Simulation             ###
####################################################
using Distributions, Optim, Statistics, ForwardDiff, StatsFuns, NLSolversBase

include("gray_dgp.jl")
include("gray_ml.jl")


MC = 10;
n = 5000;
ω = [0.18, 0.01];
α  = [0.4, 0.1];
β = [0.2, 0.7];
P = [0.9 0.03; 0.1 0.97];
σ₁ = ω[1] / (1 - α[1] - β[1]);
σ₂ = ω[2] / (1 - α[2] - β[2]);  
par = [ω; α; β; 0.9; 0.97];
distri = "norm";
k = 2;
burnin = 500;
params = Matrix{Float64}(undef, MC, 14);
for i = 1:MC
    print(i)
    Random.seed!(1234);
    (r, h, Pt, s) = simulate_gray(n + 1, distri, ω, α, β, P, burnin);
    θ̂ = fit_gray(r[1:end-1], k, nothing, distri);
    σ̂₁ = θ̂[1] / (1 - θ̂[3] - θ̂[5]);
    σ̂₂ = θ̂[2] / (1 - θ̂[4] - θ̂[6]);
    if abs(σ₁ - σ̂₁) < abs(σ₂ - σ̂₁)
        params[i, :] = [θ̂; θ̂[3] + θ̂[5]; θ̂[4] + θ̂[6]; σ̂₁; σ̂₂; s[end]; h[end, s[end]]];
    else
        params[i, :] = [θ̂[[2; 1; 4; 3; 6; 5; 8; 7]]; θ̂[4] + θ̂[6]; θ̂[3] + θ̂[5]; σ̂₂; σ̂₁; 2 - s[end] + 1; h[end, s[end]]];
    end
end
params

