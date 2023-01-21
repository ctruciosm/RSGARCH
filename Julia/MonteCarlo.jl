####################################################
###           Monte Carlo Simulation             ###
####################################################
# Regime 1: Low Vol
# Regime 2: High Vol
####################################################
using Distributions, Optim, Statistics, StatsFuns, Random, SpecialFunctions, TryCatch, DelimitedFiles

include("utils.jl")
include("DGP.jl")
include("MaximumLikelihood.jl")


function MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin) 
    if distri == "norm"
        params = Matrix{Float64}(undef, MC, 14);
        for i = 1:MC
            println(i)
            Random.seed!(1234 + i);
            (r, h, Pt, s) = simulate_gray(n, distri, ω, α, β, P, burnin);
            θ̂ = fit_gray(r, k, nothing, distri);
            σ̂₁ = θ̂[1] / (1 - θ̂[3] - θ̂[5]);
            σ̂₂ = θ̂[2] / (1 - θ̂[4] - θ̂[6]);
            ĥ  = fore_gray(r, k, θ̂, distri);
            if σ̂₁ < σ̂₂
                params[i, :] = [θ̂; θ̂[3] + θ̂[5]; θ̂[4] + θ̂[6]; σ̂₁; σ̂₂; ĥ[3]; h[end, 3]];
            else
                params[i, :] = [θ̂[[2; 1; 4; 3; 6; 5; 8; 7]]; θ̂[4] + θ̂[6]; θ̂[3] + θ̂[5]; σ̂₂; σ̂₁; ĥ[3]; h[end, 3]];
            end
        end
    elseif(distri == "student")
        params = Matrix{Float64}(undef, MC, 15);
        for i = 1:MC
            println(i)
            Random.seed!(1234 + i);
            (r, h, Pt, s) = simulate_gray(n, distri, ω, α, β, P, burnin);
            θ̂ = fit_gray(r, k, nothing, distri);
            σ̂₁ = θ̂[1] / (1 - θ̂[3] - θ̂[5]);
            σ̂₂ = θ̂[2] / (1 - θ̂[4] - θ̂[6]);
            ĥ = fore_gray(r, k, θ̂, distri);
            if σ̂₁ < σ̂₂
                params[i, :] = [θ̂; θ̂[3] + θ̂[5]; θ̂[4] + θ̂[6]; σ̂₁; σ̂₂; ĥ[3]; h[end, 3]];
            else
                params[i, :] = [θ̂[[2; 1; 4; 3; 6; 5; 8; 7; 9]]; θ̂[4] + θ̂[6]; θ̂[3] + θ̂[5]; σ̂₂; σ̂₁; ĥ[3]; h[end, 3]];
            end
        end
    else
        println("Only Normal ('norm') and Student-T ('student') distributions are available")
    end
    return params;
end

# Regime 1: Low Vol
# Regime 2: High Vol
MC = 500
ω = [0.18, 0.01];
α = [0.46, 0.16];
β = [0.20, 0.30];
P = [0.95 0.02; 0.05 0.98];
k = 2;
burnin = 500;
#########################################################################################
distri = "norm";
n = 5000;
params_5000_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_5000_n.csv",  params_5000_n, ',')
n = 2500;
params_2500_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_2500_n.csv",  params_2500_n, ',')
n = 1000;
params_1000_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_1000_n.csv",  params_1000_n, ',')

#########################################################################################
distri = "student";
n = 5000;
params_5000_t = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_5000_t.csv",  params_5000_t, ',')
n = 2500;
params_2500_t = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_2500_t.csv",  params_2500_t, ',')
n = 1000;
params_1000_t = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_1000_t.csv",  params_1000_t, ',')
