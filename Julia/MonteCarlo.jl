####################################################
###           Monte Carlo Simulation             ###
####################################################
# Regime 1: Low Vol
# Regime 2: High Vol
####################################################
using Distributions, Optim, Statistics, ForwardDiff, StatsFuns, Random, DelimitedFiles, JuMP, Ipopt, SpecialFunctions

include("utils.jl")
include("gray_dgp.jl")
include("gray_ml.jl")



function MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin) 
    if distri == "norm"
        params = Matrix{Float64}(undef, MC, 12);
        for i = 1:MC
            println(i)
            Random.seed!(1234 + i);
            (r, h, Pt, s) = simulate_gray(n, distri, ω, α, β, P, burnin);
            θ̂ = fit_gray(r, k, nothing, distri);
            σ̂₁ = θ̂[1] / (1 - θ̂[3] - θ̂[5]);
            σ̂₂ = θ̂[2] / (1 - θ̂[4] - θ̂[6]);
            if σ̂₁ < σ̂₂
                params[i, :] = [θ̂; θ̂[3] + θ̂[5]; θ̂[4] + θ̂[6]; σ̂₁; σ̂₂];
            else
                params[i, :] = [θ̂[[2; 1; 4; 3; 6; 5; 8; 7]]; θ̂[4] + θ̂[6]; θ̂[3] + θ̂[5]; σ̂₂; σ̂₁];
            end
        end
    elseif(distri == "std")
        params = Matrix{Float64}(undef, MC, 13);
        for i = 1:MC
            println(i)
            Random.seed!(1234 + i);
            (r, h, Pt, s) = simulate_gray(n, distri, ω, α, β, P, burnin);
            θ̂ = fit_gray(r, k, nothing, distri);
            σ̂₁ = θ̂[1] / (1 - θ̂[3] - θ̂[5]);
            σ̂₂ = θ̂[2] / (1 - θ̂[4] - θ̂[6]);
            if σ̂₁ < σ̂₂
                params[i, :] = [θ̂; θ̂[3] + θ̂[5]; θ̂[4] + θ̂[6]; σ̂₁; σ̂₂];
            else
                params[i, :] = [θ̂[[2; 1; 4; 3; 6; 5; 8; 7; 9]]; θ̂[4] + θ̂[6]; θ̂[3] + θ̂[5]; σ̂₂; σ̂₁];
            end
        end
    else
        params = Matrix{Float64}(undef, MC, 13);
        for i = 1:MC
            println(i)
            Random.seed!(1234 + i);
            (r, h, Pt, s) = simulate_gray(n, distri, ω, α, β, P, burnin);
            θ̂ = fit_gray(r, k, nothing, distri);
            θ̂[9] = 1/θ̂[9];
            σ̂₁ = θ̂[1] / (1 - θ̂[3] - θ̂[5]);
            σ̂₂ = θ̂[2] / (1 - θ̂[4] - θ̂[6]);
            if σ̂₁ < σ̂₂
                params[i, :] = [θ̂; θ̂[3] + θ̂[5]; θ̂[4] + θ̂[6]; σ̂₁; σ̂₂];
            else
                params[i, :] = [θ̂[[2; 1; 4; 3; 6; 5; 8; 7; 9]]; θ̂[4] + θ̂[6]; θ̂[3] + θ̂[5]; σ̂₂; σ̂₁];
            end
        end
    end
    return params;
end


MC = 5
ω = [0.18, 0.01];
α = [0.4, 0.1];
β = [0.2, 0.7];
P = [0.9 0.03; 0.1 0.97];
k = 2;
burnin = 500;

#########################################################################################
n = 5000;
distri = "norm";
params_5000_n = round.(MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin), digits = 6);
writedlm("params_5000_n.csv",  params_5000_n, ',')
distri = "std";
params_5000_t = round.(MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin), digits = 6);
writedlm("params_5000_t.csv",  params_5000_t, ',')
distri = "istd";
params_5000_it = round.(MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin), digits = 6);
writedlm("params_5000_it.csv",  params_5000_it, ',')

#########################################################################################
n = 2500;
distri = "norm";
params_2500_n = round.(MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin), digits = 6);
writedlm("params_2500_n.csv",  params_2500_n, ',')
distri = "std";
params_2500_t = round.(MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin), digits = 6);
writedlm("params_2500_t.csv",  params_2500_t, ',')
distri = "istd";
params_2500_it = round.(MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin), digits = 6);
writedlm("params_2500_it.csv",  params_2500_it, ',')

#########################################################################################
n = 1000;
distri = "norm";
params_1000_n = round.(MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin), digits = 6);
writedlm("params_1000_n.csv",  params_1000_n, ',')
distri = "std";
params_1000_t = round.(MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin), digits = 6);
writedlm("params_1000_t.csv",  params_1000_t, ',')
distri = "istd";
params_2500_it = round.(MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin), digits = 6);
writedlm("params_1000_it.csv",  params_1000_it, ',')









