####################################################
###           Monte Carlo Simulation             ###
####################################################
# Regime 1: Low Vol
# Regime 2: High Vol
####################################################
using Distributions, Optim, Statistics, StatsFuns, Random, DelimitedFiles, SpecialFunctions
include("utils.jl")
include("gray_dgp.jl")
include("gray_ml.jl")



function MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin) 
    if distri == "norm"
        params = Matrix{Float64}(undef, MC, 13);
        t_params = similar(params);
        for i = 1:MC
            println(i)
            Random.seed!(1234 + i);
            (r, h, Pt, s) = simulate_gray(n, distri, ω, α, β, P, burnin);
            θ̂ = fit_gray(r, k, nothing, distri);
            σ̂₁ = θ̂[1] / (1 - θ̂[3] - θ̂[5]);
            σ̂₂ = θ̂[2] / (1 - θ̂[4] - θ̂[6]);
            ĥ  = fore_gray(r, k, θ̂, distri);
            if σ̂₁ < σ̂₂
                params[i, :] = [θ̂; θ̂[3] + θ̂[5]; θ̂[4] + θ̂[6]; σ̂₁; σ̂₂; ĥ[3]];
            else
                params[i, :] = [θ̂[[2; 1; 4; 3; 6; 5; 8; 7]]; θ̂[4] + θ̂[6]; θ̂[3] + θ̂[5]; σ̂₂; σ̂₁; ĥ[3]];
            end
            η̂ = fit_gray_transform(r, k, nothing, distri);
            γ̂₁ = η̂[1] / (1 - η̂[3] - η̂[5]);
            γ̂₂ = η̂[2] / (1 - η̂[4] - η̂[6]);
            ĥ = fore_gray(r, k, η̂, distri);
            if γ̂₁ < γ̂₂
                t_params[i, :] = [η̂; η̂[3] + η̂[5]; η̂[4] + η̂[6]; γ̂₁; γ̂₂; ĥ[3]];
            else
                t_params[i, :] = [η̂[[2; 1; 4; 3; 6; 5; 8; 7]]; η̂[4] + η̂[6]; η̂[3] + η̂[5]; γ̂₂; γ̂₁; ĥ[3]];
            end
        end
    elseif(distri == "std")
        params = Matrix{Float64}(undef, MC, 14);
        for i = 1:MC
            println(i)
            Random.seed!(1234 + i);
            (r, h, Pt, s) = simulate_gray(n, distri, ω, α, β, P, burnin);
            θ̂ = fit_gray(r, k, nothing, distri);
            σ̂₁ = θ̂[1] / (1 - θ̂[3] - θ̂[5]);
            σ̂₂ = θ̂[2] / (1 - θ̂[4] - θ̂[6]);
            ĥ = fore_gray(r, k, θ̂, distri);
            if σ̂₁ < σ̂₂
                params[i, :] = [θ̂; θ̂[3] + θ̂[5]; θ̂[4] + θ̂[6]; σ̂₁; σ̂₂; ĥ[3]];
            else
                params[i, :] = [θ̂[[2; 1; 4; 3; 6; 5; 8; 7; 9]]; θ̂[4] + θ̂[6]; θ̂[3] + θ̂[5]; σ̂₂; σ̂₁; ĥ[3]];
            end
            η̂ = fit_gray_transform(r, k, nothing, distri);
            γ̂₁ = η̂[1] / (1 - η̂[3] - η̂[5]);
            γ̂₂ = η̂[2] / (1 - η̂[4] - η̂[6]);
            ĥ = fore_gray(r, k, η̂, distri);
            if γ̂₁ < γ̂₂
                t_params[i, :] = [η̂; η̂[3] + η̂[5]; η̂[4] + η̂[6]; γ̂₁; γ̂₂; ĥ[3]];
            else
                t_params[i, :] = [η̂[[2; 1; 4; 3; 6; 5; 8; 7; 9]]; η̂[4] + η̂[6]; η̂[3] + η̂[5]; γ̂₂; γ̂₁; ĥ[3]];
            end
        end
    else
        params = Matrix{Float64}(undef, MC, 14);
        for i = 1:MC
            println(i)
            Random.seed!(1234 + i);
            (r, h, Pt, s) = simulate_gray(n, distri, ω, α, β, P, burnin);
            θ̂ = fit_gray(r, k, nothing, distri);
            θ̂[9] = 1/θ̂[9];
            σ̂₁ = θ̂[1] / (1 - θ̂[3] - θ̂[5]);
            σ̂₂ = θ̂[2] / (1 - θ̂[4] - θ̂[6]);
            ĥ = fore_gray(r, k, θ̂, distri);
            if σ̂₁ < σ̂₂
                params[i, :] = [θ̂; θ̂[3] + θ̂[5]; θ̂[4] + θ̂[6]; σ̂₁; σ̂₂; ĥ[3]];
            else
                params[i, :] = [θ̂[[2; 1; 4; 3; 6; 5; 8; 7; 9]]; θ̂[4] + θ̂[6]; θ̂[3] + θ̂[5]; σ̂₂; σ̂₁; ĥ[3]];
            end
            η̂ = fit_gray_transform(r, k, nothing, distri);
            η̂[9] = 1/η̂[9];
            γ̂₁ = η̂[1] / (1 - η̂[3] - η̂[5]);
            γ̂₂ = η̂[2] / (1 - η̂[4] - η̂[6]);
            ĥ = fore_gray(r, k, η̂, distri);
            if γ̂₁ < γ̂₂
                t_params[i, :] = [η̂; η̂[3] + η̂[5]; η̂[4] + η̂[6]; γ̂₁; γ̂₂; ĥ[3]];
            else
                t_params[i, :] = [η̂[[2; 1; 4; 3; 6; 5; 8; 7; 9]]; η̂[4] + η̂[6]; η̂[3] + η̂[5]; γ̂₂; γ̂₁; ĥ[3]];
            end
        end
    end
    return params, t_params;
end


# Regime 1: Low Vol
# Regime 2: High Vol
MC = 1000
ω = [0.1, 0.05];
α = [0.2, 0.1];
β = [0.7, 0.4];
P = [0.9 0.03; 0.1 0.97];
k = 2;
burnin = 500;

#########################################################################################
n = 5000;
distri = "norm";
params_5000_n, t_params_5000_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_5000_n.csv",  params_5000_n, ',')
writedlm("t_params_5000_n.csv",  t_params_5000_n, ',')

distri = "istd";
params_5000_it, t_params_5000_it = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_5000_it.csv",  params_5000_it, ',')
writedlm("t_params_5000_it.csv",  t_params_5000_it, ',')
#########################################################################################
n = 2500;
distri = "norm";
params_2500_n, t_params_2500_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_2500_n.csv",  params_2500_n, ',')
writedlm("t_params_2500_n.csv",  t_params_2500_n, ',')

distri = "istd";
params_2500_it, t_params_2500_it = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_2500_it.csv",  params_2500_it, ',')
writedlm("t_params_2500_it.csv",  t_params_2500_it, ',')
#########################################################################################
n = 1000;
distri = "norm";
params_1000_n, t_params_1000_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_1000_n.csv",  params_1000_n, ',')
writedlm("t_params_1000_n.csv",  t_params_1000_n, ',')

distri = "istd";
params_1000_it, t_params_1000_it = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_1000_it.csv",  params_1000_it, ',')
writedlm("t_params_1000_it.csv",  t_params_1000_it, ',')
#########################################################################################