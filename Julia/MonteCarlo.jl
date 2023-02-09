####################################################
###           Monte Carlo Simulation             ###
####################################################
# Regime 1: Low Vol
# Regime 2: High Vol
####################################################
using Distributions, Optim, Statistics, StatsFuns, Random, SpecialFunctions, TryCatch, DelimitedFiles, StatsBase, QuadGK

include("utils.jl")
include("DGP.jl")
include("MaximumLikelihood.jl")
include("Forecast.jl")


function MonteCarlo_Gray(MC, n, ω, α, β, P, distri, k, burnin) 
    if distri == "norm"
        params = Matrix{Float64}(undef, MC, 20);
        for i = 1:MC
            println(i)
            Random.seed!(1234 + i);
            (r, h, Pt, s) = simulate_gray(n, distri, ω, α, β, P, burnin);
            θ = fit_gray(r, k, nothing, distri);
            σ₁ = θ[1] / (1 - θ[3] - θ[5]);
            σ₂ = θ[2] / (1 - θ[4] - θ[6]);
            (ĥ, Pt̂, ŝ)  = fore_gray(r, k, θ, distri);
            true_VaR, true_ES = var_es_rsgarch(0.01, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "norm");
            esti_VaR, esti_ES = var_es_rsgarch(0.01, Pt̂[end], 1 - Pt̂[end], sqrt(ĥ[1]), sqrt(ĥ[2]), "norm");  
            δ = fit_haas(r, k, nothing, distri);  
            (ĥ_haas, Pt̂_haas, ŝ_haas)  = fore_haas(r, k, δ, distri);
            esti_VaR_haas, esti_ES_haas = var_es_rsgarch(0.01, Pt̂_haas[end], 1 - Pt̂_haas[end], sqrt(ĥ_haas[1]), sqrt(ĥ_haas[2]), "norm");  
            if σ₁ < σ₂
                params[i, :] = [θ; θ[3] + θ[5]; θ[4] + θ[6]; σ₁; σ₂; ĥ[3]; h[end, 3]; esti_VaR; esti_ES; true_VaR; true_ES; esti_VaR_haas; esti_ES_haas, mean(ŝ .== s)];
            else
                params[i, :] = [θ[[2; 1; 4; 3; 6; 5; 8; 7]]; θ[4] + θ[6]; θ[3] + θ[5]; σ₂; σ₁; ĥ[3]; h[end, 3]; esti_VaR; esti_ES; true_VaR; true_ES; esti_VaR_haas; esti_ES_haas];
            end
        end
    elseif(distri == "student")
        params = Matrix{Float64}(undef, MC, 21);
        for i = 1:MC
            println(i)
            Random.seed!(1234 + i);
            (r, h, Pt, s) = simulate_gray(n, distri, ω, α, β, P, burnin);
            θ = fit_gray(r, k, nothing, distri);
            σ₁ = θ[1] / (1 - θ[3] - θ[5]);
            σ₂ = θ[2] / (1 - θ[4] - θ[6]);
            (ĥ, Pt̂, ŝ)  = fore_gray(r, k, θ, distri);
            true_VaR, true_ES = var_es_rsgarch(0.01, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "student", 7.0);
            esti_VaR, esti_ES = var_es_rsgarch(0.01, Pt̂[end], 1 - Pt̂[end], sqrt(ĥ[1]), sqrt(ĥ[2]), "student", θ[9]);  
            δ = fit_haas(r, k, nothing, distri);  
            (ĥ_haas, Pt̂_haas, ŝ_haas)  = fore_haas(r, k, δ, distri);
            esti_VaR_haas, esti_ES_haas = var_es_rsgarch(0.01, Pt̂_haas[end], 1 - Pt̂_haas[end], sqrt(ĥ_haas[1]), sqrt(ĥ_haas[2]), "student", δ[9]);  
            if σ₁ < σ₂
                params[i, :] = [θ; θ[3] + θ[5]; θ[4] + θ[6]; σ₁; σ₂; ĥ[3]; h[end, 3]; esti_VaR; esti_ES; true_VaR; true_ES; esti_VaR_haas; esti_ES_haas];
            else
                params[i, :] = [θ[[2; 1; 4; 3; 6; 5; 8; 7; 9]]; θ[4] + θ[6]; θ[3] + θ[5]; σ₂; σ₁; ĥ[3]; h[end, 3]; esti_VaR; esti_ES; true_VaR; true_ES; esti_VaR_haas; esti_ES_haas];
            end
        end
    else
        println("Only Normal ('norm') and Student-T ('student') distributions are available")
    end
    return params;
end

function MonteCarlo_Haas(MC, n, ω, α, β, P, distri, k, burnin) 
    if distri == "norm"
        params = Matrix{Float64}(undef, MC, 20);
        for i = 1:MC
            println(i)
            Random.seed!(1234 + i);
            (r, h, Pt, s) = simulate_haas(n, distri, ω, α, β, P, burnin);
            θ = fit_haas(r, k, nothing, distri);
            σ₁ = θ[1] / (1 - θ[3] - θ[5]);
            σ₂ = θ[2] / (1 - θ[4] - θ[6]);
            (ĥ, Pt̂, ŝ)  = fore_haas(r, k, θ, distri);
            true_VaR, true_ES = var_es_rsgarch(0.01, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "norm");
            esti_VaR, esti_ES = var_es_rsgarch(0.01, Pt̂[end], 1 - Pt̂[end], sqrt(ĥ[1]), sqrt(ĥ[2]), "norm");  
            δ = fit_gray(r, k, nothing, distri);  
            (ĥ_gray, Pt̂_gray, ŝ_gray)  = fore_gray(r, k, δ, distri);
            esti_VaR_gray, esti_ES_gray = var_es_rsgarch(0.01, Pt̂_gray[end], 1 - Pt̂_gray[end], sqrt(ĥ_gray[1]), sqrt(ĥ_gray[2]), "norm");  
            if σ₁ < σ₂
                params[i, :] = [θ; θ[3] + θ[5]; θ[4] + θ[6]; σ₁; σ₂; ĥ[3]; h[end, 3]; esti_VaR; esti_ES; true_VaR; true_ES; esti_VaR_gray; esti_ES_gray];
            else
                params[i, :] = [θ[[2; 1; 4; 3; 6; 5; 8; 7]]; θ[4] + θ[6]; θ[3] + θ[5]; σ₂; σ₁; ĥ[3]; h[end, 3]; esti_VaR; esti_ES; true_VaR; true_ES; esti_VaR_gray; esti_ES_gray];
            end
        end
    elseif(distri == "student")
        params = Matrix{Float64}(undef, MC, 21);
        for i = 1:MC
            println(i)
            Random.seed!(1234 + i);
            (r, h, Pt, s) = simulate_haas(n, distri, ω, α, β, P, burnin);
            θ = fit_haas(r, k, nothing, distri);
            σ₁ = θ[1] / (1 - θ[3] - θ[5]);
            σ₂ = θ[2] / (1 - θ[4] - θ[6]);
            (ĥ, Pt̂, ŝ)  = fore_haas(r, k, θ, distri);
            true_VaR, true_ES = var_es_rsgarch(0.01, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "student", 7.0);
            esti_VaR, esti_ES = var_es_rsgarch(0.01, Pt̂[end], 1 - Pt̂[end], sqrt(ĥ[1]), sqrt(ĥ[2]), "student", θ[9]);  
            δ = fit_gray(r, k, nothing, distri);  
            (ĥ_gray, Pt̂_gray, ŝ_gary)  = fore_gray(r, k, δ, distri);
            esti_VaR_gray, esti_ES_gray = var_es_rsgarch(0.01, Pt̂_gray[end], 1 - Pt̂_gray[end], sqrt(ĥ_gray[1]), sqrt(ĥ_gray[2]), "student", δ[9]);  
            if σ₁ < σ₂
                params[i, :] = [θ; θ[3] + θ[5]; θ[4] + θ[6]; σ₁; σ₂; ĥ[3]; h[end, 3]; esti_VaR; esti_ES; true_VaR; true_ES; esti_VaR_gray; esti_ES_gray];
            else
                params[i, :] = [θ[[2; 1; 4; 3; 6; 5; 8; 7; 9]]; θ[4] + θ[6]; θ[3] + θ[5]; σ₂; σ₁; ĥ[3]; h[end, 3]; esti_VaR; esti_ES; true_VaR; true_ES; esti_VaR_gray; esti_ES_gray];
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
# Parameters Gray 1996 - Table 3
ω = [0.01, 0.18];
α = [0.16, 0.46];
β = [0.30, 0.20];
P = [0.98 0.05; 0.02 0.95];
# Parameters Hass 2004 - Table 3
#ω = [0.005, 0.1];
#α = [0.025, 0.25];
#β = [0.95, 0.70];
#P = [0.75 0.30; 0.25 0.70];
# Parameter Marcucci 2005 - Table 3
#ω = [0.003, 0.09];
#α = [0.015, 0.07];
#β = [0.98, 0.85];
#P = [0.99 0.01; 0.01 0.99];
k = 2;
burnin = 500;
#########################################################################################
### Normal
#########################################################################################
distri = "norm";
n = 5000;
params_5000_gray_n = MonteCarlo_Gray(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_5000_gray_n.csv",  params_5000_gray_n, ',')
params_5000_haas_n = MonteCarlo_Haas(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_5000_haas_n.csv",  params_5000_haas_n, ',')

n = 2500;
params_2500_gray_n = MonteCarlo_Gray(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_2500_gray_n.csv",  params_2500_gray_n, ',')
params_2500_haas_n = MonteCarlo_Haas(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_2500_haas_n.csv",  params_2500_haas_n, ',')

n = 1000;
params_1000_gray_n = MonteCarlo_Gray(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_1000_gray_n.csv",  params_1000_gray_n, ',')
params_1000_haas_n = MonteCarlo_Haas(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_1000_haas_n.csv",  params_1000_haas_n, ',')



#########################################################################################
### Student-T
#########################################################################################
distri = "student";
n = 5000;
params_5000_haas_t = MonteCarlo_Haas(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_5000_haas_t.csv",  params_5000_haas_t, ',')
params_5000_gray_t = MonteCarlo_Gray(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_5000_gray_t.csv",  params_5000_gray_t, ',')

n = 2500;
params_2500_gray_t = MonteCarlo_Gray(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_2500_gray_t.csv",  params_2500_gray_t, ',')
params_2500_haas_t = MonteCarlo_Haas(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_2500_haas_t.csv",  params_2500_haas_t, ',')

n = 1000;
params_1000_gray_t = MonteCarlo_Gray(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_1000_gray_t.csv",  params_1000_gray_t, ',')
params_1000_haas_t = MonteCarlo_Haas(MC, n, ω, α, β, P, distri, k, burnin);
writedlm("params_1000_haas_t.csv",  params_1000_haas_t, ',')



