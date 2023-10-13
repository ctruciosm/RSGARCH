####################################################
###           Monte Carlo Simulation             ###
####################################################
# Regime 1: Low Vol
# Regime 2: High Vol
####################################################
using Distributions, Optim, Statistics, StatsFuns, Random, SpecialFunctions, TryCatch, DelimitedFiles, StatsBase, QuadGK, Kronecker

include("/home/ctrucios/Dropbox/Research/RegimeSwitching-GARCH/RSGARCH/Julia/utils.jl")
include("/home/ctrucios/Dropbox/Research/RegimeSwitching-GARCH/RSGARCH/Julia/DGP.jl")
include("/home/ctrucios/Dropbox/Research/RegimeSwitching-GARCH/RSGARCH/Julia/MaximumLikelihood.jl")
include("/home/ctrucios/Dropbox/Research/RegimeSwitching-GARCH/RSGARCH/Julia/Forecast.jl")


function MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, dgp) 
    # dgp: simulate_gray, simulate_klaassen, simulate_haas
    if distri == "norm"
        parameters = Matrix{Float64}(undef, MC, 38);
        for i = 1:MC
            println(i)
            Random.seed!(1234 + i);
            (r, h, Pt, s) = dgp(n, distri, ω, α, β, P, burnin);
            true_VaR_1, true_ES_1 = var_es_rsgarch(0.010, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "norm");
            true_VaR_2, true_ES_2 = var_es_rsgarch(0.025, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "norm");
            true_VaR_5, true_ES_5 = var_es_rsgarch(0.050, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "norm");
            # Include additive outliers
            #out_index = sample([1:1:n;], Integer(0.01*n));
            #r[out_index] .= r[out_index] .+ sign.(r[out_index]).*5*std(r); 
            # End 
            θ = fit_gray(r, k, nothing, distri);
            (ĥ_gray, Pt̂_gray, ŝ_gray)  = fore_gray(r, k, θ, distri);
            λ = fit_klaassen(r, k, nothing, distri);
            (ĥ_klaassen, Pt̂_klaassen, ŝ_klaassen)  = fore_klaassen(r, k, λ, distri);
            δ = fit_haas(r, k, nothing, distri);  
            (ĥ_haas, Pt̂_haas, ŝ_haas)  = fore_haas(r, k, δ, distri);

            esti_VaR_gray_1, esti_ES_gray_1 = var_es_rsgarch(0.010, Pt̂_gray[end], 1 - Pt̂_gray[end], sqrt(ĥ_gray[1]), sqrt(ĥ_gray[2]), "norm");  
            esti_VaR_gray_2, esti_ES_gray_2 = var_es_rsgarch(0.025, Pt̂_gray[end], 1 - Pt̂_gray[end], sqrt(ĥ_gray[1]), sqrt(ĥ_gray[2]), "norm");  
            esti_VaR_gray_5, esti_ES_gray_5 = var_es_rsgarch(0.050, Pt̂_gray[end], 1 - Pt̂_gray[end], sqrt(ĥ_gray[1]), sqrt(ĥ_gray[2]), "norm");  

            esti_VaR_klaassen_1, esti_ES_klaassen_1 = var_es_rsgarch(0.010, Pt̂_klaassen[end], 1 - Pt̂_klaassen[end], sqrt(ĥ_klaassen[1]), sqrt(ĥ_klaassen[2]), "norm");  
            esti_VaR_klaassen_2, esti_ES_klaassen_2 = var_es_rsgarch(0.025, Pt̂_klaassen[end], 1 - Pt̂_klaassen[end], sqrt(ĥ_klaassen[1]), sqrt(ĥ_klaassen[2]), "norm");  
            esti_VaR_klaassen_5, esti_ES_klaassen_5 = var_es_rsgarch(0.050, Pt̂_klaassen[end], 1 - Pt̂_klaassen[end], sqrt(ĥ_klaassen[1]), sqrt(ĥ_klaassen[2]), "norm");  

            esti_VaR_haas_1, esti_ES_haas_1 = var_es_rsgarch(0.010, Pt̂_haas[end], 1 - Pt̂_haas[end], sqrt(ĥ_haas[1]), sqrt(ĥ_haas[2]), "norm");  
            esti_VaR_haas_2, esti_ES_haas_2 = var_es_rsgarch(0.025, Pt̂_haas[end], 1 - Pt̂_haas[end], sqrt(ĥ_haas[1]), sqrt(ĥ_haas[2]), "norm");  
            esti_VaR_haas_5, esti_ES_haas_5 = var_es_rsgarch(0.050, Pt̂_haas[end], 1 - Pt̂_haas[end], sqrt(ĥ_haas[1]), sqrt(ĥ_haas[2]), "norm");  

            if dgp == simulate_gray
                ĥ = ĥ_gray;
                ϕ = θ;
            elseif dgp == simulate_klaassen
                ĥ = ĥ_klaassen;
                ϕ = λ
            else
                ĥ = ĥ_haas;
                ϕ = δ;
            end

            σ₁ = ϕ[1]/(1 - ϕ[3] - ϕ[5]);
            σ₂ = ϕ[2]/(1 - ϕ[4] - ϕ[6]); 

            if σ₁ < σ₂
                parameters[i, :] = [ϕ; ϕ[3] + ϕ[5]; ϕ[4] + ϕ[6]; σ₁; σ₂; ĥ[3]; h[end, 3]; 
                esti_VaR_gray_1; esti_ES_gray_1; esti_VaR_gray_2; esti_ES_gray_2; esti_VaR_gray_5; esti_ES_gray_5;
                esti_VaR_klaassen_1; esti_ES_klaassen_1; esti_VaR_klaassen_2; esti_ES_klaassen_2; esti_VaR_klaassen_5; esti_ES_klaassen_5;
                esti_VaR_haas_1; esti_ES_haas_1; esti_VaR_haas_2; esti_ES_haas_2; esti_VaR_haas_5; esti_ES_haas_5;
                true_VaR_1; true_ES_1; true_VaR_2; true_ES_2; true_VaR_5; true_ES_5];
            else
                parameters[i, :] = [ϕ[[2; 1; 4; 3; 6; 5; 8; 7]]; ϕ[4] + ϕ[6]; ϕ[3] + ϕ[5]; σ₂; σ₁; ĥ[3]; h[end, 3]; 
                esti_VaR_gray_1; esti_ES_gray_1; esti_VaR_gray_2; esti_ES_gray_2; esti_VaR_gray_5; esti_ES_gray_5;
                esti_VaR_klaassen_1; esti_ES_klaassen_1; esti_VaR_klaassen_2; esti_ES_klaassen_2; esti_VaR_klaassen_5; esti_ES_klaassen_5;
                esti_VaR_haas_1; esti_ES_haas_1; esti_VaR_haas_2; esti_ES_haas_2; esti_VaR_haas_5; esti_ES_haas_5;
                true_VaR_1; true_ES_1; true_VaR_2; true_ES_2; true_VaR_5; true_ES_5];
            end
        end
    elseif(distri == "student")
        parameters = Matrix{Float64}(undef, MC, 39);
        for i = 1:MC
            println(i)
            Random.seed!(1234 + i);
            (r, h, Pt, s) = dgp(n, distri, ω, α, β, P, burnin);
            true_VaR_1, true_ES_1 = var_es_rsgarch(0.010, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "student", 7.0);
            true_VaR_2, true_ES_2 = var_es_rsgarch(0.025, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "student", 7.0);
            true_VaR_5, true_ES_5 = var_es_rsgarch(0.050, Pt[end], 1 - Pt[end], sqrt(h[end, 1]), sqrt(h[end, 2]), "student", 7.0);
            # Include additive outliers
            #out_index = sample([1:1:n;], Integer(0.01*n));
            #r[out_index] .= r[out_index] .+ sign.(r[out_index]).*5*std(r); 
            # End 
            θ = fit_gray(r, k, nothing, distri);
            (ĥ_gray, Pt̂_gray, ŝ_gray)  = fore_gray(r, k, θ, distri);
            λ = fit_klaassen(r, k, nothing, distri);
            (ĥ_klaassen, Pt̂_klaassen, ŝ_klaassen)  = fore_klaassen(r, k, λ, distri);
            δ = fit_haas(r, k, nothing, distri);  
            (ĥ_haas, Pt̂_haas, ŝ_haas)  = fore_haas(r, k, δ, distri);

            esti_VaR_gray_1, esti_ES_gray_1 = var_es_rsgarch(0.010, Pt̂_gray[end], 1 - Pt̂_gray[end], sqrt(ĥ_gray[1]), sqrt(ĥ_gray[2]), "student", θ[9]);  
            esti_VaR_gray_2, esti_ES_gray_2 = var_es_rsgarch(0.025, Pt̂_gray[end], 1 - Pt̂_gray[end], sqrt(ĥ_gray[1]), sqrt(ĥ_gray[2]), "student", θ[9]);  
            esti_VaR_gray_5, esti_ES_gray_5 = var_es_rsgarch(0.050, Pt̂_gray[end], 1 - Pt̂_gray[end], sqrt(ĥ_gray[1]), sqrt(ĥ_gray[2]), "student", θ[9]);  

            esti_VaR_klaassen_1, esti_ES_klaassen_1 = var_es_rsgarch(0.010, Pt̂_klaassen[end], 1 - Pt̂_klaassen[end], sqrt(ĥ_klaassen[1]), sqrt(ĥ_klaassen[2]), "student", λ[9]);  
            esti_VaR_klaassen_2, esti_ES_klaassen_2 = var_es_rsgarch(0.025, Pt̂_klaassen[end], 1 - Pt̂_klaassen[end], sqrt(ĥ_klaassen[1]), sqrt(ĥ_klaassen[2]), "student", λ[9]);  
            esti_VaR_klaassen_5, esti_ES_klaassen_5 = var_es_rsgarch(0.050, Pt̂_klaassen[end], 1 - Pt̂_klaassen[end], sqrt(ĥ_klaassen[1]), sqrt(ĥ_klaassen[2]), "student", λ[9]);  

            esti_VaR_haas_1, esti_ES_haas_1 = var_es_rsgarch(0.010, Pt̂_haas[end], 1 - Pt̂_haas[end], sqrt(ĥ_haas[1]), sqrt(ĥ_haas[2]), "student", δ[9]);  
            esti_VaR_haas_2, esti_ES_haas_2 = var_es_rsgarch(0.025, Pt̂_haas[end], 1 - Pt̂_haas[end], sqrt(ĥ_haas[1]), sqrt(ĥ_haas[2]), "student", δ[9]);  
            esti_VaR_haas_5, esti_ES_haas_5 = var_es_rsgarch(0.050, Pt̂_haas[end], 1 - Pt̂_haas[end], sqrt(ĥ_haas[1]), sqrt(ĥ_haas[2]), "student", δ[9]);  

            if dgp == simulate_gray
                ĥ = ĥ_gray;
                ϕ = θ;
            elseif dgp == simulate_klaassen
                ĥ = ĥ_klaassen;
                ϕ = λ
            else
                ĥ = ĥ_haas;
                ϕ = δ;
            end

            σ₁ = ϕ[1]/(1 - ϕ[3] - ϕ[5]);
            σ₂ = ϕ[2]/(1 - ϕ[4] - ϕ[6]); 

            if σ₁ < σ₂
                parameters[i, :] = [ϕ; ϕ[3] + ϕ[5]; ϕ[4] + ϕ[6]; σ₁; σ₂; ĥ[3]; h[end, 3]; 
                esti_VaR_gray_1; esti_ES_gray_1; esti_VaR_gray_2; esti_ES_gray_2; esti_VaR_gray_5; esti_ES_gray_5;
                esti_VaR_klaassen_1; esti_ES_klaassen_1; esti_VaR_klaassen_2; esti_ES_klaassen_2; esti_VaR_klaassen_5; esti_ES_klaassen_5;
                esti_VaR_haas_1; esti_ES_haas_1; esti_VaR_haas_2; esti_ES_haas_2; esti_VaR_haas_5; esti_ES_haas_5;
                true_VaR_1; true_ES_1; true_VaR_2; true_ES_2; true_VaR_5; true_ES_5];
            else
                parameters[i, :] = [ϕ[[2; 1; 4; 3; 6; 5; 8; 7; 9]]; ϕ[4] + ϕ[6]; ϕ[3] + ϕ[5]; σ₂; σ₁; ĥ[3]; h[end, 3]; 
                esti_VaR_gray_1; esti_ES_gray_1; esti_VaR_gray_2; esti_ES_gray_2; esti_VaR_gray_5; esti_ES_gray_5;
                esti_VaR_klaassen_1; esti_ES_klaassen_1; esti_VaR_klaassen_2; esti_ES_klaassen_2; esti_VaR_klaassen_5; esti_ES_klaassen_5;
                esti_VaR_haas_1; esti_ES_haas_1; esti_VaR_haas_2; esti_ES_haas_2; esti_VaR_haas_5; esti_ES_haas_5;
                true_VaR_1; true_ES_1; true_VaR_2; true_ES_2; true_VaR_5; true_ES_5];
            end
        end
    else
        println("Only Normal ('norm') and Student-T ('student') distributions are available")
    end
    return parameters;
end


# Regime 1: Low Vol
# Regime 2: High Vol
MC = 500
# Parameters Gray 1996 - Table 3
#ω = [0.01, 0.18];
#α = [0.16, 0.46];
#β = [0.30, 0.20];
#P = [0.98 0.05; 0.02 0.95];
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
# Hotta, Trucios, Valls, Zevallos
ω = [0.01, 0.2];  #0.05 2
α = [0.05, 0.15];
β = [0.9, 0.65];
P = [0.95 0.05; 0.10 0.9];
k = 2;
burnin = 500;
#########################################################################################
### Normal
#########################################################################################
distri = "norm";
n = 5000;
parameters_5000_gray_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_gray);
writedlm("parameters_5000_gray_n_sim_04.csv",  parameters_5000_gray_n, ',')
parameters_5000_klaassen_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_klaassen);
writedlm("parameters_5000_klaassen_n_im_04.csv",  parameters_5000_klaassen_n, ',')
parameters_5000_haas_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_haas);
writedlm("parameters_5000_haas_n_sim_04.csv",  parameters_5000_haas_n, ',')

n = 2500;
parameters_2500_gray_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_gray);
writedlm("parameters_2500_gray_n_sim_04.csv",  parameters_2500_gray_n, ',')
parameters_2500_klaassen_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_klaassen);
writedlm("parameters_2500_klaassen_n_sim_04.csv",  parameters_2500_klaassen_n, ',')
parameters_2500_haas_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_haas);
writedlm("parameters_2500_haas_n_sim_04.csv",  parameters_2500_haas_n, ',')

n = 1000;
parameters_1000_gray_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_gray);
writedlm("parameters_1000_gray_n_sim_04.csv",  parameters_1000_gray_n, ',')
parameters_1000_klaassen_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_klaassen);
writedlm("parameters_1000_klaassen_n_sim_04.csv",  parameters_1000_klaassen_n, ',')
parameters_1000_haas_n = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_haas);
writedlm("parameters_1000_haas_n_sim_04.csv",  parameters_1000_haas_n, ',')


#########################################################################################
### Student-T
#########################################################################################
distri = "student";
n = 5000;
parameters_5000_gray_t = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_gray);
writedlm("parameters_5000_gray_t_sim_04.csv",  parameters_5000_gray_t, ',')
parameters_5000_klaassen_t = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_klaassen);
writedlm("parameters_5000_klaassen_t_sim_04.csv",  parameters_5000_klaassen_t, ',')
parameters_5000_haas_t = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_haas);
writedlm("parameters_5000_haas_t_sim_04.csv",  parameters_5000_haas_t, ',')

n = 2500;
parameters_2500_gray_t = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_gray);
writedlm("parameters_2500_gray_t_sim_04.csv",  parameters_2500_gray_t, ',')
parameters_2500_klaassen_t = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_klaassen);
writedlm("parameters_2500_klaassen_t_sim_04.csv",  parameters_2500_klaassen_t, ',')
parameters_2500_haas_t = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_haas);
writedlm("parameters_2500_haas_t_sim_04.csv",  parameters_2500_haas_t, ',')

n = 1000;
parameters_1000_gray_t = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_gray);
writedlm("parameters_1000_gray_t_sim_04.csv",  parameters_1000_gray_t, ',')
parameters_1000_klaassen_t = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_klaassen);
writedlm("parameters_1000_klaassen_t_sim_04.csv",  parameters_1000_klaassen_t, ',')
parameters_1000_haas_t = MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, simulate_haas);
writedlm("parameters_1000_haas_t_sim_04.csv",  parameters_1000_haas_t, ',')



