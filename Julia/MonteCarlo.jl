####################################################
###            Monte Carlo Function              ###
####################################################

function MonteCarlo(MC, n, ω, α, β, P, distri, k, burnin, dgp, outlier) 
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
            if outlier > 0 & outlier < n
                r[outlier] = r[outlier] + sign(r[outlier]) * 5 * std(r); 
            end
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
            if outlier > 0 & outlier < n
                r[outlier] = r[outlier] + sign(r[outlier]) * 5 * std(r); 
            end
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
