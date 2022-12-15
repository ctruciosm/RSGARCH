##################################################
###  RSGARCH Estim: Estimate RSGARCH Models   ####
##################################################
 
function gray_likelihood(par, r, k, distri)
    # par(numeric vector): [ω, α, β, p11, p22, 1/gl]
    n = length(r);
    h = Matrix{Float64}(undef, n, k + 1);
    Pt = Vector{Float64}(undef, n);
    log_lik = Vector{Float64}(undef, n - 1);

    ω = par[1:k];
    α = par[k + 1 : 2 * k];
    β = par[2 * k + 1 : 3 * k];
    p = par[3 * k + 1];
    q = par[4 * k];

    if sum(α + β .< 1) == 2
        Pt[1] = (1 - q) / (2 - p - q);              ## Pi = P(St = 1) - Pag 683 Hamilton (1994)
        h[1, 1:k] .= var(r[1:30]);                        
        h[1, k + 1] = Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2];
        if distri == "norm"
            for i = 2:n
                h[i, 1:k] = ω .+ α .* r[i - 1]^2 + β .* h[i - 1, k + 1];
                Pt[i] = probability_regime_given_time_n(p, q, sqrt.(h[i - 1, :]), r[i - 1], Pt[i - 1]);
                h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
                log_lik[i - 1] = log(pdf(Normal(0, sqrt(h[i, 1])), r[i]) * Pt[i] + pdf(Normal(0, sqrt(h[i, 2])), r[i]) * (1 - Pt[i]));
            end
        else
            η = par[4 * k + 1];
            ν = 1/η;
            for i = 2:n
                h[i, 1:k] = ω .+ α .* r[i - 1]^2 + β .* h[i - 1, k + 1];
                Pt[i] = probability_regime_given_time_t(p, q, sqrt.(h[i- 1, :]), r[i - 1], Pt[i - 1], ν);
                h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
                log_lik[i - 1] = log(sqrt(ν/(ν - 2)) / sqrt(h[i, 1]) * pdf(TDist(ν), r[i] * sqrt(ν/(ν - 2)) / sqrt(h[i, 1])) * Pt[i] + sqrt(ν/(ν - 2)) / sqrt(h[i, 2]) * pdf(TDist(ν), r[i] * sqrt(ν/(ν - 2)) / sqrt(h[i, 2])) * (1 - Pt[i]));
            end
        end
        return -sum(log_lik)/2;
    else
        return 99999999999.0 + rand(Uniform(1, 2), 1)[1];
    end
end


function fit_gray(r, k, par_ini, distri)
    if distri == "norm"
        if isnothing(par_ini)
            par_ini = [0.05, 0.15, 0.3, 0.1, 0.6, 0.2, 0.85, 0.92];     
            ll = gray_likelihood(par_ini, r, k, distri);
            for i in 1:1000                                 
                ω = rand(Uniform(0.0001, 0.4), k);
                α = rand(Uniform(0.01, 0.6), k);
                β = [rand(Uniform(0.1, 1 - α[1] - 1e-6), 1); rand(Uniform(0.1, 1 - α[2] - 1e-6), 1)];
                p = rand(Uniform(0.8, 0.99), k);
                par_random = [ω; α; β; p];
                if gray_likelihood(par_random, r, k, distri) < ll
                    ll = gray_likelihood(par_random, r, k, distri);
                    par_ini = par_random;
                end
            end
        end    
        optimum = optimize(par -> gray_likelihood(par, r, k, distri), [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6 ,1e-6 ,1e-6], [Inf, Inf, 1, 1, 1, 1, 1, 1], par_ini);
    else
        if isnothing(par_ini)
            par_ini = [0.05, 0.15, 0.3, 0.1, 0.6, 0.2, 0.85, 0.92, 0.2];    
            ll = gray_likelihood(par_ini, r, k, distri);
            for i in 1:1000                                 
                ω = rand(Uniform(0.0001, 0.4), k);
                α = rand(Uniform(0.01, 0.6), k);
                β = [rand(Uniform(0.1, 1 - α[1] - 1e-6), 1); rand(Uniform(0.1, 1 - α[2] - 1e-6), 1)];
                p = rand(Uniform(0.8, 0.99), k);
                η = rand(Uniform(0.01, 0.5), 1);
                par_random = [ω; α; β; p; η];
                if gray_likelihood(par_random, r, k, distri) < ll
                    ll = gray_likelihood(par_random, r, k, distri);
                    par_ini = par_random;
                end
            end
        end
        optimum = optimize(par -> gray_likelihood(par, r, k, distri), [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6 ,1e-6 ,1e-6, 1e-2], [Inf, Inf, 1, 1, 1, 1, 1, 1, 0.5], par_ini);
    end
    mle = optimum.minimizer;
    return mle;
end




