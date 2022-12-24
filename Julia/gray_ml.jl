##################################################
###  RSGARCH Estim: Estimate RSGARCH Models   ####
##################################################
 
function gray_likelihood(r::Vector{Float64}, k::Int64, distri::String, par)
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

    if sum(α + β .< 0.999) == 2
        Pt[1] = (1 - q) / (2 - p - q);              
        h[1, 1:k] .= var(r);                        
        h[1, k + 1] = Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2];
        if distri == "norm"
            @inbounds for i = 2:n
                h[i, 1:k] = ω .+ α .* r[i - 1]^2 + β .* h[i - 1, k + 1];
                Pt[i] = probability_regime_given_time_n(p, q, sqrt.(h[i - 1, :]), r[i - 1], Pt[i - 1]);
                h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
                log_lik[i - 1] = log(pdf(Normal(0, sqrt(h[i, 1])), r[i]) * Pt[i] + pdf(Normal(0, sqrt(h[i, 2])), r[i]) * (1 - Pt[i]));
            end
        elseif distri == "std"
            ν = par[4 * k + 1];
            @inbounds for i = 2:n
                h[i, 1:k] = ω .+ α .* r[i - 1]^2 + β .* h[i - 1, k + 1];
                Pt[i] = probability_regime_given_time_t(p, q, sqrt.(h[i- 1, :]), r[i - 1], Pt[i - 1], ν);
                h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
                log_lik[i - 1] = log(sqrt(ν/(ν - 2)) / sqrt(h[i, 1]) * pdf(TDist(ν), r[i] * sqrt(ν/(ν - 2)) / sqrt(h[i, 1])) * Pt[i] + sqrt(ν/(ν - 2)) / sqrt(h[i, 2]) * pdf(TDist(ν), r[i] * sqrt(ν/(ν - 2)) / sqrt(h[i, 2])) * (1 - Pt[i]));
            end
        else
            η = par[4 * k + 1];
            @inbounds for i = 2:n
                h[i, 1:k] = ω .+ α .* r[i - 1]^2 + β .* h[i - 1, k + 1];
                Pt[i] = probability_regime_given_time_it(p, q, sqrt.(h[i- 1, :]), r[i - 1], Pt[i - 1], η);
                h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
                log_lik[i - 1] = log( 1/ sqrt(h[i, 1]) * Tstudent(r[i] / sqrt(h[i, 1]), η)* Pt[i]  + 1 / sqrt(h[i, 2]) * Tstudent(r[i] / sqrt(h[i, 2]), η) * (1 - Pt[i]));
                
            end
        end
        return -sum(log_lik)/2;
    else
        return 999999999999999.0 + rand(Uniform(1, 2), 1)[1];
    end
end


function fit_gray(r::Vector{Float64}, k::Int64, par_ini, distri::String)
    if (distri == "norm")
        if isnothing(par_ini)
            par_ini = [0.05, 0.15, 0.3, 0.1, 0.6, 0.2, 0.85, 0.92];     
            ll = gray_likelihood(r, k, distri, par_ini);
            for i in 1:1000                                 
                ω = rand(Uniform(0.0001, 0.4), k);
                α = rand(Uniform(0.01, 0.6), k);
                β = [rand(Uniform(0.1, 1 - α[1] - 1e-6), 1); rand(Uniform(0.1, 1 - α[2] - 1e-6), 1)];
                p = rand(Uniform(0.8, 0.99), k);
                par_random = [ω; α; β; p];
                if gray_likelihood(r, k, distri, par_random) < ll
                    ll = gray_likelihood(r, k, distri, par_random);
                    par_ini = par_random;
                end
            end
        end    
        optimum = optimize(par -> gray_likelihood(r, k, distri, par), [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6 ,1e-6 ,1e-6], [Inf, Inf, 1, 1, 1, 1, 1, 1], par_ini);
    elseif(distri == "std")
        if isnothing(par_ini)
            par_ini = [0.05, 0.15, 0.3, 0.1, 0.6, 0.2, 0.85, 0.92, 5];    
            ll = gray_likelihood(r, k, distri, par_ini);
            for i in 1:1000                                 
                ω = rand(Uniform(0.0001, 0.4), k);
                α = rand(Uniform(0.01, 0.6), k);
                β = [rand(Uniform(0.1, 1 - α[1] - 1e-6), 1); rand(Uniform(0.1, 1 - α[2] - 1e-6), 1)];
                p = rand(Uniform(0.8, 0.99), k);
                ν = rand(Uniform(4, 35), 1);
                par_random = [ω; α; β; p; ν];
                if gray_likelihood(r, k, distri, par_random) < ll
                    ll = gray_likelihood(r, k, distri, par_random);
                    par_ini = par_random;
                end
            end
        end
        optimum = optimize(par -> gray_likelihood(r, k, distri, par), [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6 ,1e-6 ,1e-6, 4], [Inf, Inf, 1, 1, 1, 1, 1, 1, Inf], par_ini);
    else
        if isnothing(par_ini)
            par_ini = [0.05, 0.15, 0.3, 0.1, 0.6, 0.2, 0.85, 0.92, 0.2];    
            ll = gray_likelihood(r, k, distri, par_ini);
            for i in 1:1000                                 
                ω = rand(Uniform(0.0001, 0.4), k);
                α = rand(Uniform(0.01, 0.6), k);
                β = [rand(Uniform(0.1, 1 - α[1] - 1e-6), 1); rand(Uniform(0.1, 1 - α[2] - 1e-6), 1)];
                p = rand(Uniform(0.8, 0.99), k);
                η = rand(Uniform(0.01, 0.49), 1);
                par_random = [ω; α; β; p; η];
                if gray_likelihood(r, k, distri, par_random) < ll
                    ll = gray_likelihood(r, k, distri, par_random);
                    par_ini = par_random;
                end
            end
        end
        optimum = optimize(par -> gray_likelihood(r, k, distri, par), [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6 ,1e-6 ,1e-6, 0.01], [Inf, Inf, 1, 1, 1, 1, 1, 1, 0.49], par_ini);
    end
    mle = optimum.minimizer;
    return mle;
end


function fore_gray(r::Vector{Float64}, k::Int64, par, distri::String)

    n = length(r);
    h = Matrix{Float64}(undef, n + 1, k + 1);
    s = Vector{Float64}(undef, n + 1);
    e = Vector{Float64}(undef, n + 1);
    Pt = Vector{Float64}(undef, n + 1);

    ω = par[1:k];
    α = par[k + 1 : 2 * k];
    β = par[2 * k + 1 : 3 * k];
    p = par[3 * k + 1];
    q = par[4 * k];

    Pt[1] = (1 - q) / (2 - p - q);              
    h[1, 1:k] .= var(r);                        
    h[1, k + 1] = Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2];
    s[1] = wsample([1, 2], [Pt[1], 1 - Pt[1]])[1];
    e[1] = r[1] / sqrt(h[1, s[i]]);

    if distri == "norm"
        @inbounds for i = 2:n+1
            h[i, 1:k] = ω .+ α .* r[i - 1]^2 + β .* h[i - 1, k + 1];
            Pt[i] = probability_regime_given_time_n(p, q, sqrt.(h[i - 1, :]), r[i - 1], Pt[i - 1]);
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
            s[i] = wsample([1, 2], P[:, s[i-1]])[1]; 
            e[i] = r[i] / sqrt(h[i, s[i]]);
        end
    else
        η = par[4 * k + 1];
        @inbounds for i = 2:n+1
            h[i, 1:k] = ω .+ α .* r[i - 1]^2 + β .* h[i - 1, k + 1];
            Pt[i] = probability_regime_given_time_it(p, q, sqrt.(h[i- 1, :]), r[i - 1], Pt[i - 1], η);
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];     
            s[i] = wsample([1, 2], P[:, s[i-1]])[1]; 
            r[i] = e[i] * sqrt(h[i, s[i]]);       
        end
    end



end





