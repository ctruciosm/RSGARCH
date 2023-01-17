function gray_likelihood2(par, r, k, distri)
    # par = numeric vector: omega, alpha, beta, p11, p22
    n = length(r);
    h = Array{Float64}(undef, n, k + 1);
    Pt = Vector{Float64}(undef, n);
    log_lik = Vector{Float64}(undef, n - 1);

    ω = par[1:k];
    α = par[k + 1 : 2 * k];
    β = par[2 * k + 1 : 3 * k];
    p = par[3 * k + 1];
    q = par[4 * k];

    Pt[1] = (1 - q) / (2 - p - q);          ## Pi = P(St = 1) - Pag 683 Hamilton (1994)
    h[1, 1:k] .= var(r);                     ## See Fig 2 in Gray (1996)
    h[1, k + 1] = Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2];

    if (distri == "norm")
        for i = 2:n
            numA = (1 - q) * pdf(Normal(0, sqrt(h[i - 1, 2])), r[i - 1]) * (1 - Pt[i - 1]);
            numB = p * pdf(Normal(0, sqrt(h[i - 1, 1])), r[i - 1]) * Pt[i - 1];
            deno = pdf(Normal(0, sqrt(h[i - 1, 1])), r[i - 1]) * Pt[i - 1] + pdf(Normal(0, sqrt(h[i - 1, 2])), r[i - 1]) * (1 - Pt[i - 1]);
            Pt[i] = numA/deno + numB/deno;
            h[i, 1:k] = ω .+ α .* r[i - 1]^2 + β .* h[i - 1, k + 1];
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
            log_lik[i - 1] = log(pdf(Normal(0, sqrt(h[i, 1])), r[i]) * Pt[i] + pdf(Normal(0, sqrt(h[i, 2])), r[i]) * (1 - Pt[i]));
        end
    else
        ν = par[4 * k + 1];
        for i = 2:n
            numA = (1 - q) * sqrt(ν/(ν - 2)) / sqrt(h[i - 1, 2]) * pdf(TDist(ν), r[i - 1] * sqrt(ν/(ν - 2)) / sqrt(h[i - 1, 2])) * (1 - Pt[i - 1]);
            numB = p * sqrt(ν/(ν - 2)) / sqrt(h[i - 1, 1]) * pdf(TDist(ν), r[i - 1] * sqrt(ν/(ν - 2)) / sqrt(h[i - 1, 1])) * Pt[i - 1];
            deno = sqrt(ν/(ν - 2)) / sqrt(h[i - 1, 1]) * pdf(TDist(ν), r[i - 1] * sqrt(ν/(ν - 2)) / sqrt(h[i - 1, 1])) * Pt[i - 1] + 
            sqrt(ν/(ν - 2)) / sqrt(h[i - 1, 2]) * pdf(TDist(ν), r[i - 1] * sqrt(ν/(ν - 2)) / sqrt(h[i - 1, 2])) * (1 - Pt[i - 1]);
            Pt[i] = numA/deno + numB/deno;
            h[i, 1:k] = ω .+ α .* r[i - 1]^2 + β .* h[i - 1, k + 1];
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
            log_lik[i - 1] = log(sqrt(ν/(ν - 2)) / sqrt(h[i, 1]) * pdf(TDist(ν), r[i] * sqrt(ν/(ν - 2)) / sqrt(h[i, 1])) * Pt[i] + 
            sqrt(ν/(ν - 2)) / sqrt(h[i, 2]) * pdf(TDist(ν), r[i] * sqrt(ν/(ν - 2)) / sqrt(h[i, 2])) * (1 - Pt[i]));
        end
    end
    return -mean(log_lik);
end

function fit_gray2(r, k, par_ini, distri)
    if distri == "norm"
        if isnothing(par_ini)
            par_ini = [0.05, 0.15, 0.3, 0.1, 0.6, 0.2, 0.85, 0.92];     
            ll = gray_likelihood(par_ini, r, k, distri);
            for i in 1:1000                                 
                ω = rand(Uniform(0.01, 0.3), 2);
                α = rand(Uniform(0.05, 0.5), 2);
                β = [rand(Uniform(0.4, max(0.41, 1 - α[1])), 1); rand(Uniform(0.4, max(0.41, 1 - α[2])), 1)];
                p = rand(Uniform(0.8, 0.99), 2);
                par_random = [ω; α; β; p];
                if gray_likelihood(par_random, r, k, distri) < ll
                    ll = gray_likelihood(par_random, r, k, distri);
                    par_ini = par_random;
                end
            end
        end
        optimum = optimize(par -> gray_likelihood(par, r, k, distri), [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6 ,1e-6, 1e-6], [Inf, Inf, 1, 1, 1, 1, 1, 1], par_ini);
    else
        if isnothing(par_ini)
            par_ini = [0.05, 0.15, 0.3, 0.1, 0.6, 0.2, 0.85, 0.92, 5];    
            ll = gray_likelihood2(par_ini, r, k, distri);
            for i in 1:1000                                 
                ω = rand(Uniform(0.01, 0.3), 2);
                α = rand(Uniform(0.05, 0.5), 2);
                β = [rand(Uniform(0.4, max(0.41, 1 - α[1])), 1); rand(Uniform(0.4, max(0.41, 1 - α[2])), 1)];
                p = rand(Uniform(0.8, 0.99), 2);
                ν = rand(Uniform(2, 35), 1);
                par_random = [ω; α; β; p; ν];
                if gray_likelihood2(par_random, r, k, distri) < ll
                    ll = gray_likelihood2(par_random, r, k, distri);
                    par_ini = par_random;
                end
            end
        end
        optimum = optimize(par -> gray_likelihood2(par, r, k, distri), [1e-6, 1e-6, 1e-6, 1e-6 ,1e-6 ,1e-6 ,1e-6 ,1e-6, 2], [Inf, Inf, 1, 1, 1, 1, 1, 1, Inf], par_ini);
    end
    mle = optimum.minimizer;
    return mle;
end


function Tstudent(x, η)
    a = (gamma((1 + η) / (2 * η)))/(sqrt(π * (1 - 2 * η)/η) * gamma(1 / (2 * η)));
    b = (1 + η * x^2/(1 - 2 * η))^((η + 1)/(2 * η));
    return a/b;
end




function haas_likelihood(r::Vector{Float64}, k::Int64, distri::String, par)
    # par = numeric vector: omega, alpha, beta, p11, p22
    n = length(r);
    h = Matrix{Float64}(undef, n, k);
    Pt = Vector{Float64}(undef, n);
    log_lik = Vector{Float64}(undef, n - 1);

    ω = par[1:k];
    α = par[k + 1 : 2 * k];
    β = par[2 * k + 1 : 3 * k];
    p = par[3 * k + 1];
    q = par[4 * k];

    if sum(α + β .< 1) == 2
        Pt[1] = (1 - q) / (2 - p - q);             
        h[1, :] .= ω ./ (1 .- (α + β));           
 
        if (distri == "norm")
            for i = 2:n
                numA = (1 - q) * pdf(Normal(0, sqrt(h[i - 1, 2])), r[i - 1]) * (1 - Pt[i - 1]);
                numB = p * pdf(Normal(0, sqrt(h[i - 1, 1])), r[i - 1]) * Pt[i - 1];
                deno = pdf(Normal(0, sqrt(h[i - 1, 1])), r[i - 1]) * Pt[i - 1] + pdf(Normal(0, sqrt(h[i - 1, 2])), r[i - 1]) * (1 - Pt[i - 1]);
                Pt[i] = numA/deno + numB/deno;
                h[i, :] = ω .+ α .* r[i - 1]^2 + β .* h[i - 1, :];
                log_lik[i - 1] = log(pdf(Normal(0, sqrt(h[i, 1])), r[i]) * Pt[i] + pdf(Normal(0, sqrt(h[i, 2])), r[i]) * (1 - Pt[i]));
            end
        else
            η = par[4 * k + 1];
            ν = 1/η;
            for i = 2:n
                numA = (1 - q) * sqrt(ν/(ν - 2)) / sqrt(h[i - 1, 2]) * pdf(TDist(ν), r[i - 1] * sqrt(ν/(ν - 2)) / sqrt(h[i - 1, 2])) * (1 - Pt[i - 1]);
                numB = p * sqrt(ν/(ν - 2)) / sqrt(h[i - 1, 1]) * pdf(TDist(ν), r[i - 1] * sqrt(ν/(ν - 2)) / sqrt(h[i - 1, 1])) * Pt[i - 1];
                deno = sqrt(ν/(ν - 2)) / sqrt(h[i - 1, 1]) * pdf(TDist(ν), r[i - 1] * sqrt(ν/(ν - 2)) / sqrt(h[i - 1, 1])) * Pt[i - 1] + 
                sqrt(ν/(ν - 2)) / sqrt(h[i - 1, 2]) * pdf(TDist(ν), r[i - 1] * sqrt(ν/(ν - 2)) / sqrt(h[i - 1, 2])) * (1 - Pt[i - 1]);
                Pt[i] = numA/deno + numB/deno;
                h[i, :] = ω .+ α .* r[i - 1]^2 + β .* h[i - 1, :];
                log_lik[i - 1] = log(sqrt(ν/(ν - 2)) / sqrt(h[i, 1]) * pdf(TDist(ν), r[i] * sqrt(ν/(ν - 2)) / sqrt(h[i, 1])) * Pt[i] + 
                sqrt(ν/(ν - 2)) / sqrt(h[i, 2]) * pdf(TDist(ν), r[i] * sqrt(ν/(ν - 2)) / sqrt(h[i, 2])) * (1 - Pt[i]));
            end
        end
        return -sum(log_lik)/2;
    else
        return 999999999.0 + rand(Uniform(1, 2), 1)[1];
    end 
end


function fit_haas(r, k, par_ini, distri)
    if distri == "norm"
        if isnothing(par_ini)
            par_ini = [0.05, 0.15, 0.3, 0.1, 0.6, 0.2, 0.85, 0.92];     
            ll = haas_likelihood(par_ini, r, k, distri);
            for i in 1:1000                                 
                ω = rand(Uniform(0.01, 0.3), 2);
                α = rand(Uniform(0.05, 0.5), 2);
                β = [rand(Uniform(0.4, max(0.41, 1 - α[1])), 1); rand(Uniform(0.4, max(0.41, 1 - α[2])), 1)];
                p = rand(Uniform(0.8, 0.99), 2);
                par_random = [ω; α; β; p];
                if haas_likelihood(par_random, r, k, distri) < ll
                    ll = haas_likelihood(par_random, r, k, distri);
                    par_ini = par_random;
                end
            end
        end
        optimum = optimize(par -> haas_likelihood(par, r, k, distri), [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6], [Inf, Inf, 1, 1, 1, 1, 1, 1], par_ini);
    else
        if isnothing(par_ini)
            par_ini = [0.05, 0.15, 0.3, 0.1, 0.6, 0.2, 0.85, 0.92, 0.2];    
            ll = haas_likelihood(par_ini, r, k, distri);
            for i in 1:1000                                 
                ω = rand(Uniform(0.01, 0.3), 2);
                α = rand(Uniform(0.05, 0.5), 2);
                β = [rand(Uniform(0.4, max(0.41, 1 - α[1])), 1); rand(Uniform(0.4, max(0.41, 1 - α[2])), 1)];
                p = rand(Uniform(0.8, 0.99), 2);
                η = rand(Uniform(0.01, 0.5), 1);
                par_random = [ω; α; β; p; η];
                if haas_likelihood(par_random, r, k, distri) < ll
                    ll = haas_likelihood(par_random, r, k, distri);
                    par_ini = par_random;
                end
            end
        end
        optimum = optimize(par -> hass_likelihood(par, r, k, distri), [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6 ,1e-6 ,1e-6, 1e-2], [Inf, Inf, 1, 1, 1, 1, 1, 1, 0.5], par_ini);
    end
    mle = optimum.minimizer;
    return mle;
end


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
                h[i, 1:k] .= ω .+ α .* r[i - 1]^2 + β .* h[i - 1, k + 1];
                Pt[i] = probability_regime_given_time_n(p, q, sqrt.(h[i - 1, :]), r[i - 1], Pt[i - 1]);
                h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
                log_lik[i - 1] = log(pdf(Normal(0, sqrt(h[i, 1])), r[i]) * Pt[i] + pdf(Normal(0, sqrt(h[i, 2])), r[i]) * (1 - Pt[i]));
            end
        elseif distri == "std"
            ν = par[4 * k + 1];
            @inbounds for i = 2:n
                h[i, 1:k] .= ω .+ α .* r[i - 1]^2 + β .* h[i - 1, k + 1];
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
                log_lik[i - 1] = log(1/ sqrt(h[i, 1]) * Tstudent(r[i] / sqrt(h[i, 1]), η)* Pt[i]  + 1 / sqrt(h[i, 2]) * Tstudent(r[i] / sqrt(h[i, 2]), η) * (1 - Pt[i]));
                
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
            i = 1;
            while i <= 1000              
                ω = rand(Uniform(0.0001, 0.4), k);
                α = rand(Uniform(0.01, 0.6), k);
                β = [rand(Uniform(0.1, 1 - α[1] - 1e-6), 1); rand(Uniform(0.1, 1 - α[2] - 1e-6), 1)];
                p = rand(Uniform(0.8, 0.99), k);
                par_random = [ω; α; β; p];
                @try begin
                    gray_likelihood(r, k, distri, par_random);
                @catch e->e isa ArgumentError
                    ll = gray_likelihood(r, k, distri, par_ini);
                @catch e->e isa DomainError
                    ll = gray_likelihood(r, k, distri, par_ini);
                @else 
                    if gray_likelihood(r, k, distri, par_random) < ll || isnan(ll)
                        ll = gray_likelihood(r, k, distri, par_random);
                        par_ini = par_random;
                    end
                    i = i + 1;
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
                @try begin
                    gray_likelihood(r, k, distri, par_random);
                @catch e->e isa ArgumentError
                    ll = gray_likelihood(r, k, distri, par_ini);
                @catch e->e isa DomainError
                    ll = gray_likelihood(r, k, distri, par_ini);
                @else 
                    if gray_likelihood(r, k, distri, par_random) < ll
                        ll = gray_likelihood(r, k, distri, par_random);
                        par_ini = par_random;
                    end
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
                @try begin
                    gray_likelihood(r, k, distri, par_random);
                @catch e->e isa ArgumentError
                    ll = gray_likelihood(r, k, distri, par_ini);
                @catch e->e isa DomainError
                    ll = gray_likelihood(r, k, distri, par_ini);
                @else 
                    if gray_likelihood(r, k, distri, par_random) < ll
                        ll = gray_likelihood(r, k, distri, par_random);
                        par_ini = par_random;
                    end
                end
            end
        end
        optimum = optimize(par -> gray_likelihood(r, k, distri, par), [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6 ,1e-6 ,1e-6, 0.01], [Inf, Inf, 1, 1, 1, 1, 1, 1, 0.49], par_ini);
    end
    mle = optimum.minimizer;
    return mle;
end
