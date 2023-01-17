##################################################
###  RSGARCH Estim: Estimate RSGARCH Models   ####
##################################################

function garch_likelihood(r::Vector{Float64}, par)
    n = length(r);
    h = Vector{Float64}(undef, n);
    log_lik = Vector{Float64}(undef, n - 1);
    ω = exp(-par[1]);
    α = exp(-par[2]) / (1 + exp(-par[2]) + exp(-par[3]));
    β = exp(-par[3]) / (1 + exp(-par[2]) + exp(-par[3]));
    h[1] = ω /(1 - α - β);
    @inbounds for i = 2:n
        h[i] = ω + α * r[i - 1]^2 + β * h[i - 1];
        log_lik[i - 1] = log(pdf(Normal(0, sqrt(h[i])), r[i]));
    end
    return -sum(log_lik)/2;
end

function fit_garch(r::Vector{Float64})
    par_ini = [5, 0.3, 0.6];  
    ll = garch_likelihood(r, par_ini);
    i = 1
    while i <= 1000
        tω = rand(Uniform(-7, 7), 1);
        tα = rand(Uniform(-4, 4), 1);
        tβ = rand(Uniform(-4, 4), 1);
        par_random = [tω; tα; tβ];
        if garch_likelihood(r, par_random) < ll
            ll = garch_likelihood(r, par_random);
            par_ini = par_random;
        end
        i = i + 1;
    end
    optimum = optimize(par -> garch_likelihood(r, par), par_ini);
    mle = garch_transform(optimum.minimizer);
    return mle;
end

function haas_likelihood(r::Vector{Float64}, k::Int64, distri::String, par)
    # par = numeric vector: omega, alpha, beta, p11, p22
    n = length(r);
    h = Matrix{Float64}(undef, n, k);
    Pt = Vector{Float64}(undef, n);
    log_lik = Vector{Float64}(undef, n - 1);

    ω = exp.(-par[1:2]);
    α = exp.(-par[3:4]) ./ (1 .+ exp.(-par[3:4]) .+ exp.(-par[5:6]));
    β = exp.(-par[5:6]) ./ (1 .+ exp.(-par[3:4]) .+ exp.(-par[5:6]));
    p = 1 / (1 + exp(-par[7]));
    q = 1 / (1 + exp(-par[8]));

    Pt[1] = (1 - q) / (2 - p - q);             
    h[1, :] .= ω ./ (1 .- (α + β));   
    if (distri == "norm")
        @inbounds for i = 2:n
            h[i, :] .= ω .+ α .* r[i - 1]^2 + β .* h[i - 1, :];
            Pt[i] = probability_regime_given_time_n(p, q, sqrt.(h[i - 1, :]), r[i - 1], Pt[i - 1]);
            log_lik[i - 1] = log(pdf(Normal(0, sqrt(h[i, 1])), r[i]) * Pt[i] + pdf(Normal(0, sqrt(h[i, 2])), r[i]) * (1 - Pt[i]));
        end
    else
        η = 1 / (2 + exp(-par[9]));
        @inbounds for i = 2:n
            h[i, :] .= ω .+ α .* r[i - 1]^2 + β .* h[i - 1, :];
            Pt[i] = probability_regime_given_time_it(p, q, sqrt.(h[i- 1, :]), r[i - 1], Pt[i - 1], η);
            log_lik[i - 1] = log(1/ sqrt(h[i, 1]) * Tstudent(r[i] / sqrt(h[i, 1]), η)* Pt[i]  + 1 / sqrt(h[i, 2]) * Tstudent(r[i] / sqrt(h[i, 2]), η) * (1 - Pt[i])); 
        end
    end
    return -sum(log_lik)/2;
end

function fit_haas(r::Vector{Float64}, k::Int64, par_ini, distri::String)
    ll = Vector{Float64}(undef, 5001);
    if distri == "norm"
        if isnothing(par_ini)
            par_ini = Matrix{Float64}(undef, 5001, 8);
            par_ini[1, :] = [5, 2.5, 0.3, 0.1, 0.6, 4, 4, 3];  
            ll[1] = haas_likelihood(r, k, distri, par_ini[1, :]);
            for i in 2:5001
                tα = rand(Uniform(-4, 4), k);
                tω = rand(Uniform(-7, 7), k);
                tp = rand(Uniform(1, 5), k);
                tβ = rand(Uniform(-4, 4), k);
                par_random = [tω; tα; tβ; tp];
                @try begin
                    haas_likelihood(r, k, distri, par_random);
                @catch e->e isa ArgumentError
                    ll[i] = NaN64;
                @catch e->e isa DomainError
                    ll[i] = NaN64;
                @else 
                    ll[i] = haas_likelihood(r, k, distri, par_random);
                    par_ini[i, :] .= par_random;
                end
            end
            par_ini = par_ini[sortperm(ll),:];
            opt = optimize(par -> haas_likelihood(r, k, distri, par), par_ini[1, :]).minimizer;
            aux = optimize(par -> haas_likelihood(r, k, distri, par), par_ini[2, :]).minimizer;
            if haas_likelihood(r, k, distri, aux) < haas_likelihood(r, k, distri, opt)
                opt = aux;
            end
            aux = optimize(par -> haas_likelihood(r, k, distri, par), par_ini[3, :]).minimizer;
            if haas_likelihood(r, k, distri, aux) < haas_likelihood(r, k, distri, opt)
                opt = aux;
            end
            mle = param_transform(opt);
        else
            opt = optimize(par -> haas_likelihood(r, k, distri, par), par_ini).minimizer;
            mle = param_transform(opt);
        end
    elseif distri == "student"
        if isnothing(par_ini)
            par_ini = Matrix{Float64}(undef, 5001, 9);
            par_ini[1, :] = [5, 2.5, 0.3, 0.1, 0.6, 4, 4, 3, 1.5];  
            ll[1] = haas_likelihood(r, k, distri, par_ini[1, :]);
            for i in 2:5001
                tα = rand(Uniform(-4, 4), k);
                tω = rand(Uniform(-7, 7), k);
                tp = rand(Uniform(1, 5), k);
                tβ = rand(Uniform(-4, 4), k);
                tν = rand(Uniform(-5, 3), 1);
                par_random = [tω; tα; tβ; tp; tν];
                @try begin
                    haas_likelihood(r, k, distri, par_random);
                @catch e->e isa ArgumentError
                    ll[i] = NaN64;
                @catch e->e isa DomainError
                    ll[i] = NaN64;
                @else 
                    ll[i] = haas_likelihood(r, k, distri, par_random);
                    par_ini[i, :] .= par_random;
                end
            end
            par_ini = par_ini[sortperm(ll),:];
            opt = optimize(par -> haas_likelihood(r, k, distri, par), par_ini[1, :]).minimizer;
            aux = optimize(par -> haas_likelihood(r, k, distri, par), par_ini[2, :]).minimizer;
            if haas_likelihood(r, k, distri, aux) < haas_likelihood(r, k, distri, opt)
                opt = aux;
            end
            aux = optimize(par -> haas_likelihood(r, k, distri, par), par_ini[3, :]).minimizer;
            if haas_likelihood(r, k, distri, aux) < haas_likelihood(r, k, distri, opt)
                opt = aux;
            end
            mle = [param_transform(opt[1:8]); 2 + exp(-opt[9])];
        else
            opt = optimize(par -> haas_likelihood(r, k, distri, par), par_ini).minimizer;
            mle = [param_transform(opt[1:8]); 2 + exp(-opt[9])];
        end
    else
        mle = NaN64;
        println("Only Normal ('norm') and Student-T ('student') distributions are available")
    end
    return mle;
end


