##################################################
###  RSGARCH: Maximum Likelihood Estimation   ####
##################################################


##################################################
### Likelihoods: par .= [ω, α, β, p11, p22, 1/gl]
##################################################
function gray_likelihood(r::Vector{Float64}, k::Int64, distri::String, par)
    n = length(r);
    h = Matrix{Float64}(undef, n, k + 1);
    Pt = Vector{Float64}(undef, n);
    log_lik = Vector{Float64}(undef, n - 1);
    # Transformations
    ω = exp.(-par[1:2]);
    α = exp.(-par[3:4]) ./ (1 .+ exp.(-par[3:4]) .+ exp.(-par[5:6]));
    β = exp.(-par[5:6]) ./ (1 .+ exp.(-par[3:4]) .+ exp.(-par[5:6]));
    p = 1 ./(1 .+ exp(-par[7]));
    q = 1 ./(1 .+ exp(-par[8]));
    # Likelihood
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
    else
        η = 1 / (2 + exp(-par[9]));
        @inbounds for i = 2:n
            h[i, 1:k] .= ω .+ α .* r[i - 1]^2 + β .* h[i - 1, k + 1];
            Pt[i] = probability_regime_given_time_it(p, q, sqrt.(h[i- 1, :]), r[i - 1], Pt[i - 1], η);
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
            log_lik[i - 1] = log(1/ sqrt(h[i, 1]) * Tstudent(r[i] / sqrt(h[i, 1]), η)* Pt[i]  + 1 / sqrt(h[i, 2]) * Tstudent(r[i] / sqrt(h[i, 2]), η) * (1 - Pt[i]));
        end
    end
    return -sum(log_lik)/2;
end

function gray_likelihood2(r::Vector{Float64}, k::Int64, distri::String, par)
    n = length(r);
    h = Matrix{Float64}(undef, n, k + 1);
    Pt = Vector{Float64}(undef, n);
    log_lik = Vector{Float64}(undef, n - 1);
    # Transformations
    ω = par[1:2];
    α = par[3:4];
    β = par[5:6];
    p = par[7];
    q = par[8];
    # Likelihood
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
    else
        η = 1 / (2 + exp(-par[9]));
        @inbounds for i = 2:n
            h[i, 1:k] .= ω .+ α .* r[i - 1]^2 + β .* h[i - 1, k + 1];
            Pt[i] = probability_regime_given_time_it(p, q, sqrt.(h[i- 1, :]), r[i - 1], Pt[i - 1], η);
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
            log_lik[i - 1] = log(1/ sqrt(h[i, 1]) * Tstudent(r[i] / sqrt(h[i, 1]), η)* Pt[i]  + 1 / sqrt(h[i, 2]) * Tstudent(r[i] / sqrt(h[i, 2]), η) * (1 - Pt[i]));
        end
    end
    return -sum(log_lik)/2;
end
##################################################
function haas_likelihood(r::Vector{Float64}, k::Int64, distri::String, par)
    n = length(r);
    h = Matrix{Float64}(undef, n, k);
    Pt = Vector{Float64}(undef, n);
    log_lik = Vector{Float64}(undef, n - 1);
    # Transformations
    ω = exp.(-par[1:2]);
    α = exp.(-par[3:4]) ./ (1 .+ exp.(-par[3:4]) .+ exp.(-par[5:6]));
    β = exp.(-par[5:6]) ./ (1 .+ exp.(-par[3:4]) .+ exp.(-par[5:6]));
    p = 1 ./(1 .+ exp(-par[7]));
    q = 1 ./(1 .+ exp(-par[8]));
    # Likelihood
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

##################################################
### Fit models
##################################################
function fit_gray(r::Vector{Float64}, k::Int64, par_ini, distri::String)
    sd = std(r);
    r = r/sd;
    ll = Vector{Float64}(undef, 5000);
    if distri == "norm"
        if isnothing(par_ini)
            par_ini = Matrix{Float64}(undef, 5000, 8);
            @try begin
                opt = optimize(par -> gray_likelihood(r, k, distri, par), [5, 2.5, 0.3, 0.1, 0.6, 4, 4, 3], iterations = 10000).minimizer;
            @catch e->e isa ArgumentError
                opt = optimize(par -> gray_likelihood(r, k, distri, par), [-5, -2.5, 0.3, 0.1, 0.6, 4, 4, 3], iterations = 10000).minimizer;
            @catch e->e isa DomainError
                opt = optimize(par -> gray_likelihood(r, k, distri, par), [-5, -2.5, 0.3, 0.1, 0.6, 4, 4, 3], iterations = 10000).minimizer;
            end
            for i in 1:5000
                Random.seed!(i);
                ω = rand(Uniform(0, 2), k);
                α = rand(Uniform(0, 1), k);
                β = [rand(Uniform(0, 1 - α[1])), rand(Uniform(0, 1 - α[2]))];
                a = (α .+ β) ./ (1 .- α .- β);
                tω = -log.(ω);
                tα = -log.((1 .+ a) .* α);
                tβ = -log.((1 .+ a) .* β);
                tp = rand(Uniform(1, 5), k);
                par_random = [tω; tα; tβ; tp];
                @try begin
                    gray_likelihood(r, k, distri, par_random);
                @catch e->e isa ArgumentError
                    par_random = [-tω; tα; tβ; tp];
                    ll[i] = gray_likelihood(r, k, distri, par_random);
                    par_ini[i, :] .= par_random;
                @catch e->e isa DomainError
                    par_random = [-tω; tα; tβ; tp];
                    ll[i] = gray_likelihood(r, k, distri, par_random);
                    par_ini[i, :] .= par_random;
                @else 
                    ll[i] = gray_likelihood(r, k, distri, par_random);
                    par_ini[i, :] .= par_random;
                end
            end
            par_ini = par_ini[sortperm(ll),:];
            j = 1;
            l = 1;
            while l < 4
                @try begin
                    aux = optimize(par -> gray_likelihood(r, k, distri, par), par_ini[j,:]).minimizer;
                    l = l + 1;
                    if gray_likelihood(r, k, distri, aux) < gray_likelihood(r, k, distri, opt)
                        opt = aux;
                    end
                @catch e->e isa ArgumentError
                    l = l;
                @catch e->e isa DomainError
                    l = l;
                end
                j = j + 1;
            end
            mle = param_transform(opt);
            mle[1:2] .= sd^2 .* mle[1:2];
        else
            opt = optimize(par -> gray_likelihood(r, k, distri, par), par_ini, iterations = 10000).minimizer;
            mle = param_transform(opt);
            mle[1:2] .= sd^2 .* mle[1:2];
        end
    elseif distri == "student"
        if isnothing(par_ini)
            par_ini = Matrix{Float64}(undef, 5000, 9);
            @try begin
                opt = optimize(par -> gray_likelihood(r, k, distri, par), [5, 2.5, 0.3, 0.1, 0.6, 4, 4, 3, 1.5], iterations = 10000).minimizer;
            @catch e->e isa ArgumentError
                opt = optimize(par -> gray_likelihood(r, k, distri, par), [-5, -2.5, 0.3, 0.1, 0.6, 4, 4, 3, 1.5], iterations = 10000).minimizer;
            @catch e->e isa DomainError
                opt = optimize(par -> gray_likelihood(r, k, distri, par), [-5, -2.5, 0.3, 0.1, 0.6, 4, 4, 3, 1.5], iterations = 10000).minimizer;
            end
            for i in 1:5000
                Random.seed!(i);
                ω = rand(Uniform(0, 2), k);
                α = rand(Uniform(0, 1), k);
                β = [rand(Uniform(0, 1 - α[1])), rand(Uniform(0, 1 - α[2]))];
                a = (α .+ β) ./ (1 .- α .- β);
                tω = -log.(ω);
                tα = -log.((1 .+ a) .* α);
                tβ = -log.((1 .+ a) .* β);
                tp = rand(Uniform(1, 5), k);
                tν = rand(Uniform(-5, 3), 1);
                par_random = [tω; tα; tβ; tp; tν];
                @try begin
                    gray_likelihood(r, k, distri, par_random);
                @catch e->e isa ArgumentError
                    par_random = [-tω; tα; tβ; tp; tν];
                    ll[i] = gray_likelihood(r, k, distri, par_random);
                    par_ini[i, :] .= par_random;
                @catch e->e isa DomainError
                    par_random = [-tω; tα; tβ; tp; tν];
                    ll[i] = gray_likelihood(r, k, distri, par_random);
                    par_ini[i, :] .= par_random;
                @else 
                    ll[i] = gray_likelihood(r, k, distri, par_random);
                    par_ini[i, :] .= par_random;
                end
            end
            par_ini = par_ini[sortperm(ll),:];
            j = 1;
            l = 1;
            while l < 4
                @try begin
                    aux = optimize(par -> gray_likelihood(r, k, distri, par), par_ini[j,:]).minimizer;
                    l = l + 1;
                    if gray_likelihood(r, k, distri, aux) < gray_likelihood(r, k, distri, opt)
                        opt = aux;
                    end
                @catch e->e isa ArgumentError
                    l = l;
                @catch e->e isa DomainError
                    l = l;
                end
                j = j + 1;
            end
            mle = [param_transform(opt[1:8]); 2 + exp(-opt[9])];
            mle[1:2] .= sd^2 .* mle[1:2];
        else
            opt = optimize(par -> gray_likelihood(r, k, distri, par), par_ini, iterations = 10000).minimizer;
            mle = [param_transform(opt[1:8]); 2 + exp(-opt[9])];
            mle[1:2] .= sd^2 .* mle[1:2];
        end
    else
        mle = NaN64;
        println("Only Normal ('norm') and Student-T ('student') distributions are available")
    end
    return mle;
end
##################################################
function fit_haas(r::Vector{Float64}, k::Int64, par_ini, distri::String)
    sd = std(r);
    r = r/sd;
    ll = Vector{Float64}(undef, 5000);
    if distri == "norm"
        if isnothing(par_ini)
            par_ini = Matrix{Float64}(undef, 5000, 8);
            @try begin
                opt = optimize(par -> haas_likelihood(r, k, distri, par), [5, 2.5, 0.3, 0.1, 0.6, 4, 4, 3]).minimizer;
            @catch e->e isa ArgumentError
                opt = optimize(par -> haas_likelihood(r, k, distri, par), [-5, -2.5, 0.3, 0.1, 0.6, 4, 4, 3]).minimizer;
            @catch e->e isa DomainError
                opt = optimize(par -> haas_likelihood(r, k, distri, par), [-5, -2.5, 0.3, 0.1, 0.6, 4, 4, 3]).minimizer;
            end
            for i in 1:5000
                Random.seed!(i);
                ω = rand(Uniform(0, 2), k);
                α = rand(Uniform(0, 1), k);
                β = [rand(Uniform(0, 1 - α[1])), rand(Uniform(0, 1 - α[2]))];
                a = (α .+ β) ./ (1 .- α .- β);
                tω = -log.(ω);
                tα = -log.((1 .+ a) .* α);
                tβ = -log.((1 .+ a) .* β);
                tp = rand(Uniform(1, 5), k);
                par_random = [tω; tα; tβ; tp];
                @try begin
                    haas_likelihood(r, k, distri, par_random);
                @catch e->e isa ArgumentError
                    par_random = [-tω; tα; tβ; tp];
                    ll[i] = haas_likelihood(r, k, distri, par_random);
                    par_ini[i, :] .= par_random;
                @catch e->e isa DomainError
                    par_random = [-tω; tα; tβ; tp];
                    ll[i] = haas_likelihood(r, k, distri, par_random);
                    par_ini[i, :] .= par_random;
                @else 
                    ll[i] = haas_likelihood(r, k, distri, par_random);
                    par_ini[i, :] .= par_random;
                end
            end
            par_ini = par_ini[sortperm(ll),:];
            j = 1;
            l = 1;
            while l < 4
                @try begin
                    aux = optimize(par -> haas_likelihood(r, k, distri, par), par_ini[j,:]).minimizer;
                    l = l + 1;
                    if haas_likelihood(r, k, distri, aux) < haas_likelihood(r, k, distri, opt)
                        opt = aux;
                    end
                @catch e->e isa ArgumentError
                    l = l;
                @catch e->e isa DomainError
                    l = l;
                end
                j = j + 1;
            end
            mle = param_transform(opt);
            mle[1:2] .= sd^2 .* mle[1:2];
        else
            opt = optimize(par -> haas_likelihood(r, k, distri, par), par_ini).minimizer;
            mle = param_transform(opt);
            mle[1:2] .= sd^2 .* mle[1:2];
        end
    elseif distri == "student"
        if isnothing(par_ini)
            par_ini = Matrix{Float64}(undef, 5000, 9);
            @try begin
                opt = optimize(par -> haas_likelihood(r, k, distri, par), [5, 2.5, 0.3, 0.1, 0.6, 4, 4, 3, 1.5]).minimizer;
            @catch e->e isa ArgumentError
                opt = optimize(par -> haas_likelihood(r, k, distri, par), [-5, -2.5, 0.3, 0.1, 0.6, 4, 4, 3, 1.5]).minimizer;
            @catch e->e isa DomainError
                opt = optimize(par -> haas_likelihood(r, k, distri, par), [-5, -2.5, 0.3, 0.1, 0.6, 4, 4, 3, 1.5]).minimizer;
            end
            for i in 1:5000
                Random.seed!(i);
                ω = rand(Uniform(0, 2), k);
                α = rand(Uniform(0, 1), k);
                β = [rand(Uniform(0, 1 - α[1])), rand(Uniform(0, 1 - α[2]))];
                a = (α .+ β) ./ (1 .- α .- β);
                tω = -log.(ω);
                tα = -log.((1 .+ a) .* α);
                tβ = -log.((1 .+ a) .* β);
                tp = rand(Uniform(1, 5), k);
                tν = rand(Uniform(-5, 3), 1);
                par_random = [tω; tα; tβ; tp; tν];
                @try begin
                    haas_likelihood(r, k, distri, par_random);
                @catch e->e isa ArgumentError
                    par_random = [-tω; tα; tβ; tp; tν];
                    ll[i] = haas_likelihood(r, k, distri, par_random);
                    par_ini[i, :] .= par_random;
                @catch e->e isa DomainError
                    par_random = [-tω; tα; tβ; tp; tν];
                    ll[i] = haas_likelihood(r, k, distri, par_random);
                    par_ini[i, :] .= par_random;
                @else 
                    ll[i] = haas_likelihood(r, k, distri, par_random);
                    par_ini[i, :] .= par_random;
                end
            end
            par_ini = par_ini[sortperm(ll),:];
            j = 1;
            l = 1;
            while l < 4
                @try begin
                    aux = optimize(par -> haas_likelihood(r, k, distri, par), par_ini[j,:]).minimizer;
                    l = l + 1;
                    if haas_likelihood(r, k, distri, aux) < haas_likelihood(r, k, distri, opt)
                        opt = aux;
                    end
                @catch e->e isa ArgumentError
                    l = l;
                @catch e->e isa DomainError
                    l = l;
                end
                j = j + 1;
            end
            mle = [param_transform(opt[1:8]); 2 + exp(-opt[9])];
            mle[1:2] .= sd^2 .* mle[1:2];
        else
            opt = optimize(par -> haas_likelihood(r, k, distri, par), par_ini).minimizer;
            mle = [param_transform(opt[1:8]); 2 + exp(-opt[9])];
            mle[1:2] .= sd^2 .* mle[1:2];
        end
    else
        mle = NaN64;
        println("Only Normal ('norm') and Student-T ('student') distributions are available")
    end
    return mle;
end
##################################################
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

##################################################
### Forecast h
##################################################
function fore_gray(r::Vector{Float64}, k::Int64, par, distri::String)
    # Regime 1: Low Vol
    # Regime 2: High Vol
    n = length(r);
    h = Matrix{Float64}(undef, n + 1, k + 1);
    s = Vector{Int32}(undef, n + 1);
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
    if distri == "norm"
        @inbounds for i = 2:n+1
            h[i, 1:k] = ω .+ α .* r[i - 1]^2 + β .* h[i - 1, k + 1];
            Pt[i] = probability_regime_given_time_n(p, q, sqrt.(h[i - 1, :]), r[i - 1], Pt[i - 1]);
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
            s[i] = wsample([1, 2], P[:, s[i-1]])[1]; 
        end
    else
        ν = par[4 * k + 1];
        @inbounds for i = 2:n+1
            h[i, 1:k] = ω .+ α .* r[i - 1]^2 + β .* h[i - 1, k + 1];
            Pt[i] = probability_regime_given_time_t(p, q, sqrt.(h[i- 1, :]), r[i - 1], Pt[i - 1], ν);
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];     
            s[i] = wsample([1, 2], P[:, s[i-1]])[1]; 
        end
    end
    return h[end,:];
end
##################################################
function fore_haas(r::Vector{Float64}, k::Int64, par, distri::String)
    # Regime 1: Low Vol
    # Regime 2: High Vol
    n = length(r);
    h = Matrix{Float64}(undef, n + 1, k + 1);
    s = Vector{Int32}(undef, n + 1);
    Pt = Vector{Float64}(undef, n + 1);

    ω = par[1:k];
    α = par[k + 1 : 2 * k];
    β = par[2 * k + 1 : 3 * k];
    p = par[3 * k + 1];
    q = par[4 * k];

    Pt[1] = (1 - q) / (2 - p - q);              
    h[1, 1:k] .= ω ./ (1 .- (α .+ β));                        
    h[1, k + 1] = Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2];
    s[1] = wsample([1, 2], [Pt[1], 1 - Pt[1]])[1];
    if distri == "norm"
        @inbounds for i = 2:n+1
            h[i, 1:k] = ω .+ α .* r[i - 1]^2 + β .* h[i - 1, 1:k];
            Pt[i] = probability_regime_given_time_n(p, q, sqrt.(h[i - 1, 1:k]), r[i - 1], Pt[i - 1]);
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
            s[i] = wsample([1, 2], P[:, s[i-1]])[1]; 
        end
    else
        ν = par[4 * k + 1];
        @inbounds for i = 2:n+1
            h[i, 1:k] = ω .+ α .* r[i - 1]^2 + β .* h[i - 1, 1:k];
            Pt[i] = probability_regime_given_time_t(p, q, sqrt.(h[i- 1, 1:k]), r[i - 1], Pt[i - 1], ν);
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];     
            s[i] = wsample([1, 2], P[:, s[i-1]])[1]; 
        end
    end
    return h[end,:];

end


