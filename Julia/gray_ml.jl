##################################################
###  RSGARCH Estim: Estimate RSGARCH Models   ####
##################################################

function gray_likelihood(r::Vector{Float64}, k::Int64, distri::String, par)

    # par(numeric vector): [ω, α, β, p11, p22, 1/gl]
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
        η = η = 1 / (2 + exp(-par[9]));
        @inbounds for i = 2:n
            h[i, 1:k] .= ω .+ α .* r[i - 1]^2 + β .* h[i - 1, k + 1];
            Pt[i] = probability_regime_given_time_it(p, q, sqrt.(h[i- 1, :]), r[i - 1], Pt[i - 1], η);
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
            log_lik[i - 1] = log(1/ sqrt(h[i, 1]) * Tstudent(r[i] / sqrt(h[i, 1]), η)* Pt[i]  + 1 / sqrt(h[i, 2]) * Tstudent(r[i] / sqrt(h[i, 2]), η) * (1 - Pt[i]));
        end
    end
    return -sum(log_lik)/2;
end

function fit_gray(r::Vector{Float64}, k::Int64, par_ini, distri::String)
    if distri == "norm"
        if isnothing(par_ini)
            if var(r) < 10
                par_ini = [5, 2.5, 0.3, 0.1, 0.6, 4, 4, 3];  
                optimum = optimize(par -> gray_likelihood(r, k, distri, par), par_ini);
            else
                par_ini = [-5, -2.5, 0.3, 0.1, 0.6, 4, 4, 3];  
                optimum = optimize(par -> gray_likelihood(r, k, distri, par), par_ini);
            end
            ll = gray_likelihood(r, k, distri, optimum.minimizer);
            par_ini = optimum.minimizer;

            i = 1;
            while i <= 5000
                tω = rand(Uniform(-7, 7), k);
                tα = rand(Uniform(-4, 4), k);
                tβ = rand(Uniform(-4, 4), k);
                tp = rand(Uniform(1, 5), k);
                par_random = [tω; tα; tβ; tp];
                gray_likelihood(r, k, distri, par_random)
                @try begin
                    gray_likelihood(r, k, distri, par_random);
                @catch e->e isa ArgumentError
                    ll = gray_likelihood(r, k, distri, par_ini);
                @catch e->e isa DomainError
                    ll = gray_likelihood(r, k, distri, par_ini);
                @else 
                    if gray_likelihood(r, k, distri, par_random) < ll || isnan(ll)
                        println(i)
                        @try begin
                            optimum = optimize(par -> gray_likelihood(r, k, distri, par), par_random);
                        @catch e->e isa ArgumentError
                            ll = gray_likelihood(r, k, distri, par_ini);
                        @catch e->e isa DomainError
                            ll = gray_likelihood(r, k, distri, par_ini);
                        @else 
                            ll = gray_likelihood(r, k, distri, optimum.minimizer);
                            par_ini = optimum.minimizer;
                        end
                    end
                    i = i + 1; 
                end
            end
            optimum = optimize(par -> gray_likelihood_transform(r, k, distri, par), par_random);
            if gray_likelihood(r, k, distri, optimum.minimizer) < ll
                par_ini = optimum.minimizer;
            end
        else
            optimum = optimize(par -> gray_likelihood_transform(r, k, distri, par), par_ini);
            par_ini = optimum.minimizer;
        end
        mle = param_transform(par_ini);
    elseif distri == "student"
        if isnothing(par_ini)
            if var(r) < 10
                par_ini = [5, 2.5, 0.3, 0.1, 0.6, 4, 4, 3, 1.5];
                optimum = optimize(par -> gray_likelihood(r, k, distri, par), par_ini);
            else
                par_ini = [-5, -2.5, 0.3, 0.1, 0.6, 4, 4, 3, 1.5];
                optimum = optimize(par -> gray_likelihood(r, k, distri, par), par_ini);
            end
            ll = gray_likelihood(r, k, distri, optimum.minimizer);
            par_ini = optimum.minimizer;

            i = 1;
            while i<= 5000                               
                tω = rand(Uniform(-1, 7), k);
                tα = rand(Uniform(-4, 4), k);
                tβ = rand(Uniform(-4, 4), k);
                tp = rand(Uniform(1, 5), k);
                η = rand(Uniform(0.01, 0.49), 1);
                par_random = [gray_transform([tω; tα; tβ; tp]); η];
                @try begin
                    gray_likelihood(r, k, distri, par_random);
                @catch e->e isa ArgumentError
                    ll = gray_likelihood(r, k, distri, par_ini);
                @catch e->e isa DomainError
                    ll = gray_likelihood(r, k, distri, par_ini);
                @else 
                    if gray_likelihood(r, k, distri, par_random) < ll || isnan(ll)
                        @try begin
                            optimum = optimize(par -> gray_likelihood(r, k, distri, par), par_random);
                        @catch e->e isa ArgumentError
                            ll = gray_likelihood(r, k, distri, par_ini);
                        @catch e->e isa DomainError
                            ll = gray_likelihood(r, k, distri, par_ini);
                        @else 
                            ll = gray_likelihood(r, k, distri, optimum.minimizer);
                            par_ini = optimum.minimizer;
                        end
                    end
                    i = i + 1;
                end 
            end
        else
            optimum = optimize(par -> gray_likelihood_transform(r, k, distri, par), par_ini);
            par_ini = optimum.minimizer;
        end
        mle = [par_transform(par_ini[1:8]); 2 + exp(-par_ini[9])];
    else
        println("Only Normal ('norm') and Student-T ('student') distributions are available")
    end
    return mle;
end

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
        η = par[4 * k + 1];
        @inbounds for i = 2:n+1
            h[i, 1:k] = ω .+ α .* r[i - 1]^2 + β .* h[i - 1, k + 1];
            Pt[i] = probability_regime_given_time_it(p, q, sqrt.(h[i- 1, :]), r[i - 1], Pt[i - 1], η);
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];     
            s[i] = wsample([1, 2], P[:, s[i-1]])[1]; 
        end
    end
    return h[end,:];
end



