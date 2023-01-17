##################################################
###  RSGARCH Estim: Estimate RSGARCH Models   ####
##################################################
 
function gray_likelihood_jump(r::Vector{Float64}, k::Int64, par...)
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

    Pt[1] = (1 - q) / (2 - p - q);              
    h[1, 1:k] .= var(r);                        
    h[1, k + 1] = Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2];
    @inbounds for i = 2:n
        h[i, 1] = ω[1] + α[1]*r[i - 1]^2 + β[1]*h[i - 1, k + 1];
        h[i, 2] = ω[2] + α[2]*r[i - 1]^2 + β[2]*h[i - 1, k + 1];
        Pt[i] = probability_regime_given_time_n(p, q, sqrt.(h[i - 1, :]), r[i - 1], Pt[i - 1]);
        h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
        log_lik[i - 1] = log(pdf(Normal(0, sqrt(h[i, 1])), r[i]) * Pt[i] + pdf(Normal(0, sqrt(h[i, 2])), r[i]) * (1 - Pt[i]));
    end
    return -sum(log_lik)/2;
end


function fit_gray_jump(r::Vector{Float64}, k::Int64, par_ini, distri::String)
    if distri == "norm"
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

        par_ini = [0.05, 0.15, 0.3, 0.1, 0.6, 0.2, 0.85, 0.92];  
        make_closures(r, k) = par -> gray_likelihood_jump(r, k, par...)
        ll = make_closures(r, k)
        ll(par_ini)

        model = Model(Ipopt.Optimizer);
        @variable(model, par[1:8] .>= 1e-6);
        register(model, :gray_likelihood_jump, 8, gray_likelihood_jump; autodiff = true);
        @constraint(model, c1, par[3] + par[5] <= 0.999999);
        @constraint(model, c2, par[4] + par[6] <= 0.999999);
        @constraint(model, c3, par[7:8] .<= 0.999999);
        @NLobjective(model, Min, gray_likelihood_jump(par...));
        optimize!(model)
    
    



        optimum = optimize(par -> gray_likelihood(r, k, distri, par), [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6 ,1e-6 ,1e-6], [Inf, Inf, 1, 1, 1, 1, 1, 1], par_ini);
    else
        if isnothing(par_ini)
            par_ini = [0.05, 0.15, 0.3, 0.1, 0.6, 0.2, 0.85, 0.92, 0.2];    
            ll = gray_likelihood(r, k, distri, par_ini);
            for i in 1:1000                                 
                ω = rand(Uniform(0.0001, 0.4), k);
                α = rand(Uniform(0.01, 0.6), k);
                β = [rand(Uniform(0.1, 1 - α[1] - 1e-6), 1); rand(Uniform(0.1, 1 - α[2] - 1e-6), 1)];
                p = rand(Uniform(0.8, 0.99), k);
                η = rand(Uniform(0.01, 0.5), 1);
                par_random = [ω; α; β; p; η];
                if gray_likelihood(r, k, distri, par_random) < ll
                    ll = gray_likelihood(r, k, distri, par_random);
                    par_ini = par_random;
                end
            end
        end
        optimum = optimize(par -> gray_likelihood(r, k, distri, par), [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6 ,1e-6 ,1e-6, 1e-2], [Inf, Inf, 1, 1, 1, 1, 1, 1, 0.5], par_ini);
    end
    mle = optimum.minimizer;
    return mle;
end




