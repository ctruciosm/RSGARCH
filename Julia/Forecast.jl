##################################################
###            RSGARCH: Forecasts             ####
##################################################



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
    return h[end,:], Pt[end];
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
    return h[end,:], Pt[end];

end


##################################################
### Forecast VaR and ES
##################################################
function var_es_rsgarch(α, p1, p2, σ₁, σ₂, distri, ν = nothing)
    if distri == "norm"
        opt = optimize(par -> min_var_n(α, p1, p2, σ₁, σ₂, par), [0.0], iterations = 10000).minimizer;
        f(x) = (x * p1 * pdf(Normal(0, σ₁), x) + x * p2 * pdf(Normal(0, σ₂), x))/α
        a = -Inf;
        b = -opt[1];
        ES, E = quadgk(f, a, b, rtol=1e-8);
    elseif distri == "student"
        opt = optimize(par -> min_var_t(α, p1, p2, σ₁, σ₂, ν, par), [0.0], iterations = 10000).minimizer;
        g(x) = (x * p1 * sqrt(ν/(ν-2))/σ₁ * pdf(TDist(ν), x*sqrt(ν/(ν-2))/σ₁) + x * p2 * + sqrt(ν/(ν-2))/σ₂ * pdf(TDist(ν), x*sqrt(ν/(ν-2))/σ₂))/α;
        a = -Inf;
        b = -opt[1];
        ES, E = quadgk(g, a, b, rtol=1e-8);
    else
        opt = NaN64
        ES = NaN64
        println("Error: Distribution should be Normal ('norm') or Student-T ('student').")
    end
    return [-opt; ES];
end

function var_es_rsgarch_mc(α, p1, p2, σ₁, σ₂, distri, ν = nothing)
    n = 1_000_000;
    n_1 = floor(Int, p1*n);
    n_2 = n - n_1;
    l = length(α);
    ES = Vector{Float64}(undef, l);
    if distri == "norm"
        x_sim = [rand(Normal(0, σ₁), n_1); rand(Normal(0, σ₂), n_2)];
        VaR = percentile(x_sim, α*100);
        [ES[i] = mean(x_sim[x_sim .< VaR[i]]); for i in 1:l];
    elseif distri == "student"
        x_sim = [rand(TDist(ν)*σ₁/sqrt(ν/(ν-2)), n_1); rand(TDist(ν)*σ₂/sqrt(ν/(ν-2)), n_2)]
        VaR = percentile(x_sim, α*100);
        [ES[i] = mean(x_sim[x_sim .< VaR[i]]); for i in 1:l];
    else
        VaR = NaN64;
        ES .= NaN64;
        println("Error: Distribution should be Normal ('norm') or Student-T ('student').")
    end
    return [VaR, ES];
end

function var_rsgarch_marcucci(α, p1, p2, σ₁, σ₂, distri)
    if distri == "norm"
        VaR = p1 * quantile(Normal(0, σ₁), α) + p2 * quantile(Normal(0, σ₂), α);
    elseif distri == "student"
        VaR = p1 * quantile(Normal(0, σ₁), α) + p2 * quantile(Normal(0, σ₂), α);
    else
        VaR = NaN64
        println("Error: Distribution should be Normal ('norm').")
    end
    return VaR;
end


