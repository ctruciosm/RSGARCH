#################################################
###  RSGARCH GPD: Simulate RSGARCH Models    ####
#################################################


#################################################
function simulate_gray(n, distri, ω, α, β, P, burnin)
    # P = [p11, 1 - p22; 1 - p11, p22]
    ntot = n + burnin + 1;
    k = length(ω);
    h = Matrix{Float64}(undef, ntot, k + 1);
    r = Vector{Float64}(undef, ntot);
    s = Vector{Int32}(undef, ntot);
    Pt = Vector{Float64}(undef, ntot);

    e = ifelse(distri == "norm", rand(Normal(), ntot), sqrt(5/7).* rand(TDist(7), ntot));
    h[1, 1:k] .= 1.0;
    p = P[1, 1];
    q = P[2, 2];
    Pt[1] = (1 - q) / (2 - p - q);  
    h[1, k + 1] = Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2];
    s[1] = wsample([1, 2], [Pt[1], 1 - Pt[1]])[1];
    r[1] = e[1] * sqrt(h[1, s[1]]);
    if distri == "norm"
        for i = 2:ntot
            h[i, 1:k] = ω .+ α.* r[i - 1]^2 + β.* h[i - 1, k + 1];
            Pt[i] = probability_regime_given_time_n(p, q, sqrt.(h[i - 1, :]), r[i - 1], Pt[i - 1]);
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
            s[i] = wsample([1, 2], P[:, s[i-1]])[1]; 
            r[i] = e[i] * sqrt(h[i, s[i]]);
        end
    else
        for i = 2:ntot
            h[i, 1:k] = ω .+ α.* r[i - 1]^2 + β.* h[i - 1, k + 1];
            Pt[i] = probability_regime_given_time_t(p, q, sqrt.(h[i - 1, :]), r[i- 1], Pt[i - 1], 7);
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
            s[i] = wsample([1, 2], P[:, s[i-1]])[1];
            r[i] = e[i] * sqrt(h[i, s[i]]);
        end
    end
    return r[burnin + 1: end - 1], h[burnin + 1: end, :], Pt[burnin + 1: end], s[burnin + 1: end];
end
#################################################
function simulate_haas(n, distri, ω, α, β, P, burnin)
    # P = [p11, 1 - p22; 1 - p11, p22]
    ntot = n + burnin + 1;
    s = Vector{Int32}(undef, ntot);
    k = length(ω);
    h = Matrix{Float64}(undef, ntot, k + 1);
    r = Vector{Float64}(undef, ntot);
    Pt = Vector{Float64}(undef, ntot);

    e = ifelse(distri == "norm", rand(Normal(), ntot), sqrt(5/7).* rand(TDist(7), ntot));
    p = P[1, 1];
    q = P[2, 2];
    h[1, 1:k] .= ω ./ (1 .- (α .+ β));
    s[1] =  1;                                  
    r[1] = e[1] * sqrt(h[1, s[1]]);
    if distri == "norm"
        for i = 2:ntot
            h[i, 1:k] .= ω .+ α.* r[i - 1]^2 + β.* h[i - 1, 1:k];
            Pt[i] = probability_regime_given_time_n(p, q, sqrt.(h[i - 1, 1:k]), r[i - 1], Pt[i - 1]);
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
            s[i] = wsample([1, 2], P[:, s[i-1]])[1];
            r[i] = e[i] * sqrt(h[i, s[i]]);
        end
    else
        for i = 2:ntot
            h[i, 1:k] = ω .+ α.* r[i - 1]^2 + β.* h[i - 1, 1:k];
            Pt[i] = probability_regime_given_time_t(p, q, sqrt.(h[i - 1, 1:k]), r[i- 1], Pt[i - 1], 7);
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
            s[i] = wsample([1, 2], P[:, s[i-1]])[1];
            r[i] = e[i] * sqrt(h[i, s[i]]);
        end
    end
    return r[burnin + 1: end - 1], h[burnin + 1: end, :], Pt[burnin + 1: end], s[burnin + 1: end];
end
#################################################
function simulate_garch(n, ω, α, β, burnin)
    ntot = n + burnin + 1;
    h = Vector{Float64}(undef, ntot);
    r = Vector{Float64}(undef, ntot);
    e = rand(Normal(), ntot);
    h[1] = ω / (1 - α - β);
    for i = 2:ntot
        h[i] = ω + α* r[i - 1]^2 + β* h[i - 1];
        r[i] = e[i] * sqrt(h[i]);
    end
    return r[burnin + 1: end - 1]
end