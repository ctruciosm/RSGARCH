#################################################
###  RSGARCH GPD: Simulate RSGARCH Models    ####
#################################################
using Random, Distributions

function hamilton_filter_n(p, q, σ, r, Pt)
    numA = (1 - q) * pdf(Normal(0, σ[2]), r) * (1 - Pt);
    numB = p * pdf(Normal(0, σ[1]), r) * Pt;
    deno = pdf(Normal(0, σ[1]), r) * Pt + pdf(Normal(0, σ[2]), r) * (1 - Pt);
    return numA/deno + numB/deno;
end

function hamilton_filter_t(p, q, σ, r, Pt, ν)
    numA = (1 - q) * sqrt(ν/(ν-2))/σ[2] * pdf(TDist(ν), r*sqrt(ν/(ν-2))/σ[2]) * (1 - Pt);
    numB = p * sqrt(ν/(ν-2))/σ[1] * pdf(TDist(ν), r*sqrt(ν/(ν-2))/σ[1]) * Pt;
    deno = sqrt(ν/(ν-2))/σ[1] * pdf(TDist(ν), r*sqrt(ν/(ν-2))/σ[1]) * Pt + sqrt(ν/(ν-2))/σ[2] * pdf(TDist(ν), r*sqrt(ν/(ν-2))/σ[2]) * (1 - Pt);
    return numA/deno + numB/deno;
end


function simulate_gray(n, distri, ω, α, β, P, burnin)
    # P = [p11, 1 - p22; 1 - p11, p22]
    ntot = n + burnin;
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
            Pt[i] = hamilton_filter_n(p, q, sqrt.(h[i - 1, :]), r[i - 1], Pt[i - 1]);
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
            s[i] = wsample([1, 2], P[:, s[i-1]])[1]; 
            r[i] = e[i] * sqrt(h[i, s[i]]);
        end
    else
        for i = 2:ntot
            h[i, 1:k] = ω .+ α.* r[i - 1]^2 + β.* h[i - 1, k + 1];
            Pt[i] = hamilton_filter_t(p, q, sqrt.(h[i - 1, :]), r[i- 1], Pt[i - 1], 7);
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
            s[i] = wsample([1, 2], P[:, s[i-1]])[1];
            r[i] = e[i] * sqrt(h[i, s[i]]);
        end
    end
    return r[burnin + 1: end], h[burnin + 1: end, :], Pt[burnin + 1: end], s[burnin + 1: end];
end


function simulate_gray_tv(n, distri, ω, α, β, C, D, burnin)
    # P = [p11, 1 - p22; 1 - p11, p22]
    ntot = n + burnin;
    k = length(ω);
    h = Matrix{Float64}(undef, ntot, k + 1);
    r = Vector{Float64}(undef, ntot);
    s = Vector{Int32}(undef, ntot);
    Pt = Vector{Float64}(undef, ntot);
    P = Matrix{Float32}(undef, k, k);

    e = ifelse(distri == "norm", rand(Normal(), ntot), sqrt(5/7).* rand(TDist(7), ntot));
    h[1, 1:k] .= 1.0;
    P[1, 1] = cdf(Normal(), C[1] + D[1] * 0);             
    P[2, 2] = cdf(Normal(), C[2] + D[2] * 0);             
    P[2, 1] = 1.0 - P[1, 1];
    P[1, 2] = 1.0 - P[2, 2];
    p = P[1, 1];
    q = P[2, 2];
    Pt[1] = (1 - q) / (2 - p - q);                 
    h[1, k + 1] = Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2];
    s[1] = wsample([1, 2], [Pt[1], 1 - Pt[1]])[1];
    r[1] = e[1] * sqrt(h[1, s[1]]);

    if distri == "norm"
        for i = 2:ntot
            P[1, 1] = cdf(Normal(), C[1] + D[1] * r[i - 1]);
            P[2, 1] = 1.0 - P[1, 1];
            P[2, 2] = cdf(Normal(), C[2] + D[2] * r[i - 1]);
            P[1, 2] = 1.0 - P[2, 2];
            p = P[1, 1];
            q = P[2, 2];
            h[i, 1:k] = ω .+ α.* r[i - 1]^2 + β.* h[i - 1, k + 1];
            Pt[i] = hamilton_filter_n(p, q, sqrt.(h[i - 1, :]), r[i - 1], Pt[i - 1]);
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
            s[i] = wsample([1, 2], P[:, s[i-1]])[1];
            r[i] = e[i] * sqrt(h[i, s[i]]);
        end
    else
        for i = 2:ntot
            P[1, 1] = cdf(Normal(), C[1] + D[1] * r[i - 1]);
            P[2, 1] = 1.0 - P[1, 1];
            P[2, 2] = cdf(Normal(), C[2] + D[2] * r[i - 1]);
            P[1, 2] = 1.0 - P[2, 2];
            p = P[1, 1];
            q = P[2, 2];
            h[i, 1:k] = ω .+ α.* r[i - 1]^2 + β.* h[i - 1, k + 1];
            Pt[i] = hamilton_filter_t(p, q, sqrt.(h[i - 1, :]), r[i- 1], Pt[i - 1], 7);
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
            s[i] = wsample([1, 2], P[:, s[i-1]])[1];
            r[i] = e[i] * sqrt(h[i, s[i]]);
        end
    end
    return r[burnin + 1: end], h[burnin + 1: end, :], Pt[burnin + 1: end], s[burnin + 1: end];
end









 
    

    

  