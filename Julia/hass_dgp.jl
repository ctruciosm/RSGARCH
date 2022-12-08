
#################################################
###  RSGARCH GPD: Simulate RSGARCH Models    ####
#################################################
using Random, Distributions

# https://github.com/yonghanjung/RegimeSwitching_GARCH/blob/master/simulation/msGarchSim.m


function simulate_haas(n, distri, ω, α, β, P, burnin)
    ntot = n + burnin;
    s = Vector{Int32}(undef, ntot);
    k = length(ω);
    h = zeros(ntot, k);
    r = zeros(ntot);
    e = ifelse(distri == "norm", rand(Normal(), ntot), sqrt(5/7).* rand(TDist(7), ntot));
    Pt = zeros(ntot);
    p = P[1, 1];
    q = P[2, 2];
    Pt[1] = (1 - q) / (2 - p - q);                         
    s[1] =  wsample([1, 2], [Pt[1], 1 - Pt[1]], 1)[1];
    h[1, :] = ω ./ (1 .- (α + β));
    r[1] = e[1] * sqrt(h[1, s[1]]);
    if (distri == "norm")
        for i = 2:ntot
            h[i, :] = ω .+ α.* r[i - 1]^2 + β.* h[i - 1, :];
            numA = (1 - q) * pdf(Normal(0, sqrt(h[i - 1, 2])), r[i - 1]) * (1 - Pt[i - 1]);
            numB = p * pdf(Normal(0, sqrt(h[i - 1, 1])), r[i - 1]) * Pt[i - 1];
            deno = pdf(Normal(0, sqrt(h[i - 1, 1])), r[i - 1]) * Pt[i - 1] +  pdf(Normal(0, sqrt(h[i - 1, 2])), r[i - 1]) * (1 - Pt[i - 1]);
            Pt[i] = numA/deno + numB/deno;
            s[i] = wsample([1, 2], [Pt[i], 1 - Pt[i]], 1)[1];
            r[i] = e[i] * sqrt(h[i, s[i]]);
        end
    else
        for i = 2:ntot
            h[i, :] = ω .+ α.* r[i - 1]^2 + β.* h[i - 1, :];
            numA = (1 - q) * sqrt(7/5) / sqrt(h[i - 1, 2]) * pdf(TDist(7), r[i - 1] * sqrt(7/5) / sqrt(h[i - 1, 2])) * (1 - Pt[i - 1]);
            numB = p * sqrt(7/5) / sqrt(h[i - 1, 1]) * pdf(TDist(7), r[i - 1] * sqrt(7/5) / sqrt(h[i - 1, 1])) * Pt[i - 1];
            deno = sqrt(7/5) / sqrt(h[i - 1, 1]) * pdf(TDist(7), r[i - 1] * sqrt(7/5) / sqrt(h[i - 1, 1])) * Pt[i - 1] + 
            sqrt(7/5) / sqrt(h[i - 1, 2]) * pdf(TDist(7), r[i - 1] * sqrt(7/5) / sqrt(h[i - 1, 2])) * (1 - Pt[i - 1]);
            Pt[i] = numA/deno + numB/deno;
            s[i] = wsample([1, 2], [Pt[i], 1 - Pt[i]], 1)[1];
            r[i] = e[i] * sqrt(h[i, s[i]]);
        end
    end
    return r[burnin + 1: end], h[burnin + 1: end, :], Pt[burnin + 1: end], s[burnin + 1: end];
end


