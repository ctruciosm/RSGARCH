
#################################################
###  RSGARCH GPD: Simulate RSGARCH Models    ####
#################################################
using Random, Distributions, ForwardDiff

# https://github.com/yonghanjung/RegimeSwitching_GARCH/blob/master/simulation/msGarchSim.m


function simulate_haas(n, distri, ω, α, β, P, burnin)
    # P = [p11, 1 - p22; 1 - p11, p22]
    ntot = n + burnin;
    s = Vector{Int32}(undef, ntot);
    k = length(ω);
    h = Matrix{Float64}(undef, ntot, k);
    r = Vector{Float64}(undef, ntot);
    e = ifelse(distri == "norm", rand(Normal(), ntot), sqrt(5/7).* rand(TDist(7), ntot));
    h[1, :] .= ω ./ (1 .- (α .+ β));
    s[1] =  1;                                  
    r[1] = e[1] * sqrt(h[1, s[1]]);
    for i = 2:ntot
        h[i, :] .= ω .+ α.* r[i - 1]^2 + β.* h[i - 1, :];
        s[i] = wsample([1, 2], P[:, s[i-1]])[1];
        r[i] = e[i] * sqrt(h[i, s[i]]);
    end
    return r[burnin + 1: end], h[burnin + 1: end, :], s[burnin + 1: end];
end


