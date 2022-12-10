#################################################
###  RSGARCH GPD: Simulate RSGARCH Models    ####
#################################################
using Random, Distributions

function simulate_klaassen(n, distri, ω, α, β, P, burnin)   

    ntot = n + burnin;
    k = length(ω);
    e = ifelse(distri == "norm", rand(Normal(), ntot), sqrt(5/7).* rand(TDist(7), ntot));
    h = zeros(ntot, k + 1);
    r = zeros(ntot);
    s = Vector{Int32}(undef, ntot);
    Pt = zeros(ntot);
    h[1, 1:k] .= 1.0;
    p = P[1, 1];
    q = P[2, 2];
    Pt[1] = (1 - q) / (2 - p - q);                          ## P(St = 1) - Pag 683 Hamilon (1994)
    h[1, k + 1] = Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2];
    s[1] = wsample([1, 2], [Pt[1], 1 - Pt[1]], 1)[1];
    r[1] = e[1] * sqrt(h[1, s[1]]);
    if distri == "norm"
        for i = 2:ntot
            h[i, 1:k] = ω .+ α.* r[i - 1]^2 + β.* h[i - 1, k + 1];
            numA = (1 - q) * pdf(Normal(0, sqrt(h[i - 1, 2])), r[i - 1]) * (1 - Pt[i - 1]);
            numB = p * pdf(Normal(0, sqrt(h[i - 1, 1])), r[i - 1]) * Pt[i - 1];
            deno = pdf(Normal(0, sqrt(h[i - 1, 1])), r[i - 1]) * Pt[i - 1] + 
                   pdf(Normal(0, sqrt(h[i - 1, 2])), r[i - 1]) * (1 - Pt[i - 1]);
            Pt[i] = numA/deno + numB/deno;
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
            s[i] = wsample([1, 2], [Pt[i], 1 - Pt[i]], 1)[1];
            r[i] = e[i] * sqrt(h[i, s[i]]);
        end
    else
        for i = 2:ntot
            h[i, 1:k] = ω .+ α.* r[i - 1]^2 + β.* h[i - 1, k + 1];
            numA = (1 - q) * sqrt(7/5) / sqrt(h[i - 1, 2]) * pdf(TDist(7), r[i - 1] * sqrt(7/5) / sqrt(h[i - 1, 2])) * (1 - Pt[i - 1]);
            numB = p * sqrt(7/5) / sqrt(h[i - 1, 1]) * pdf(TDist(7), r[i - 1] * sqrt(7/5) / sqrt(h[i - 1, 1])) * Pt[i - 1];
            deno = sqrt(7/5) / sqrt(h[i - 1, 1]) * pdf(TDist(7), r[i - 1] * sqrt(7/5) / sqrt(h[i - 1, 1])) * Pt[i - 1] + 
            sqrt(7/5) / sqrt(h[i - 1, 2]) * pdf(TDist(7), r[i - 1] * sqrt(7/5) / sqrt(h[i - 1, 2])) * (1 - Pt[i - 1]);
            Pt[i] = numA/deno + numB/deno;
            h[i, k + 1] = Pt[i] * h[i, 1] + (1 - Pt[i]) * h[i, 2];
            s[i] = wsample([1, 2], [Pt[i], 1 - Pt[i]], 1)[1];
            r[i] = e[i] * sqrt(h[i, s[i]]);
        end
    end
 



   return r[burnin + 1: end], h[burnin + 1: end, :], Pt[burnin + 1: end], s[burnin + 1: end];
end

