#################################################
###  RSGARCH GPD: Simulate RSGARCH Models    ####
#################################################
using Random, Distributions

function simulate_gray(n, distri, ω, α, β, time_varying, P, C, D, burnin)
   if (!time_varying & isnothing(P)) 
    @error "Transition matrix P should be provided"
   end
   if (time_varying & (isnothing(C) || isnothing(D)))
    @error "Vectors C and D should be provided"
   end
   

   ntot = n + burnin;
   k = length(ω);
   e = ifelse(distri == "norm", rand(Normal(), ntot), sqrt(5/7).* rand(TDist(7), ntot));
   h = zeros(ntot, k + 1);
   r = Vector{Float64}(undef, ntot);
   s = Vector{Int32}(undef, ntot);
   h[1, 1:k] .= 1.0;

   if (!time_varying)
    h[1, k + 1] = Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2];
    s[1] = 1;                       # Initial regime
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
   else
    p = cdf(Normal(), C[1] + D[1] * 0);              ## 0 é a melhor opção?
    q = cdf(Normal(), C[2] + D[2] * 0);              ## 0 é a melhor opção?
    Pt[1] = (1 - q) / (2 - p - q);                   ## Unconditional Probability: P(St = 1) - Pag 683 Hamilon (1994)
    h[1, k + 1] = Pt[1] * h[1, 1] + (1 - Pt[1]) * h[1, 2];
    s[1] = wsample([1, 2], [Pt[1], 1 - Pt[1]], 1)[1];
    r[1] = e[1] * sqrt(h[1, s[1]]);
    if distri == "norm"
        for i = 2:ntot
            p = cdf(Normal(), C[1] + D[1] * r[i - 1]);
            q = cdf(Normal(), C[2] + D[2] * r[i - 1]);
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
            p = cdf(Normal(), C[1] + D[1] * r[i - 1]);
            q = cdf(Normal(), C[2] + D[2] * r[i - 1]);
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
   end
   return r[burnin + 1: end], h[burnin + 1: end, :], Pt[burnin + 1: end], s[burnin + 1: end];
end


