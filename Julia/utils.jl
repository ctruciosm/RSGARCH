#########################################
###       Auxiliary Functions       #####
######################################### 

function Tstudent(x::Real, η::Real)
    a = (gamma((1 + η) / (2 * η)))/(sqrt(π * (1 - 2 * η)/η) * gamma(1 / (2 * η)));
    b = (1 + η * x^2/(1 - 2 * η))^((η + 1)/(2 * η));
    return a/b;
end


function probability_regime_given_time_n(p::Real, q::Real, σ::Vector{Float64}, r::Real, Pt::Real)
    numA = (1 - q) * pdf(Normal(0, σ[2]), r) * (1 - Pt);
    numB = p * pdf(Normal(0, σ[1]), r) * Pt;
    deno = pdf(Normal(0, σ[1]), r) * Pt + pdf(Normal(0, σ[2]), r) * (1 - Pt);
    return numA/deno + numB/deno;
end

function probability_regime_given_time_t(p::Real, q::Real, σ::Vector{Float64}, r::Real, Pt::Real, ν::Real)
    numA = (1 - q) * sqrt(ν/(ν-2))/σ[2] * pdf(TDist(ν), r*sqrt(ν/(ν-2))/σ[2]) * (1 - Pt);
    numB = p * sqrt(ν/(ν-2))/σ[1] * pdf(TDist(ν), r*sqrt(ν/(ν-2))/σ[1]) * Pt;
    deno = sqrt(ν/(ν-2))/σ[1] * pdf(TDist(ν), r*sqrt(ν/(ν-2))/σ[1]) * Pt + sqrt(ν/(ν-2))/σ[2] * pdf(TDist(ν), r*sqrt(ν/(ν-2))/σ[2]) * (1 - Pt);
    return numA/deno + numB/deno;
end