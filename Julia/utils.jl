#########################################
###       Auxiliary Functions       #####
######################################### 

function Tstudent(x::Real, η::Real)
    a = gamma((1 + η) / (2 * η))/(sqrt(π * (1 - 2 * η)/η) * gamma(1 / (2 * η)));
    b = (1 + η * x^2/(1 - 2 * η))^((η + 1)/(2 * η));
    return a/b;
end

function Gaussian(x::Real, μ::Real, σ::Real)
    return exp(-(x - μ)^2/(2*σ^2))/(sqrt(2*pi)*σ);
end

function probability_regime_given_time_n(p::Real, q::Real, σ::Vector{Float64}, r::Real, Pt::Real)
    numA = (1 - q) * pdf(Normal(0, σ[2]), r) * (1 - Pt);
    numB = p * pdf(Normal(0, σ[1]), r) * Pt;
    deno = pdf(Normal(0, σ[1]), r) * Pt + pdf(Normal(0, σ[2]), r) * (1 - Pt);
    l = numA/deno + numB/deno;
    return l;
end

function probability_regime_given_time_t(p::Real, q::Real, σ::Vector{Float64}, r::Real, Pt::Real, ν::Real)
    numA = (1 - q) * sqrt(ν/(ν-2))/σ[2] * pdf(TDist(ν), r*sqrt(ν/(ν-2))/σ[2]) * (1 - Pt);
    numB = p * sqrt(ν/(ν-2))/σ[1] * pdf(TDist(ν), r*sqrt(ν/(ν-2))/σ[1]) * Pt;
    deno = sqrt(ν/(ν-2))/σ[1] * pdf(TDist(ν), r*sqrt(ν/(ν-2))/σ[1]) * Pt + sqrt(ν/(ν-2))/σ[2] * pdf(TDist(ν), r*sqrt(ν/(ν-2))/σ[2]) * (1 - Pt);
    return numA/deno + numB/deno;
end

function probability_regime_given_time_it(p::Real, q::Real, σ::Vector{Float64}, r::Real, Pt::Real, η::Real)
    numA = (1 - q) * 1/σ[2] * Tstudent(r / σ[2], η) *(1 - Pt);
    numB = p * 1/σ[1] * Tstudent(r / σ[1], η) * Pt;
    deno = 1/σ[1] * Tstudent(r / σ[1], η) * Pt +  1/σ[2] * Tstudent(r / σ[2], η) *(1 - Pt);
    return numA/deno + numB/deno;
end

function gray_transform2(param)
    # param: omega1, omega2, alpha1, alpha2, beta1, beta2, P, Q
    # Mauricio
    t_param = similar(param);
    t_param[1:2] .= log.(param[1:2]);
    t_param[3:8] .= log.(param[3:8] ./ (1 .- param[3:8]));
    return t_param;
end

function param_transform(t_param)
    param = similar(t_param);
    param[1:2] .= exp.(-t_param[1:2]);
    param[3:4] .= exp.(-t_param[3:4]) ./ (1 .+ exp.(-t_param[3:4]) .+ exp.(-t_param[5:6]));
    param[5:6] .= exp.(-t_param[5:6]) ./ (1 .+ exp.(-t_param[3:4]) .+ exp.(-t_param[5:6]));
    param[7:8] .= 1 ./(1 .+ exp.(-t_param[7:8])); 
    return param;
end

function garch_transform(t_param)
    param = similar(t_param);
    param[1] = exp(-t_param[1]);
    param[2] = exp(-t_param[2]) / (1 + exp(-t_param[2]) + exp(-t_param[3]));
    param[3] = exp(-t_param[3]) / (1 + exp(-t_param[2]) + exp(-t_param[3]));
    return param;
end