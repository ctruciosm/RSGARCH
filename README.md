# RSGARCH

Codes used in the paper "Forecasting Risk by GARCH Markov Chain Models" by Luiz K. Hotta, Maurício Zevallos, Carlos Trucíos and Pedro Valls. Codes are implemented in #julialang. 

To replicate the Monte Carlo experiment use the `MonteCarlo.jl` code.
To replicate the empirical application use the `EmpiricalApplication.jl` code.

- `DGP.jl`: Includes all DGPs used in the Monte Carlo simulation.
- `MaximumLIkelihood.jl`: Maximum Likelihood estimation according to Gray (1996) and Haas et al. (2004).
- `Forecast.jl`: Forecasts the conditional variance, VaR and ES.
- `utils.jl`: contains several internal functions used in `DGP.jl`, `MaximumLikelihood.jl`, and `Forecast.jl`.


## References

- Gray, S. F. (1996). Modeling the conditional distribution of interest rates as a regime-switching process. Journal of Financial Economics, 42(1), 27-62.
- Klaassen, F. (2002). Improving GARCH volatility forecasts with regime-switching GARCH. In Advances in Markov-switching models (pp. 223-254). Physica, Heidelberg.
- Haas, M., Mittnik, S., & Paolella, M. S. (2004). A new approach to Markov-switching GARCH models. Journal of financial Econometrics, 2(4), 493-530.
