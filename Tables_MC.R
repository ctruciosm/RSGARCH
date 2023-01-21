#####################################################
###        Monte Carlo Simulation Tables          ###
#####################################################
library(dplyr)
library(xtable)
library(ggplot2)
library(tidyr)

params_5000_n <- read.csv("params_5000_n.csv", head = FALSE)
params_2500_n <- read.csv("params_2500_n.csv", head = FALSE)
params_1000_n <- read.csv("params_1000_n.csv", head = FALSE)

omega_1 <- 0.18; alpha_1 <- 0.46; beta_1  <- 0.20; p11 <- 0.95
omega_2 <- 0.01; alpha_2 <- 0.16; beta_2  <- 0.30; p22 <- 0.98
sigma2_1 <- omega_1 / (1 - alpha_1 - beta_1)
sigma2_2 <- omega_2 / (1 - alpha_2 - beta_2)
true_parameters  <- c(omega_2, omega_1, alpha_2, alpha_1, beta_2, beta_1, p22, p11, alpha_2 + beta_2, alpha_1 + beta_1, sigma2_2, sigma2_1)

insample_measures <- function(v_true_params, m_hat_params) {
    m_true_params <- matrix(rep(v_true_params, 500), ncol = 12, byrow = TRUE)
    bias <- data.frame(m_true_params - m_hat_params[, 1:12], h = m_hat_params[, 14] - m_hat_params[, 13])
    mse <- bias^2
    avg_bias <- apply(bias, 2, mean)
    avg_mse <- apply(mse, 2, mean)
    return(data.frame(avg_bias, avg_mse))
}

xtable(cbind(insample_measures(true_parameters, params_5000_n),
             insample_measures(true_parameters, params_2500_n),
             insample_measures(true_parameters, params_1000_n[-213,])), digits = 5)

############### Boxplots ###############
coef_5000 <- data.frame(params_5000_n[, 1:12] / matrix(rep(true_parameters, 500), ncol = 12, byrow = TRUE), h = params_5000_n[, 13] / params_5000_n[, 14]) 
coef_2500 <- data.frame(params_2500_n[, 1:12] / matrix(rep(true_parameters, 500), ncol = 12, byrow = TRUE), h = params_2500_n[, 13] / params_2500_n[, 14])
coef_1000 <- data.frame(params_1000_n[, 1:12] / matrix(rep(true_parameters, 500), ncol = 12, byrow = TRUE), h = params_1000_n[, 13] / params_1000_n[, 14])

names <- c("omega_1", "omega_2", "alpha_1", "alpha_2", "beta_1", "beta_2", "p11", "p22", "pers_1", "pers_2", "sigma2_1", "sigma2_2", "h")
colnames(coef_5000) <- names
colnames(coef_2500) <- names
colnames(coef_1000) <- names

coef_5000 <- coef_5000 |> pivot_longer(cols = everything(), names_to = "parameters", values_to = "values") |> mutate(N = "N = 5000")
coef_2500 <- coef_2500 |> pivot_longer(cols = everything(), names_to = "parameters", values_to = "values") |> mutate(N = "N = 2500")
coef_1000 <- coef_1000 |> pivot_longer(cols = everything(), names_to = "parameters", values_to = "values") |> mutate(N = "N = 1000")
coef_estim <- rbind(coef_5000, coef_2500, coef_1000)

ggplot(coef_estim) + geom_boxplot(aes(y = values, x = N, fill = N)) + geom_hline(yintercept = 1, color = "red") + 
facet_grid(.~parameters) + ylim(c(0, 5.5)) +  theme(axis.text.x = element_text(angle = 90))

############### Densities ###############
coef_5000 <- data.frame(params_5000_n[, 1:12]) 
coef_2500 <- data.frame(params_2500_n[, 1:12])
coef_1000 <- data.frame(params_1000_n[-213, 1:12])

names <- c("omega_1", "omega_2", "alpha_1", "alpha_2", "beta_1", "beta_2", "p11", "p22", "pers_1", "pers_2", "sigma2_1", "sigma2_2")
colnames(coef_5000) <- names
colnames(coef_2500) <- names
colnames(coef_1000) <- names

coef_5000 <- coef_5000 |> pivot_longer(cols = everything(), names_to = "parameters", values_to = "values") |> mutate(N = "N = 5000")
coef_2500 <- coef_2500 |> pivot_longer(cols = everything(), names_to = "parameters", values_to = "values") |> mutate(N = "N = 2500")
coef_1000 <- coef_1000 |> pivot_longer(cols = everything(), names_to = "parameters", values_to = "values") |> mutate(N = "N = 1000")
coef_estim <- rbind(coef_5000, coef_2500, coef_1000)

coef_estim %>% ggplot() + geom_density(aes(x = values, color = N))  + 
  facet_wrap(vars(parameters),  scale = "free")