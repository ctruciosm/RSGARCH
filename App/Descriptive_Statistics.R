##################################################
###       Results Empirical Application       ####
##################################################
library(ggplot2)
library(dplyr)
library(Rcpp)
library(stringr)
library(xtable)
library(tidyr)
sourceCpp("scoring_functions.cpp")
source("Function_VaR_VQR.R")
source("GiacominiRossiTest.R")

log_returns <- function(x) {
  100*(log(x) - log(lag(x)))
}
  


### Import results
series <- read.csv("../FRB_H10.csv", head = TRUE)[-1, ] |> 
  drop_na() |> select(Date, EUR_USD, JPY_USD, CAD_USD, DKK_USD) |> 
  mutate_if(is.numeric, log_returns) |> 
  drop_na()


tidy_series <- pivot_longer(series, cols = EUR_USD:DKK_USD, values_to = "returns", names_to = "assets")


ggplot(tidy_series) + 
  geom_line(aes(x = as.Date(Date), y = returns, group = assets), color = "green4") + 
  facet_wrap(factor(assets, 
                    levels = c("EUR_USD", "JPY_USD", "CAD_USD", "DKK_USD"),
                    labels = c("EUR/USD", "JPY/USD", "CAD/USD", "DKK/USD")) ~ .) +
  ylab("Returns") + xlab("") +
  theme_bw()




tidy_series %>% 
  select(-Date) %>% 
  group_by(assets) %>% 
  summarise(minimum = min(returns),
            maximum = max(returns),
            mean = mean(returns),
            sd = sd(returns),
            skewness = moments::skewness(returns),
            kurtosis = moments::kurtosis(returns),
            LjungBox = Box.test(returns, lag = 10, "Ljung-Box")$p.value) %>% 
  knitr::kable(digits = 4, format = "latex", align = "lccccccc",
               table.envir = "table", label = "descriptive_statistics") %>% 
  save_kable(keep_tex = T, file = paste0("descriptive_statistics.tex"))





