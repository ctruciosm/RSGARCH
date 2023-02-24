##################################################
###       Results Empirical Application       ####
##################################################
library(ggplot2)
library(dplyr)
library(stringr)
library(modelconf)
library(GAS)
library(esback)
library(rugarch)
library(xtable)
library(esreg)
library(optimx)
library(kableExtra)
library(Rcpp)
library(quantreg)
sourceCpp("scoring_functions.cpp")
source("Function_VaR_VQR.R")
source("GiacominiRossiTest.R")

### Import results
r_oos <- read.csv("r_oos_EUR_USD.csv", head = FALSE)[,1]
ES_1 <- read.csv("ES1_EUR_USD.csv", head = FALSE)[-2011,]
ES_2 <- read.csv("ES2_EUR_USD.csv", head = FALSE)[-2011,]
ES_5 <- read.csv("ES5_EUR_USD.csv", head = FALSE)[-2011,]
VaR_1 <- read.csv("VaR1_EUR_USD.csv", head = FALSE)[-2011,]
VaR_2 <- read.csv("VaR2_EUR_USD.csv", head = FALSE)[-2011,]
VaR_5 <- read.csv("VaR5_EUR_USD.csv", head = FALSE)[-2011,]

# Setup
K <- ncol(ES_1)
a1 <- 0.010
a2 <- 0.025
a5 <- 0.050
BackVaRES1 = BackVaRES2 = BackVaRES5 = matrix(0,ncol = 12,nrow = K) 
colnames(BackVaRES1) = colnames(BackVaRES2) = colnames(BackVaRES5) = c("Hits", "UC", "CC", "DQ", "VQ", "MFE", "NZ", "ESR_3", "AQL", "AFZG", "ANZ", "AAL")

for (i in 1:K) { 
  print(i)
  set.seed(1234)
  VaRBack1 <- BacktestVaR(r_oos, VaR_1[,i], alpha = a1, Lags = 4)
  VaRBack2 <- BacktestVaR(r_oos, VaR_2[,i], alpha = a2, Lags = 4)
  VaRBack5 <- BacktestVaR(r_oos, VaR_5[,i], alpha = a5, Lags = 4)

  EBack1 = ESTest(alpha = a1, r_oos, ES_1[,i], VaR_1[,i], conf.level = 0.95,  boot = TRUE, n.boot = 5000)
  EBack2 = ESTest(alpha = a2, r_oos, ES_2[,i], VaR_2[,i], conf.level = 0.95,  boot = TRUE, n.boot = 5000)
  EBack5 = ESTest(alpha = a5, r_oos, ES_5[,i], VaR_5[,i], conf.level = 0.95,  boot = TRUE, n.boot = 5000)

  BackVaRES1[i,] = c(mean(r_oos < VaR_1[,i])*100, 
                     VaRBack1$LRuc[2], VaRBack1$LRcc[2],VaRBack1$DQ$pvalue, VaR_VQR(r_oos, VaR_1[,i], a1),
                     EBack1$boot.p.value,
                     cc_backtest(r_oos,  VaR_1[,i], ES_1[,i], alpha  = a1)$pvalue_twosided_simple, 
                     esr_backtest(r_oos, VaR_1[,i], ES_1[,i],alpha  = a1, B = 0, version = 3)$pvalue_onesided_asymptotic,
                     mean(QL(matrix(VaR_1[,i], ncol = 1) ,r_oos, alpha = a1)),
                     mean(FZG(matrix(VaR_1[,i], ncol = 1), matrix(ES_1[,i], ncol = 1), r_oos, alpha = a1)),
                     mean(NZ(matrix(VaR_1[,i], ncol = 1), matrix(ES_1[,i], ncol = 1), r_oos, alpha = a1)),
                     mean(AL(matrix(VaR_1[,i], ncol = 1), matrix(ES_1[,i], ncol = 1), r_oos, alpha = a1)))
  
  BackVaRES2[i,] = c(mean(r_oos < VaR_2[,i])*100, 
                     VaRBack2$LRuc[2], VaRBack2$LRcc[2],VaRBack2$DQ$pvalue, VaR_VQR(r_oos, VaR_2[,i], a2),
                     EBack2$boot.p.value,
                     cc_backtest(r_oos,  VaR_2[,i], ES_2[,i], alpha  = a2)$pvalue_twosided_simple, 
                     esr_backtest(r_oos, VaR_2[,i], ES_2[,i],alpha  = a2, B = 0, version = 3)$pvalue_onesided_asymptotic,
                     mean(QL(matrix(VaR_2[,i], ncol = 1) ,r_oos, alpha = a2)),
                     mean(FZG(matrix(VaR_2[,i], ncol = 1), matrix(ES_2[,i], ncol = 1), r_oos, alpha = a2)),
                     mean(NZ(matrix(VaR_2[,i], ncol = 1), matrix(ES_2[,i], ncol = 1), r_oos, alpha = a2)),
                     mean(AL(matrix(VaR_2[,i], ncol = 1), matrix(ES_2[,i], ncol = 1), r_oos, alpha = a2)))
  
  BackVaRES5[i,] = c(mean(r_oos < VaR_5[,i])*100, 
                     VaRBack5$LRuc[2], VaRBack5$LRcc[2],VaRBack5$DQ$pvalue, VaR_VQR(r_oos, VaR_5[,i], a5),
                     EBack5$boot.p.value,
                     cc_backtest(r_oos,  VaR_5[,i], ES_5[,i], alpha  = a5)$pvalue_twosided_simple, 
                     esr_backtest(r_oos, VaR_5[,i], ES_5[,i],alpha  = a5, B = 0, version = 3)$pvalue_onesided_asymptotic,
                     mean(QL(matrix(VaR_5[,i], ncol = 1) ,r_oos, alpha = a5)),
                     mean(FZG(matrix(VaR_5[,i], ncol = 1), matrix(ES_5[,i], ncol = 1), r_oos, alpha = a5)),
                     mean(NZ(matrix(VaR_5[,i], ncol = 1), matrix(ES_5[,i], ncol = 1), r_oos, alpha = a5)),
                     mean(AL(matrix(VaR_5[,i], ncol = 1), matrix(ES_5[,i], ncol = 1), r_oos, alpha = a5))) 
}


xtable(BackVaRES1, digits = 4)

