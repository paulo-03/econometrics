### Library Imports ###
library("car")
library("lmtest")
library("sandwich")
library("prais")
library("ivreg")

# For converting regression results to latex tables
library("stargazer")


### Data Import ###
# Set working directory
setwd("C:\\Users\\Babis\\Desktop\\Econometrics2")
# Import the data
data = read.csv("fish.csv")


### Part I ###

# (a) lavgprc regression on the four daily dummy variables
# Fit the linear regression model
model_dummy = lm(lavgprc ~ mon + tues + wed + thurs + t, data = data)
# Model results
summary(model_dummy)
# Comment: Since none of the coefficients of the dummy variables is statistically
# significant, we conclude that there is no significant evidence in the data that
# the price varies systematically within a week.

# (b) dummy regression with wave2 and wave3
model_dummy_waves = lm(lavgprc ~ mon + tues + wed + thurs + t + wave2 
				+ wave3, data = data)
# Model results
summary(model_dummy_waves)
# Comment: Individually, both variables are statistically significant at confidence
# level 5%.
# To test joint significance, we use the previous model without the wave variables
# to perform an F-test, based on the restricted model.
linearHypothesis(model_dummy_waves, c("wave2=0", "wave3=0"))
# Comment: We reject the null hypothesis that both coefficients are zero.

# (c) Time trend after the introduction of the wave variables
# Comment: The coefficient of the time trend became statistically insignificant in
# the second model. A possible explanation is omitted variable bias.

# (d) Autocorrelation of residual terms
# Plot of the time series of the residuals of model in (b)
dummy_waves_residual_ts = model_dummy_waves$resid
plot(dummy_waves_residual_ts, type = 'b', xlab="Time Stamp", ylab="Regression Residuals")
title("Evolution of Regression Residuals through Time")
# Comment: The plot suggests that there is positive autocorrelation in the time
# series of the residuals, while also exhibiting mean-reversion.
# Formal test for AR(1) autocorrelation
# 1. Durbin-Watson test:
durbinWatsonTest(model_dummy_waves)
# Comment: The Durbin-Watson test result suggests that there is evidence of autocorrelation
# in the data.
# 2. Breusch-Godfrey test
bgtest(model_dummy_waves, order=1)
# Comment: The Breusch-Godfrey test also suggests that there is evidence of
# autocorrelation in the data.
# Conclusion: Since there is significant evidence that autocorrelation is present,
# the OLS estimator is not BLUE anymore. Assuming that strict exogeneity holds (i.e.,
# our model properly captures the way causation works), the estimators continue to
# be unbiased. The standard errors are also wrong and need to be corrected.

# (e) Newey-West standard errors
# Newey-West variance-covariance matrix estimate
nw_vcov = NeweyWest(model_dummy_waves, lag = 4, prewhite = F, adjust = T)
# t-tests for corrected model
coeftest(model_dummy_waves, vcov = nw_vcov)
# Comment: Both wave2 and wave3 variables remain statistically significant at confidence
# level 5%.

# (f) Prais-Winsten estimates
# Prais-Winsten estimates for full model
model_dummy_waves_prais = prais_winsten(model_dummy_waves, data = data, index = "t")
# F-test for joint significance using the semi-robust covariance matrix estimators (vcovHC)
linearHypothesis(model_dummy_waves_prais, c("wave2=0", "wave3=0"), vcov.=vcovHC(model_dummy_waves_prais))
# Comment: We reject the null hypothesis that both coefficients are zero.
# We get an estimation for the rho coefficient by regressing the lagged residuals to the residuals
e_lag = model_dummy_waves_prais$resid[-length(model_dummy_waves_prais$resid)]
e_original = model_dummy_waves_prais$resid[-1]
lagged_regression = lm(e_original ~ e_lag - 1)
summary(lagged_regression)


### Part II ###

# (g) OLS regression for log(quantity)
quantity_model = lm(ltotqty ~ lavgprc + mon + tues + wed + thurs + t, data = data)
summary(quantity_model)
# Comment: The coefficient of log(price) is negative, which implies that as price increases, we expect the quantity
# to decrease, which is correct from an economical (supply and demand) standpoint. However, this is problematic from
# a modelling perspective, since this is the case of simultaneity and reverse causality.

# (h) wave2 variable as instrument
# A candidate instrument variable must satisfy two conditions:
# 1. Relevance condition: It is logical to assume that the wave influences the price of the product (since rough seas make 
# fishing more challenging, bad weather might affect the supply of fish and, consequently, their price)
# 2. Exclusion restriction: This assumption is difficult to establish without a thorough understanding of potential confounding 
# variables, which requires the a combination of further statistical analysis, economic reasoning and sensitivity analyses.

# (i) First-stage OLS regression
first_stage_reg = lm(lavgprc ~ wave2 + mon + tues + wed + thurs + t, data = data)
summary(first_stage_reg)
# Comment: Following the rule of thumb that the t-value must be >= 3.16 (since we only have one instrument), we conclude that
# there is significant evidence in the data that the wave2 variable is a strong instrument (assuming that the 2 assumptions
# hold).

# (j) IV regression with wave2 as instrument
wave_instrument_reg = ivreg(ltotqty ~ mon + tues + wed + thurs + t | lavgprc | wave2, data = data)
summary(wave_instrument_reg)
# Comment: After the addition of the instrumental variable, we expect that the price will reduce the quantity of a product even
# further.
# Standard errors: 0.184138 (OLS) and 0.421982 (IV). We know that the IV estimator is less precise than the OLS one, especially
# if x and z are only weakly correlated.

# (k) IV regression with speed3 as instrument
speed_instrument_reg = ivreg(ltotqty ~ mon + tues + wed + thurs + t | lavgprc | speed3, data = data)
summary(speed_instrument_reg)
# First stage regression
first_stage_reg_speed = lm(lavgprc ~ speed3 + mon + tues + wed + thurs + t, data = data)
summary(lfirst_stage_reg_speed)
# Comment: The t-value is now 2.565, which implies that speed3 is not a very strong instrument.

# (l) IV regression with wave2 and speed3 as instruments
multiple_instruments_reg = ivreg(ltotqty ~ mon + tues + wed + thurs + t | lavgprc | wave2 + speed3, data = data)
summary(multiple_instruments_reg)
# We can examine whether the two instruments are sufficiently strong by examining the first-stage regression
first_stage_reg_multiple = lm(lavgprc ~ wave2 + speed3 + mon + tues + wed + thurs + t, data = data)
linearHypothesis(first_stage_reg_multiple, c("wave2=0", "speed3=0"))
# Comment: We reject the null hypothesis that both coefficients are zero.

# (m) Οveridentifying restriction test
summary(multiple_instruments_reg)
# Comment: Since we dont reject the null hypothesis (Sargan: 0.1611), we conclude that there is evidence that our moments are
# consistent with the data, under the assumption that some instruments are valid.

# (n) Endogeneity test
summary(multiple_instruments_reg)
# Comment: At 5% confidence level, we do not reject the hypothesis that log(price) is exogenous, thus, there is evidence in the
# data that we could proceed with OLS.
