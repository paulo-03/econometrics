# Install package if needed, then import librairies
# install.packages("lmtest")
# install.packages("rigr")
library(lmtest)
library(sandwich)
library(rigr)

# Read the CSV file into a data frame
data <- read.csv("/Users/pauloribeiro/Desktop/EPFL/Master Data Science/Semestre 1/Econometrics/Assignments/assignment_1/FIN403_GA1_data.csv")

######## Problem (a) ########

# Check if any duplicate value of "pid" attribute
duplicates_indices <- which(duplicated(data$pid))
print(length(duplicates_indices))

# Delete the duplicate rows
data <- data.frame(unique(data))

######## Problem (b) ########

# Transform the values of "centralair" attribute into dummy variables
data$centralair <- ifelse(data$centralair == "Y", 1, 0)

# Summarize statistical values for each attribute
summary(data)

hist(data$saleprice, main = "Histogram of Sale Price", 
     xlab = "Sale Price ($)", ylab = "Frequency", col = "lightblue")

hist(data$grlivarea, main = "Histogram of General Living Area", 
     xlab = "Living Area (ft²)", ylab = "Frequency", col = "lightblue")

hist(data$yearbuilt, main = "Histogram of Year of Construction", 
     xlab = "Year", ylab = "Frequency", col = "lightblue")

######## Problem (c) ########

# Fit a linear model using OLS w/ outliers
ols_model <- lm(saleprice ~ grlivarea, data = data)

# Summarize coefficient/information values of OLS model 
summary(ols_model)

# Create a scatter plot of saleprice against grlivarea
plot(data$grlivarea, data$saleprice, main = "Sale Price vs Living Area",
     xlab = "Living Area (ft²)", ylab = "Sale Price ($)")

# Fit a linear model using OLS w/o outliers
ols_model_no_outliers <- lm(saleprice ~ grlivarea, data = data[data$grlivarea < 4500,])

# Summarize coefficient/information values of OLS model w/o outliners
summary(ols_model_no_outliers)

# Add the fitted regression lines to the plot
abline(ols_model, col = "red", lwd = 2)
abline(ols_model_no_outliers, col = "orange", lwd = 2)

# Add a legend with separate labels for each regression line
legend("topright", legend = c("OLS Regression (w/ outliers)", "OLS Regression (w/o outliers)"), 
       col = c("red", "orange"), lwd = 2, cex = 0.8)

# Compute residuals for OLS model for part (e)
residuals_ols <- residuals(ols_model)

# Compute the hat matrix
X <- data$grlivarea
hat_matrix <- X %*% solve(t(X) %*% X) %*% X

# Check if outliers have high-leverage
K <- sum(diag(hat_matrix))
N <- length(X)
print("high-leverage points ?")
print(hat_matrix[which(data$grlivarea > 4500)] > 3*K/N)

# Insight of the three suspicious observations
print(data[data$grlivarea > 4500,])

# From now on, we drop the transactions with "salecondition" != "Normal"
data_normal <- data[data$salecondition == "Normal", ]

######## Problem (d) ######## 

# Create a histogram of "saleprice"
hist(data_normal$saleprice, main = "Histogram of Sale Price", 
     xlab = "Sale Price", ylab = "Frequency", col = "lightblue")

# Create a histogram of the logarithm of "saleprice"
hist(log(data_normal$saleprice), main = "Histogram of Log(Sale Price)", 
     xlab = "Log(Sale Price)", ylab = "Frequency", col = "lightblue")

# Fit a linear model using OLS for ln(saleprice)
ols_model_log <- lm(log(saleprice) ~ grlivarea, data = data_normal)

# Summarize coefficient/information values of OLS Log model 
summary(ols_model_log)

# Create a scatter plot of saleprice against grlivarea
plot(data_normal$grlivarea, log(data_normal$saleprice), 
     main = "Sale Price vs Living Area", xlab = "Living Area (ft²)", 
     ylab = "Ln(Sale Price) ($)")

# Add the fitted regression line to the plot
abline(ols_model_log, col = "red", lwd = 2)

# Add a legend
legend("topright", legend = "OLS Regression Line", col = "red", 
       lwd = 2, cex = 0.8)

# Predict the value of a specific living area witl OLS Log model
prediction <- predict(ols_model_log, newdata = data.frame(grlivarea = 2500))
print(prediction)

# Compute the std of the error terms to come back to non-logarithmic prediction
std_error_log <- sd(ols_model_log$residuals)
print(exp(prediction+0.5*(std_error_log**2)))

######## Problem (e) ######## 

# Compute residuals for OLS Log model
residuals_ols_log <- residuals(ols_model_log)

# Create scatter plot of residuals vs. grlivarea for ols_model
plot(data$grlivarea, residuals_ols, main = "Residuals vs. Living Area (OLS model)",
     xlab = "Living Area (grlivarea)", ylab = "Residuals")

# Create scatter plot of residuals vs. grlivarea for old_model_log
plot(data_normal$grlivarea, residuals_ols_log, 
     main = "Residuals vs. Living Area (OLS Log model)",
     xlab = "Living Area (grlivarea)", ylab = "Residuals")

######## Problem (f) ########

# Fit the more complex OLS model (regress() for format purpose)
complex_model_ <- regress("mean", log(saleprice) ~ grlivarea + yearbuilt + fullbath + centralair, data = data_normal)

# Summarize coefficient/information values of complex OLS Log model
complex_model_[["coefficients"]]

# Initialize the constant of your test
test <- c(0, 100, -9, 0, 0)

# Perform the test
lincom(complex_model_, test)

######## Problem (g) ########

# Reduced models
model_gr <- lm(grlivarea ~ yearbuilt + fullbath + centralair, data = data_normal)
model_fu <- lm(fullbath ~ grlivarea + yearbuilt + centralair, data = data_normal)
model_ye <- lm(yearbuilt ~ grlivarea + fullbath + centralair, data = data_normal)
model_ce <- lm(centralair ~ grlivarea + yearbuilt + fullbath, data = data_normal)

# Calculate the R-squared value of the reduced model
rsquared_gr <- summary(model_gr)$r.squared
rsquared_fu <- summary(model_fu)$r.squared
rsquared_ye <- summary(model_ye)$r.squared
rsquared_ce <- summary(model_ce)$r.squared

# VIF values
vif_gr <- 1/(1-rsquared_gr)
vif_fu <- 1/(1-rsquared_fu)
vif_ye <- 1/(1-rsquared_ye)
vif_ce <- 1/(1-rsquared_ce)

# RESET test w/o abnormal observations
resettest(log(saleprice) ~ grlivarea + yearbuilt + fullbath + centralair, data = data_normal)

# RESET test w/ abnormal observations
resettest(log(saleprice) ~ grlivarea + yearbuilt + fullbath + centralair, data = data)

######## Problem (h) ########

# Let's add a new attributes in our model
complex_model_bis <- lm(log(saleprice) ~ grlivarea + yearbuilt + fullbath + centralair + overallqual, 
                           data = data_normal)

# Summarize coefficient/information values of complex OLS Log model w/ overallqual
summary(complex_model_bis)

# Correlation between 'overallqual' and log(saleprice)
cor_overallqual_log_saleprice <- cor(data_normal$overallqual, log(data_normal$saleprice))
print(paste("Correlation between overallqual and log(saleprice):", cor_overallqual_log_saleprice))

# Correlation between 'grlivarea' and 'overallqual'
cor_grlivarea_overallqual <- cor(data_normal$grlivarea, data_normal$overallqual)
print(paste("Correlation between grlivarea and overallqual:", cor_grlivarea_overallqual))

######## Problem (i) ########

# Create the 'oldhouse' column
data_normal$oldhouse <- ifelse(data_normal$yearbuilt < 1970, 0, 1)

# Model with interaction and less attributes
old_house_model <- lm(log(saleprice) ~ grlivarea + fullbath + centralair + oldhouse + grlivarea:oldhouse, 
                      data = data_normal)

# Summarize coefficient/information values of complex OLS old house model
summary(old_house_model)

######## Problem (j) ########

# Define the threshold for old and new houses
cutoff_year <- 1970

# Create separate datasets for old and new houses
old_houses <- data_normal[data_normal$yearbuilt < cutoff_year, ]
new_houses <- data_normal[data_normal$yearbuilt >= cutoff_year, ]

# Fit separate regression models for old and new houses
model_old <- lm(log(saleprice) ~ grlivarea + fullbath + centralair, data = old_houses)
model_new <- lm(log(saleprice) ~ grlivarea + fullbath + centralair, data = new_houses)

# Calculate the sum of squared residuals for each model
s_square_old <- sum(model_old$residuals^2)
s_square_new <- sum(model_new$residuals^2)

# Fit a pooled model
model_pooled <- lm(log(saleprice) ~ grlivarea + fullbath + centralair, data = data_normal)

# Calculate the sum of squared residuals for the pooled model
s_square_pooled <- sum(model_pooled$residuals^2)

# Calculate the Chow statistic
chow_statistic <- ((s_square_pooled - (s_square_old + s_square_new)) / 3) / ((s_square_old + s_square_new) / (nrow(data_normal) - 6))

# Perform a hypothesis test to check for differences
p_value <- pf(chow_statistic, df1 = 3, df2 = nrow(data_normal) - 2)

# Print results
print(paste("Chow-statistics:", chow_statistic))
print(paste("P-value:", p_value))

