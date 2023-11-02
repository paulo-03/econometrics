### Set-up ###

# Library Imports
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import abline_plot

# Data Import
raw_data = pd.read_csv("C:\\Users\\Babis\\Desktop\\FIN403_GA1_2023\\FIN403_GA1_data.csv")

# Missing values check
print("Missing Values Number for each Variable\n", raw_data.isnull().sum())


### Question (a) ###

filtered_data = raw_data.drop_duplicates("pid")

print("Total number of duplicates: ", raw_data.shape[0] - filtered_data.shape[0])


### Question (b) ###

# Dummy Variable: 1-hot encoding (1:True and 0:False)
# Performing inplace encoding
for row in range(filtered_data.shape[0]):
    
    if filtered_data["centralair"][row] == 'Y':
        
        filtered_data["centralair"][row] = 1
        
    else:
        
        filtered_data["centralair"][row] = 0
        
        
# Summary Statistics #

# 1. Continuous Variables: Table presenting 3 groups of descriptive Statistics
# A) Measures of central tendency: Mean, Median, Mode
# B) Measures of dispersion: Standard deviation, Coefficient of variation, IQR (Interquartile range)
# C) Measures of shape of distribution: Skewness, Kurtosis

# Function to compute the Coefficient of variation
cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100 

# Select continuous variables (yearbuilt can be consider continuous, as the number of
# categories is quite large)
continuous_variables = ['saleprice', 'grlivarea', 'yearbuilt']        

# Dictionary to store descriptive measures
df_cont_descriptive_measures = {"variable_names": continuous_variables, "mean":[], "median":[], 
                           "mode":[], "min":[], "max":[], "sd":[], "cv":[], "iqr":[], "skew":[], "kurt":[]}

for variable in continuous_variables:
    
    df_cont_descriptive_measures["mean"].append(filtered_data[variable].mean())
    df_cont_descriptive_measures["median"].append(filtered_data[variable].median())
    df_cont_descriptive_measures["mode"].append(filtered_data[variable].mode())
    df_cont_descriptive_measures["min"].append(filtered_data[variable].min())
    df_cont_descriptive_measures["max"].append(filtered_data[variable].max())
    df_cont_descriptive_measures["sd"].append(filtered_data[variable].std())
    df_cont_descriptive_measures["cv"].append(cv(filtered_data[variable]))
    df_cont_descriptive_measures["iqr"].append(filtered_data[variable].quantile(0.75) - filtered_data[variable].quantile(0.25))
    df_cont_descriptive_measures["skew"].append(filtered_data[variable].skew())
    df_cont_descriptive_measures["kurt"].append(filtered_data[variable].kurt())
    
    
# 2. Categorical Variables: Frequency tables for each category

categorical_variables = ['overallqual', 'centralair', 'fullbath', 'salecondition']

# Dictionary to store frequency tables
df_cat_descriptive_measures = {key: None for key in categorical_variables}

for variable in categorical_variables:
    
    df_cat_descriptive_measures[variable] = filtered_data[variable].value_counts()


### Question (c) ###
simple_model = smf.ols('saleprice ~ grlivarea', data = filtered_data)
simple_results = simple_model.fit()

# Regression plot for full dataset
ax = filtered_data.plot(x = 'grlivarea', y = 'saleprice', kind='scatter', xlabel = "Living Area", ylabel = "Price")
abline_plot(model_results=simple_results, ax=ax, color='r')

# Influence and Leverage
from statsmodels.stats.outliers_influence import OLSInfluence

influence = OLSInfluence(simple_results)
OLSInfluence.plot_influence(influence, size = 12)
# Comment: Observations 2181, 2180 and 1498 are problematic. All of those observations have abnormal sale conditions.
# Store previous values
import copy
abnormal_filtered_data = copy.deepcopy(filtered_data)
# Dropping observations with abnormal sale conditions:
filtered_data = filtered_data[filtered_data['salecondition'] == "Normal"].reset_index().iloc[:, 1:]
# Re-fit the regression model
simple_model = smf.ols('saleprice ~ grlivarea', data = filtered_data)
simple_results = simple_model.fit()


### Question (d) ###
# Setting up plots
figure, axis = plt.subplots(1, 2)
# Histogram of log(saleprice)
axis[0].hist(np.log(filtered_data['saleprice']))
axis[0].set_title("Histogram of Log Transf. of Prices") 
# Histogram of saleprice
axis[1].hist(filtered_data['saleprice'])
axis[1].set_title("Histogram of Prices")

# Fitting model for log prices
log_simple_model = smf.ols('np.log(saleprice) ~ grlivarea', data = filtered_data)
log_simple_results = log_simple_model.fit()
# Predicting price for 2,500 sqft. houses
predicted_log_price = np.dot(log_simple_results.params, np.array([1, 2500]))
# To translate the log price into the actual price, we need to use the formula: exp(log_prediction + 0.5 * s^2)
# Calculating sample residual variance
s2 = np.sum(log_simple_results.resid ** 2) / (filtered_data.shape[0] - 1)
# Final price prediction
predicted_price = np.exp(predicted_log_price + 0.5 * s2)


### Question (e) ###
# Setting up plots
figure, axis = plt.subplots(1, 2)
# Histogram of log(saleprice)
axis[0].scatter(filtered_data['grlivarea'], simple_results.resid)
axis[0].set_title("Res vs Liv. Area (No Transf.)") 
# Histogram of saleprice
axis[1].scatter(filtered_data['grlivarea'], log_simple_results.resid)
axis[1].set_title("Res vs Liv. Area (Log Transf.)") 
# Comment: In plot (1) it is implied that heteroskedasticity is present (observations are not uniformly scattered around 0).


### Question (f) ###
# Fitting complex model for log prices
complex_model = smf.ols('np.log(saleprice) ~ grlivarea + yearbuilt + fullbath + centralair', data = filtered_data)
complex_model_results = complex_model.fit()
# Comment: beta_1 represents the intercept, i.e., the expected value of the natural logarithm of sale price (ln(saleprice))
# when all the other predictor variables (grlivarea, yearbuilt, fullbath, and centralair) are equal to zero. Since this is 
# unrealistic, we cannot practically interpret it more than simply the intercept (since a house with zero features is no house).

# For the effects test, we need to test the following hypothesis: H_0: 100 beta_2 - 9 beta_3 = 0
# We use a simple t-test (slide 50, Chapter 2) to test the hypothesis:
r_vector = np.array([0, 0, 100, -9, 0])
complex_model_results.t_test(r_matrix = r_vector)
# Comment: Evidence that the two have equal effects to the log(price).


### Question (g) ###
# Check 1: Colinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Calculating VIF scores
variables = complex_model.exog
vif = [variance_inflation_factor(variables, i) for i in range(1, variables.shape[1])]
# Comment: No significant evidence of colinearity found.

# Check 2: Functional form (RESET)
from statsmodels.stats.diagnostic import linear_reset
# Ramsey’s RESET test for neglected nonlinearity
linear_reset(complex_model_results)

# RESET for data with abnormal transactions
complex_model_abnormal = smf.ols('np.log(saleprice) ~ grlivarea + yearbuilt + fullbath + centralair', data = abnormal_filtered_data)
complex_model_abnormal_results = complex_model_abnormal.fit()
linear_reset(complex_model_abnormal_results)


### Question (h) ###
# Calculating the new model
complex_model_2 = smf.ols('np.log(saleprice) ~ grlivarea + yearbuilt + fullbath + centralair + overallqual', data = filtered_data)
complex_model_results_2 = complex_model_2.fit()
# Comment: beta_2 goes from 0.0005 to 0.0004, thus its impact on the log(saleprice) is reduced.
    
# Correlation Coefficients
# 1. log(saleprice) and overallqual
rho_1 = np.corrcoef(np.log(filtered_data['saleprice']), filtered_data['overallqual'])[0, 1]
# 2. overallqual and grlivarea
rho_2 = np.corrcoef(filtered_data['grlivarea'], filtered_data['overallqual'])[0, 1]


### Question (i) ###
# Creating dummy variable oldhouse
oldhouse = [int(filtered_data.iloc[i, 2] >= 1970) for i in range(filtered_data.shape[0])]
# Update the data set
filtered_data['oldhouse'] = oldhouse
# Fitting the new model
complex_model_3 = smf.ols('np.log(saleprice) ~ grlivarea + fullbath + centralair + grlivarea * oldhouse', data = filtered_data)
complex_model_results_3 = complex_model_3.fit()
# Comments: 1) oldhouse is statistically insignificant, thus the old houses do not influence more/less the log_price compared to the newer ones.
# 2) livarea*oldhouse is statistically significant, which implies that the impact of living area on the sale price is different
# for old houses compared to newer houses. 


### Question (j) ###
# Partition the dataset
oldhouse_data = filtered_data[filtered_data['oldhouse'] == 0].reset_index(drop = True)
newhouse_data = filtered_data[filtered_data['oldhouse'] == 1].reset_index(drop = True)
# Fit a model for each category and calculate SSR
oldhouse_model = smf.ols('np.log(saleprice) ~ grlivarea + fullbath + centralair', data = oldhouse_data)
oldhouse_ssr = oldhouse_model.fit().ssr
newhouse_model = smf.ols('np.log(saleprice) ~ grlivarea + fullbath + centralair', data = newhouse_data)
newhouse_ssr = newhouse_model.fit().ssr
general_model = smf.ols('np.log(saleprice) ~ grlivarea + fullbath + centralair', data = filtered_data)
general_ssr = general_model.fit().ssr

J = filtered_data.shape[1]
k = oldhouse_data.shape[1]
N1 = oldhouse_data.shape[0]
N2 = newhouse_data.shape[0]
# Calculate test statistic value
chow = ((general_ssr-(oldhouse_ssr+newhouse_ssr))/J)/((oldhouse_ssr+newhouse_ssr)/(N1+N2-2*k))
# Perform Chow Test
chow_pvalue = scipy.stats.f.cdf(chow, J, N1+N2-2*k)
# Comment: Do not reject H_0