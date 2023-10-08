import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


'''LOADING THE DATA'''
# Load the training and test data
X_train = np.loadtxt('/home/apisti01/Erasmus/Progetti_ML/exercise_6/disease_X_train.txt', delimiter=' ')
y_train = np.loadtxt('/home/apisti01/Erasmus/Progetti_ML/exercise_6/disease_y_train.txt', delimiter=' ')
X_test = np.loadtxt('/home/apisti01/Erasmus/Progetti_ML/exercise_6/disease_X_test.txt', delimiter=' ')
y_test = np.loadtxt('/home/apisti01/Erasmus/Progetti_ML/exercise_6/disease_y_test.txt', delimiter=' ')

#print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

'''A  BASELINE MODEL'''
# Compute the regression baseline
baseline_pred = np.mean(y_train)

# Compute the baseline MSE
baseline_mse = mean_squared_error(y_test, np.full_like(y_test, baseline_pred))

# Print the baseline MSE
print("Baseline MSE: ", baseline_mse)

'''B LINEAR REGRESSION'''
# Fit a linear model to the data
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predict the values for the test data using linear regression
y_pred_lin = lin_reg.predict(X_test)

# Print the test set MSE for linear regression
test_mse_lin = mean_squared_error(y_test, y_pred_lin)
print("Test set MSE for linear regression: ", test_mse_lin)

'''C DECISION TREE'''
# Fit a decision tree to the data
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)

# Predict the values for the test data using decision tree
y_pred_tree = tree_reg.predict(X_test)

# Print the test set MSE for decision tree
test_mse_tree = mean_squared_error(y_test, y_pred_tree)
print("Test set MSE for decision tree: ", test_mse_tree)

'''D RANDOM FOREST'''
# Fit a random forest to the data
rf_reg = RandomForestRegressor()
rf_reg.fit(X_train, y_train)

# Predict the values for the test data using random forest
y_pred_rf = rf_reg.predict(X_test)

# Print the test set MSE for random forest
test_mse_rf = mean_squared_error(y_test, y_pred_rf)
print("Test set MSE for random forest: ", test_mse_rf)
