from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd 
import numpy as np

# Load dataset
housing = fetch_california_housing()
X,Y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
lr= LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
rf= RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

def print_metrics(model_name,y_test,y_pred):
    print(f"forward {model_name} performance:")
    print("rmse:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("mae:", mean_absolute_error(y_test, y_pred))
    print_metrics("Linear Regression", y_test, y_pred_lr)
    print_metrics("Random Forest", y_test, y_pred_rf)

print_metrics("Linear Regression", y_test, y_pred_lr)
print_metrics("Random Forest", y_test, y_pred_rf)