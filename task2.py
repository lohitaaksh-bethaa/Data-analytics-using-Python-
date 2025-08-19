import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# --- 1. Feature Selection and Preprocessing ---

# Sample data with categorical zipcode
data = {
    'Experience': [1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
    'Age':        [22, 25, 28, 32, 36, 24, 27, 30, 34, 40],
    'Role':       ['Analyst', 'Analyst', 'Analyst', 'Manager', 'Manager',
                   'Analyst', 'Analyst', 'Manager', 'Manager', 'Director'],
    'Zipcode':    ['10001', '10002', '10001', '10003', '10001',
                   '10002', '10003', '10001', '10003', '10002'],
    'Salary':     [30000, 40000, 50000, 65000, 70000, 35000, 45000, 60000, 68000, 90000]
}

df = pd.DataFrame(data)

# Select relevant features for the model (excluding Salary, which is the target)
features = ['Experience', 'Age', 'Role', 'Zipcode']
target = 'Salary'
X = df[features]
y = df[target]

# Preprocessing: One-hot encode categoricals, scale numericals
categorical_features = ['Role', 'Zipcode']
numerical_features = ['Experience', 'Age']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# --- 2. Model Building ---

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)

# --- 3. Model Evaluation ---

y_pred = pipeline.predict(X_test)

print("📊 Salary Prediction Metrics:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R^2 score:", r2_score(y_test, y_pred))
