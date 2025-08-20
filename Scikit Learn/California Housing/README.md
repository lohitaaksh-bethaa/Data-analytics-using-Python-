# **California Housing Price Prediction**

This project demonstrates how to build and evaluate Machine Learning regression models to predict housing prices in California. We compare Linear Regression and Random Forest Regressor using the popular fetch_california_housing dataset.

# **Project Overview**

**📊 Dataset: California Housing**
 from sklearn.dataset

**🔧 Models:**

- Linear Regression
- Random Forest Regressor

**📈 Metrics:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)

**⚙ Workflow**

- Data Loading – Fetch California Housing dataset.
- Train-Test Split – 80% training, 20% testing.
- Model Training – Fit Linear Regression & Random Forest.
- Evaluation – Compare models using regression metrics.

**Code Example**
 Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

**Random Forest Regressor**
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

**Print metrics**
print_metrics("Linear Regression", y_test, y_pred_lr)
print_metrics("Random Forest", y_test, y_pred_rf)

**📊 Expected Output**

Example model performance (values may differ):

Model	RMSE	MAE	R²
Linear Regression	0.72	0.53	0.60
Random Forest	0.48	0.34	0.82

**🛠️ Tech Stack**
- Python 🐍
- Scikit-learn 🤖
- Pandas 🐼
- NumPy 🔢

**🌟 Key Takeaways**
- Linear Regression gives a baseline performance.
- Random Forest significantly improves accuracy by capturing non-linear patterns.
- RMSE and MAE show error magnitude, while R² explains variance explained by the model.

**🔮 Future Enhancements**
- Add feature scaling and feature importance analysis.
- Try advanced models: Gradient Boosting, XGBoost, Neural Networks.
- Build a web app for interactive housing price predictions.
