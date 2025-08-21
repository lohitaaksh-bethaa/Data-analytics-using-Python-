# 💼 Salary Prediction with Machine Learning  

This project demonstrates how to build a **machine learning pipeline** to predict employee salaries based on factors like **experience, age, role, and zipcode**. The project uses **Random Forest Regression** along with preprocessing techniques like **one-hot encoding** for categorical data and **scaling** for numerical features.  

---

## 📌 Project Workflow  

### 1️⃣ Data Preparation  
- Created a **sample dataset** with the following features:  
  - **Experience** → Number of years of experience  
  - **Age** → Age of the employee  
  - **Role** → Job designation (Analyst, Manager, Director)  
  - **Zipcode** → Encodes geographic area  
  - **Salary** → Target variable to be predicted  

### 2️⃣ Feature Selection & Preprocessing  
- **Selected features**: `Experience`, `Age`, `Role`, `Zipcode`  
- **Target variable**: `Salary`  
- **Preprocessing techniques applied**:  
  - `StandardScaler` → Standardizes numerical features (`Experience`, `Age`)  
  - `OneHotEncoder` → Converts categorical features (`Role`, `Zipcode`) into machine-readable format  
- Used a `ColumnTransformer` to apply preprocessing steps.  

### 3️⃣ Model Building  
- Chose **Random Forest Regressor** (an ensemble model known for robustness).  
- Integrated preprocessing + model inside a **Pipeline** for seamless training and prediction.  

### 4️⃣ Model Training & Testing  
- Split dataset into **train (80%)** and **test (20%)** sets.  
- Trained the pipeline on training data.  
- Made predictions on test data.  

### 5️⃣ Model Evaluation  
Metrics used for evaluation:  
- **RMSE (Root Mean Squared Error)** → Measures average prediction error.  
- **MAE (Mean Absolute Error)** → Average absolute difference between predicted and actual salary.  
- **R² Score** → Explains how much variance in salary is captured by the model.  

---

## ⚙️ Tech Stack  
- **Python** 🐍  
- **Pandas** → Data handling  
- **Scikit-learn** → Preprocessing, model building, evaluation  
- **NumPy** → Numerical computations  

---

## 📊 Example Output  

After training and evaluation, the script prints metrics such as:  

