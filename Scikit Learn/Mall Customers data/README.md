# 🛒 Mall Customers Spending Classification

This project applies **Machine Learning** to classify mall customers into **spending categories** (`Low`, `Medium`, `High`) based on their demographic and financial details.  
It uses the **Random Forest Classifier** from `scikit-learn` to build and evaluate the model.

---

## 📌 Project Overview
- Dataset: [Mall Customers Cleaned](https://github.com/sayooj17/task-1-data-cleaning/blob/main/mall_customers_cleaned.csv)  
- Objective: Predict **customer spending category** from features like:
  - Annual Income
  - Age
  - Gender
- Categories:
  - **High** → Spending Score ≥ 85  
  - **Medium** → Spending Score 50–84  
  - **Low** → Spending Score < 50  

---

## ⚙️ Tech Stack
- **Python**
- **pandas, numpy** → Data handling
- **scikit-learn** → Preprocessing & Model building
- **matplotlib, seaborn** → Visualization

---

## 🧑‍💻 Steps in the Project
1. **Data Loading & Cleaning**  
   - Read CSV from GitHub  
   - Created `spend_category` target column  

2. **Preprocessing**  
   - Encoded `Gender` and target variable  
   - Scaled numeric features (Age, Income)  

3. **Modeling**  
   - Trained **Random Forest Classifier**  
   - Used **train-test split (80/20)**  

4. **Evaluation**  
   - Accuracy Score  
   - Classification Report (Precision, Recall, F1)  
   - Confusion Matrix & Feature Importance  

---

## 📊 Results
- Achieved **high accuracy** in predicting customer categories  
- Feature importance revealed **Annual Income** and **Spending Score** as strong predictors  

---

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/mall-customers-classification.git
   cd mall-customers-classification
