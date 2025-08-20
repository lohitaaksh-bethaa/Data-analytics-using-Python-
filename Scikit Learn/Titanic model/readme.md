# 🚢 Titanic Survival Prediction – End-to-End Machine Learning Pipeline  

This project builds a **Machine Learning pipeline** to predict whether a passenger survived the Titanic disaster using the Titanic dataset.  

Unlike many ad-hoc Titanic ML solutions, this project emphasizes **clean, reusable, and production-ready pipelines** using **Scikit-learn’s Pipeline and ColumnTransformer**.  

---

## 📖 Problem Statement  

The RMS Titanic sank in 1912, and survival was influenced by factors such as age, gender, class, and fare paid.  
The **goal** is to predict whether a passenger survived based on key features from the dataset.  

---

## 📂 Dataset Information  

Dataset used: [Titanic dataset](https://github.com/raorao/datasciencedojo/blob/master/Datasets/titanic.csv)  

Features used in this project:  

- **Pclass** – Ticket class (1 = First, 2 = Second, 3 = Third)  
- **Sex** – Gender of the passenger  
- **Age** – Age in years  
- **Fare** – Passenger fare paid  

Target variable:  

- **Survived** – Survival indicator (0 = Did not survive, 1 = Survived)  

---

## ⚙️ Workflow  

### 🔹 Step 1 – Data Preprocessing  
- Handle missing values with **SimpleImputer**  
  - Age → filled with mean  
  - Categorical variables → filled with “missing”  
- Encode categorical features with **OneHotEncoder**  
- Scale numeric features with **StandardScaler**  

### 🔹 Step 2 – Model Building  
- Use **Logistic Regression** for binary classification  
- Combine preprocessing and model into a **single pipeline** for cleaner code  

### 🔹 Step 3 – Model Training  
- Split dataset: 80% training, 20% testing  
- Fit pipeline on training data  
- Generate predictions on test data  

### 🔹 Step 4 – Evaluation  
- Use **Accuracy Score** to measure performance  
- Print accuracy on test set  

---

## 📊 Workflow Diagram  

```mermaid
flowchart TD
    A[Load Titanic Dataset] --> B[Feature Selection: Pclass, Sex, Age, Fare]
    B --> C[Preprocessing]
    C --> C1[Numeric: SimpleImputer + StandardScaler]
    C --> C2[Categorical: SimpleImputer + OneHotEncoder]
    C1 --> D[Pipeline Integration]
    C2 --> D
    D --> E[Logistic Regression Model]
    E --> F[Predictions]
    F --> G[Evaluate Accuracy]
