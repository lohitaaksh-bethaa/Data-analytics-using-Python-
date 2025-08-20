# 🏡 King County Housing Data Analysis – Exploratory Data Analysis (EDA)

This project performs an **Exploratory Data Analysis (EDA)** on the **King County Housing Dataset**, which contains housing sales data for King County, USA.  

The goal is to uncover **key insights, trends, and correlations** that influence house prices, using data visualization and feature engineering techniques.  

---

## 📖 Project Overview  

- 📥 Load and explore the King County housing dataset  
- 🔍 Inspect data structure, missing values, and duplicates  
- 📊 Visualize price distribution, correlations, and feature relationships  
- 🏗️ Perform basic **feature engineering** (house age, year sold)  
- 📈 Analyze how features such as **bedrooms, bathrooms, sqft living, grade, and year sold** affect housing prices  

---

## 📂 Dataset Information  

- **Source:** `kc_house_data.csv`  
- **Rows:** ~21,600  
- **Columns:** 21  

Key features:  

- `price` → Target variable (house price 💰)  
- `date` → Date of house sale  
- `bedrooms` → Number of bedrooms  
- `bathrooms` → Number of bathrooms  
- `sqft_living` → Living area square footage  
- `sqft_lot` → Lot size in square feet  
- `floors` → Number of floors  
- `waterfront` → House overlooking waterfront (binary)  
- `grade` → Overall grade based on King County grading system  
- `yr_built` → Year the house was built  

---

## ⚙️ Workflow  

### 🔹 Step 1 – Data Exploration  
- Check shape, columns, datatypes  
- Handle missing values and duplicates  
- Convert date column to `datetime`  

### 🔹 Step 2 – Univariate Analysis  
- Price distribution  
- Summary statistics  

### 🔹 Step 3 – Correlation Analysis  
- Correlation heatmap  
- Identify top features influencing price  

### 🔹 Step 4 – Bivariate Analysis  
- Boxplots: Bedrooms vs Price, Floors vs Price  
- Scatterplots: Living Area vs Price, Grade vs Price  

### 🔹 Step 5 – Feature Engineering  
- Extract **year sold** from date  
- Calculate **house age**  

### 🔹 Step 6 – Visual Analysis  
- Price trends over years  
- House age vs Price relationship  

---

## 🧑‍💻 Code Example  

```python
# 📊 Price Distribution
plt.figure(figsize=(10,6))
sns.histplot(df['price'], bins=50, kde=True, color='skyblue')
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
