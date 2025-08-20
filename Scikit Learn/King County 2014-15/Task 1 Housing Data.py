# 📦 Import libraries
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns

# 🔁 Optional: Expand file path (for safety across systems)
file_path = os.path.expanduser("C:/Users/Lohit/Documents/kc_house_data.csv")

# 📥 Load data
df = pd.read_csv(file_path)

# 🧾 Basic Data Info
print("🔹 Shape:", df.shape)
print("\n🔹 Columns:\n", df.columns)
print("\n🔹 Data Types:\n", df.dtypes)
print("\n🔹 Missing values:\n", df.isnull().sum())
print("\n🔹 Basic Statistics:\n", df.describe())

# 📆 Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# 🔍 Check for duplicates
print("\n🔹 Duplicates:", df.duplicated().sum())

# 📊 Price Distribution
plt.figure(figsize=(10,6))
sns.histplot(df['price'], bins=50, kde=True, color='skyblue')
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 🔥 Correlation Heatmap (only numeric features)
plt.figure(figsize=(14,10))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# 📈 Top correlated features with price
correlation_with_price = corr['price'].sort_values(ascending=False)
print("\n🔹 Top features correlated with price:\n", correlation_with_price)

# 📦 Boxplot: Bedrooms vs Price
plt.figure(figsize=(14,6))
sns.boxplot(x='bedrooms', y='price', data=df, palette='Set2')
plt.title('Price vs Bedrooms')
plt.tight_layout()
plt.show()

# 📦 Boxplot: Floors vs Price
plt.figure(figsize=(14,6))
sns.boxplot(x='floors', y='price', data=df, palette='Set3')
plt.title('Price vs Floors')
plt.tight_layout()
plt.show()

# 📌 Scatter: Living Area vs Price
plt.figure(figsize=(14,6))
sns.scatterplot(x='sqft_living', y='price', data=df, alpha=0.5)
plt.title('Price vs Living Area (sqft)')
plt.tight_layout()
plt.show()

# 📌 Scatter: Grade vs Price
plt.figure(figsize=(14,6))
sns.scatterplot(x='grade', y='price', data=df, alpha=0.5)
plt.title('Price vs Grade')
plt.tight_layout()
plt.show()

# 🏗️ Feature Engineering: Year Sold & House Age
df['year_sold'] = df['date'].dt.year
df['house_age'] = df['year_sold'] - df['yr_built']

# 📦 Boxplot: Price by Year Sold
plt.figure(figsize=(12,6))
sns.boxplot(x='year_sold', y='price', data=df, palette='coolwarm')
plt.title('House Prices by Year Sold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 🕰️ Scatter: House Age vs Price
plt.figure(figsize=(12,6))
sns.scatterplot(x='house_age', y='price', data=df, alpha=0.5)
plt.title('House Age vs Price')
plt.tight_layout()
plt.show()