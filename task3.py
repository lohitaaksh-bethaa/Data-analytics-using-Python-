import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/carseats.csv"
df = pd.read_csv(url)
df = df.dropna()  # Ensure no missing values for simplicity
# --- 1. Feature Selection and Preprocessing ---
# Select relevant features for the model (excluding Sales, which is the target)
print(df[['Income', 'Price']].isnull().sum())
x = df[['Income', 'Price']].copy()
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
df['Cluster'] = kmeans.fit_predict(x)

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='Income', y='Price', hue='Cluster', palette='Set1')
plt.title('KMeans Clustering of Carseats Data')
plt.xlabel('Income')
plt.ylabel('Price')
plt.grid(True)    
plt.show()