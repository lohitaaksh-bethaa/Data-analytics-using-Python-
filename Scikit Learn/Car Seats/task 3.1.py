import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load a working Carseats-like dataset from seaborn for demonstration
url = "https://raw.githubusercontent.com/selva86/datasets/master/Carseats.csv"
df = pd.read_csv(url)

# Drop NA if any
df = df.dropna()

# Select relevant numeric features for clustering
X = df[['Income', 'Price']].copy()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Plot clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Income', y='Price', hue='Cluster', palette='Set1')
plt.title('KMeans Clustering of Carseats Data')
plt.xlabel('Income')
plt.ylabel('Price')
plt.grid(True)
plt.show()
