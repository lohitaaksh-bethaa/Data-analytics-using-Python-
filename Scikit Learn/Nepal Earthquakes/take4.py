import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Load a working Carseats-like dataset from seaborn for demonstration
url = "https://raw.githubusercontent.com/amitness/earthquakes/master/earthquakes.csv"
df = pd.read_csv(url)

# Drop NA if any
df = df.dropna()

# Select relevant numeric features for clustering
X = df[['Latitude', 'Longitude']].copy()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans
db = DBSCAN(eps=0.5, min_samples=10)
df['Cluster'] = db.fit_predict(X)

# Plot clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Latitude', y='Longitude', hue='Cluster', palette='Set1')
plt.title('DBSCAN Clustering of Earthquake Data')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.grid(True)
plt.show()  
