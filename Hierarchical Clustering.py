import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

data = {
    'Fresh': [100, 120, 110, 800, 820, 790, 300, 320, 310, 900],
    'Milk': [200, 210, 190, 100, 110, 105, 700, 720, 690, 120],
    'Grocery': [300, 310, 290, 150, 160, 140, 100, 110, 105, 170],
    'Frozen': [50, 60, 55, 400, 420, 390, 80, 90, 85, 410],
    'Detergents_Paper': [40, 45, 35, 30, 35, 25, 500, 520, 490, 40],
    'Delicassen': [20, 25, 18, 70, 75, 65, 30, 35, 28, 80]
}

df = pd.DataFrame(data)

print("Customer Dataset:")
print(df)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

linked = linkage(scaled_data, method='single')

plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

clusters = fcluster(linked, 3, criterion='maxclust')
df['Cluster'] = clusters

print("\nCustomer Clusters:")
print(df[['Fresh', 'Milk', 'Grocery', 'Cluster']])

plt.scatter(df['Fresh'], df['Milk'], c=clusters)

plt.title("Customer Clusters")
plt.xlabel("Fresh Spending")
plt.ylabel("Milk Spending")
plt.show()
