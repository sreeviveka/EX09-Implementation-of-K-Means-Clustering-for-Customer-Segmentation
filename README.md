# EX 9 Implementation of K Means Clustering for Customer Segmentation
## DATE: 
## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preprocessing
 2. Initialize Centroids
 3. Assign Clusters
 4. Update Centroids

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Sreeviveka V.S 
RegisterNumber:  2305001031
*/
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

data=pd.read_csv('/content/Mall_Customers_EX8.csv')
data

#Extract features
X=data[['Annual Income (k$)','Spending Score (1-100)']]

plt.figure(figsize=(4,4))
plt.scatter(X['Annual Income (k$)'],X['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

k=3
Kmeans=KMeans(n_clusters=k)
Kmeans.fit(X)
centroids=Kmeans.cluster_centers_


labels=Kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)

colors=['r','g','b']
for i in range(k):
  cluster_points=X[labels==i]
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],color=colors[i],label=f'Cluster{i+1}')
  distances=euclidean_distances(cluster_points,[centroids[i]])
  radius=np.max(distances)
  circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,color='k',label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```

## Output:
![WhatsApp Image 2024-10-26 at 15 20 10_69c0bd52](https://github.com/user-attachments/assets/8015de13-a46e-4231-bb08-2309682fa787)
![WhatsApp Image 2024-10-26 at 15 20 10_a6b6eb62](https://github.com/user-attachments/assets/f876d1c7-8dc2-4cd1-af26-5bb4398cc437)
![WhatsApp Image 2024-10-26 at 15 20 10_82f20aad](https://github.com/user-attachments/assets/05b183ac-5975-42ca-9fc1-38bbf203c374)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
