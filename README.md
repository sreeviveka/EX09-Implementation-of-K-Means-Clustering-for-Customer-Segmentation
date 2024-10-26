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
RegisterNumber: 2305001031  
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
![WhatsApp Image 2024-10-26 at 15 20 10_0e605aac](https://github.com/user-attachments/assets/d6e4214b-5b2d-425a-9cab-8b73566216fd)
![WhatsApp Image 2024-10-26 at 15 20 10_673f6841](https://github.com/user-attachments/assets/4a013a58-df07-4e13-89f5-d2b33e158427)
![WhatsApp Image 2024-10-26 at 15 20 10_2e6cc0d0](https://github.com/user-attachments/assets/74e35b92-11dc-4471-8dd8-954e58cdb28a)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
