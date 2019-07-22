#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing msme dataset
dataset = pd.read_csv('msme.csv')

dataset.head()
dataset.dtypes

# fitting dataset
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
dataset = dataset.apply(le.fit_transform)

dataset.dtypes
dataset.head()

x = dataset.iloc[:, [0,2,3,4]].values
x

# elbow method to find optimal no. of clusters
from sklearn.cluster import KMeans
wcss= []
for i in range (1,11):
    kmeans= KMeans(n_clusters=i, init='k-means++',max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('No_of_clusters')
plt.ylabel('wcss')
plt.show()

# applying k-means to msme dataset
kmeans= KMeans(n_clusters=3, init='k-means++',max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(x)
y_kmeans

# visualing clusters
plt.scatter(x[y_kmeans == 0, 2], x[y_kmeans == 0, 3], s = 100, c = 'red', label = 'Quantity<200')
plt.scatter(x[y_kmeans == 1, 2], x[y_kmeans == 1, 3], s = 100, c = 'blue', label = 'Quantity>400')
plt.scatter(x[y_kmeans == 2, 2], x[y_kmeans == 2, 3], s = 100, c = 'green', label = 'Quantity<400')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of msme products')
plt.xlabel('Quantity')
plt.ylabel('Pct')
plt.legend()
plt.show()