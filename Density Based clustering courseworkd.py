#!/usr/bin/env python
# coding: utf-8

# In[130]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics 

df = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv')

df.head()

df.describe()

df.shape

X = df.iloc[:, 1:3]
X.head()

scaler = StandardScaler()
scaled_X = pd.DataFrame(scaler.fit_transform(X))
scaled_X.columns = X.columns

# Data Visualization
sns.pairplot(scaled_X.iloc[:,0:3])

#Initialize empty lists for storing Silhouette scores and combinations of epsilon and min_samples
scores = []
param_combinations = []

#Define ranges to explore for epsilon and min_samples
eps_options = range(3, 12)
min_samples_options = range(3, 8)

#Iterate through different combinations of epsilon and min_samples
for eps in eps_options:
    for min_samples in min_samples_options:
       # Set the model and its parameters
        dbscan = DBSCAN(eps=(eps/100), min_samples=min_samples)
        # Fit the model to the data
        cluster_model = dbscan.fit(scaled_X)
         # Calculate the Silhouette score and append to the list
        score = metrics.silhouette_score(scaled_X, cluster_model.labels_, metric='euclidean')
        scores.append(score)
        # Append the combination of epsilon and min_samples to the list
        param_combinations.append(str(eps) + "|" + str(min_samples))

#Plot the resulting Silhouette scores on a graph
plt.figure(figsize=(20,8), dpi=300)
plt.plot(param_combinations, scores, 'bo-', color='black')
plt.xlabel('Epsilon/100 | MinPts')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score based on different combinations of Hyperparameters')
plt.show()





# Create a DBSCAN instance with desired parameters
dbscan = DBSCAN(eps=0.09, min_samples=4)

# Fit the model to the data
dbscan.fit(scaled_X)

# Predict the clusters
clusters = dbscan.fit_predict(scaled_X)

X['cluster'] = clusters

# visualize the clusters
plt.scatter(X['W0'], X['W1'], c=X['cluster'], cmap='rainbow', label = X['cluster'])
plt.title("Cluster Visualization")
plt.show()


# In[ ]:




