#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cmath as math
import sys
from docx2python import docx2python
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# reading word file
doc_data = docx2python('Carbonemissiondata.docx')

# get separate components of the document
headings = ["per capita CO2 (kg per person)", "per capita CO (kg per person)", "per capita CH4 (kg per person)"]

# making our dataframe using pandas
df = pd.DataFrame(doc_data.body[0][1:]). \
    applymap(lambda val: val[0].strip("\t"))

# retrieving original first row (columns headings)
df.columns = [val[0].strip("\t") for val in doc_data.body[0][0]]

# converting columns read from word file to float since It was found that docx 2 returns tables data as string
for i in range(3):
    df[headings[i]] = df[headings[i]].astype(float)
X= df.iloc[:,[1, 2, 3]].values
y_set = df.iloc[:,[0]].values #TYPE column with the four class initial assignment
# print(y_set)
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter=300, n_init = 10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s= 70, c = 'yellow', label='Centroids')
ax.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], X[y_kmeans == 0, 2], s= 50, c='red', label = 'cluster 1')
ax.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], X[y_kmeans == 1, 2], s=50, c='black', label = 'cluster 2')
ax.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], X[y_kmeans == 2, 2], s= 50, c='blue', label = 'cluster 3')

plt.title('Carbon Emission')
ax.set_xlabel(headings[0])
ax.set_ylabel(headings[1])
ax.set_zlabel(headings[2])
plt.legend()
plt.show()
# showGraph("Carbon Emission", headings[0], [min(x), max(x)], headings[1], [min(y), max(y)], headings[2], [min(z)-1, max(z)], [t1,t2,t3,t4,centroids])



