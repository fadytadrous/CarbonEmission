import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from docx2python import docx2python
import pandas as pd

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

# we will work on col. 2 and 3
X = df.iloc[:, [2, 3]].values

wcss = []
# using elbow method to detect optimum clusters numbers
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
# plt.plot(range(1,11), wcss)
# plt.title('The Elbow Method')
# plt.xlabel("number of clusters")
# plt.ylabel('wcss')
# plt.show()

# the optimum number of clusters found were 4 using the elbow method
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# plotting using MatPlotlib
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='blue', label='cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='red', label='cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='cluster 4')
plt.title('Carbon Emission')
plt.xlabel("per capita CO")
plt.ylabel("per capita CH4")
plt.legend()
plt.show()
