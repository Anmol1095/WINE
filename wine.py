#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[11]:


## read csv file
wine = pd.read_csv("WineQT.csv",index_col="Id")
wine.head()


# In[12]:


##shape of data
wine.shape


# In[13]:


## features in data 
wine.columns


# In[14]:


wine.describe()


# In[15]:


wine.isnull().any().any()


# In[16]:


wine.info()


# In[17]:


## number of unique values in each feature 
for col in wine.columns.values:
    print("Number of unique values of {} : {}".format(col, wine[col].nunique()))


# In[18]:


sns.catplot(x='quality',data=wine,kind='count')


# In[19]:


plt.figure(figsize=(10,10))
sns.heatmap(wine.corr(),color ="k",annot=True)


# In[20]:


plt.figure(figsize=(10,15))
for i, col in enumerate(list(wine.columns.values)):
    plt.subplot(4,3,i+1)
    wine.boxplot(col)
    plt.grid()
    plt.tight_layout()


# In[21]:


plt.figure(figsize=(20,16))
for i, col in enumerate(list(wine.columns.values)):
    plt.subplot(4,3,i + 1)
    sns.distplot(wine[col], color='r',kde=True,label='data')
    plt.grid()
    plt.legend(loc='upper right')
    plt.tight_layout()


# # Does the data include any distinct groups of wines? If so, please identify and present these appropriately.
# 

# In[22]:


from sklearn.cluster import KMeans


# In[23]:


#kmeans = KMeans(n_clusters=2)


# In[24]:


#kmeans.fit(wine.drop('Private',axis=1))


# In[25]:


#kmeans.cluster_centers_


# In[26]:


#def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0


# In[ ]:


#df['Cluster'] = df['Private'].apply(converter)


# In[ ]:


#from sklearn.metrics import confusion_matrix,classification_report
#print(confusion_matrix(df['Cluster'],kmeans.labels_))
#print(classification_report(df['Cluster'],kmeans.labels_))


# In[ ]:


# kmeans = KMeans(n_clusters=3, init = 'random', max_iter = 100, random_state = 5).fit(wine.iloc[:,[12,1]])
# centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns = list(wine_df.iloc[:,[12,1]].columns.values))
# fig, ax = plt.subplots(1, 1)
# wine_df.plot.scatter(x = 'Alcohol', y = 'OD280/OD315', c= kmeans.labels_, figsize=(12,8), colormap='jet', ax=ax, mark_right=False)
# centroids_df.plot.scatter(x = 'Alcohol', y = 'OD280/OD315', ax = ax,  s = 80, mark_right=False)


# In[30]:


# Normalizing over the standard deviation
wine_dropped = wine.drop('quality', axis=1)
X =wine_dropped.values[:, 1:]
Clus_dataset = StandardScaler().fit_transform(X)


# In[31]:


# Basically, number of clusters = the x-axis value of the point that is the corner of the "elbow"(the plot looks often looks like an elbow)
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300, n_init=12, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
f3, ax = plt.subplots(figsize=(8, 6))
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[32]:


# build the model with the output from elbow method which is 2
clusterNum = 2
k_means =KMeans(init='k-means++', n_clusters=clusterNum, n_init=12)
k_means.fit(X)
labels = k_means.labels_
print(labels)


# In[33]:


# We assign the labels to each row in dataframe.
wine_dropped['Clus_km'] = labels
print(wine_dropped.head())

print(wine_dropped.groupby('Clus_km').mean())


# In[34]:


# create 2 dimensional graph
f3, ax = plt.subplots(figsize=(16, 12))
plt.scatter(X[:, 9], X[:, 5], c=labels.astype(np.float), alpha=.5)
plt.xlabel('alcohol', fontsize=18)
plt.ylabel('total sulfur dioxide', fontsize=16)


# In[35]:


# create 3 dimensional graph
from mpl_toolkits.mplot3d import Axes3D
f4 = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(f4, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
ax.set_xlabel('alcohol')
ax.set_ylabel('total sulfur dioxide')
ax.set_zlabel('pH')

ax.scatter(X[:, 9], X[:, 5], X[:, 7], c= labels.astype(np.float))


# # Splitting dataset
# 

# In[1]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=40)


# In[ ]:




