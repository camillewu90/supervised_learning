### supervised_learning numeric and visual EDA

from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
iris = datasets.load_iris()
## iris is a Bunch which is similar to a dictionary that has key value pair.
type(iris)
## print out all the keys
print(iris.keys())
## print our target data
print(iris.target)
## target data is a numpy array
type(iris.target)
# print out features data,features data is also a numpy array
print(iris.data)
# print out the dimesions of feature data
iris.data.shape
# print out the labels of the target data,0=setosa,1=versicolor,2=virginica
iris.target_names
# print out the feature names, which is the 4 column names 
iris.feature_names
# data description
iris.DESCR
# build the pandas dataframe for analysis
X = iris.data
Y = iris.target
df = pd.DataFrame(X,columns = iris.feature_names)
df.head()
# c stands for color,plots are color coded by different species
# s stands for size, it modulate the marker size
_ = pd.plotting.scatter_matrix(frame=df,c=Y,figsize=(8,8),s=30,marker='D')
plt.show()
