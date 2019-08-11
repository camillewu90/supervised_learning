## Use knn to predict the iris species

# Import KNeighborsClassifier from sklearn.neighbors
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
iris = datasets.load_iris()
iris.keys()
# Create arrays for the features and the response variable
y = iris.target
X = iris.data
# Create train and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.3,
                                                 random_state=12,
                                                 stratify=y)
# Create a k-NN classifier with 10 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=10)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Predict the labels for the training data X
y_pred = knn.predict(X_train)

# Predict and print the label for the test data point X_test
new_prediction = knn.predict(X_test)
print("Prediction: {}".format(new_prediction))

# Print the accuracy of the classifier on test set
print(knn.score(X_test, y_test))

# plot the knn model complexity curve,
# plot the training and testing accuracy scores for a variety of 
# different neighbor values.

# Setup arrays to store train and test accuracies
neighbors=np.arange(1,12)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i,k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train,y_train)
    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test,y_test)

# Generate a model complexity plot
plt.plot(neighbors,train_accuracy,label='Training Accuracy')
plt.plot(neighbors,test_accuracy,label='Testing Accuracy')
plt.title('k-NN: Varying Number of Neighbors')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
    


