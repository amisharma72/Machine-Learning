import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


#loading datsets
digits = datasets.load_digits()
n_samples = len(digits.images)




X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.5, shuffle=False)


#DecisionTreeClassifier
tree=DecisionTreeClassifier(max_depth=10)
tree.fit(X_train,y_train)
predictions=tree.predict(X_test)
tree_accuracy=accuracy_score(y_test, predictions)
print('The accuracy score using DecisionTreeClassifier is',tree_accuracy)

#Multi Layer Perceptron
mlp=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(100,100), random_state=1)
mlp.fit(X_train,y_train)
mlp_pred=mlp.predict(X_test)
mlp_accuracy=accuracy_score(y_test, mlp_pred)
print('The accuracy score using MLP Classifier (using backprop)', mlp_accuracy)

#K nearest neighbor
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
pred_knn=knn.predict(X_test)
knn_acc=accuracy_score(y_test,pred_knn)
print('The accuracy using K Nearest Neighbor is', knn_acc)

#KMeans Clustering
wcss=[]
for i in range(1,20):
        kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10,random_state=0 )
        kmeans.fit(digits.data)
        wcss.append(kmeans.inertia_)

plt.plot(range(1,20),wcss)
plt.title('The Elbow Method Graph for Handwritten Digit datasets')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
