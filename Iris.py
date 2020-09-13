

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans


data=pd.read_csv("Iris.csv")

#sns.pairplot(data, hue = 'Species', palette = 'magma')
X=data.drop('Species', axis=1)

y=data['Species']

scaler=MinMaxScaler()
label=LabelEncoder()
X=X.drop('Id',axis=1)
y=label.fit_transform(y)
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2,random_state=1)
X_tr_sc=scaler.fit_transform(X_train)
X_tt_sc=scaler.transform(X_test)

#Decision Tree Classifier
tree=DecisionTreeClassifier(max_depth=2)
tree.fit(X_train,y_train)
predictions=tree.predict(X_test)
tree_acc= accuracy_score(y_test, predictions)
print('The accuracy using Decision Tree Cassifier', tree_acc)

#MLP classifier
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=1)
mlp.fit(X_tr_sc,y_train)
mlp_pred=mlp.predict(X_tt_sc)
mlp_acc= accuracy_score(y_test, mlp_pred)
print('The accuracy using Multi Classifier Perceptron is', mlp_acc)

#K nearesr Neighbors
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_tr_sc,y_train)
pred_knn=knn.predict(X_tt_sc)
knn_acc=accuracy_score(y_test,pred_knn)
print('The accuracy using K Nearest Neighbor is',knn_acc)




#Kmeans finding the best number of clusters
wcss=[]
for i in range(1,11):
        kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10,random_state=0 )
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
kmeans=KMeans(n_clusters=3)
kmeans=kmeans.fit(X)
print('The predicted labels are',kmeans.labels_)


#predictedY = np.choose(kmeans.labels_, [1, 0, 2]).astype(np.int64)
