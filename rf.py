import numpy
import scipy.io

mat = scipy.io.loadmat('ELEC4830_Final_project.mat')

Y_raw = mat['trainState']
noNan = numpy.where(numpy.isnan(Y_raw) == False)
Y = Y_raw.take(noNan[1])
Y_raw.shape, Y.shape

X_raw =  mat['trainSpike']
prev_data = 10
# X = X_raw[:,noNan[1]]
X = numpy.array([])
X = numpy.mean(X_raw[:, noNan[1][0]-prev_data:noNan[1][0]], axis = 1)[numpy.newaxis].T
for i in range(1, len(noNan[1])):
    a = numpy.mean(X_raw[:, noNan[1][i]-prev_data:noNan[1][i]], axis = 1)[numpy.newaxis].T
    X = numpy.append(X, a, axis = 1)
X_raw.shape, X.shape

X = X.T
Y = Y.T
X.shape, Y.shape

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

X_train.shape, Y_train.shape

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

rf = RandomForestClassifier()

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# grid_search.fit(X_train,Y_train)
# grid_search.best_params_

clf = RandomForestClassifier(bootstrap= True,
 max_depth= 90,
 max_features= 3,
 min_samples_leaf= 3,
 min_samples_split= 8,
 n_estimators= 300)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

import pandas as pd
feature_imp = pd.Series(clf.feature_importances_).sort_values(ascending=False)
feature_imp

import matplotlib.pyplot as plt
import seaborn as sns
# Creating a bar plot
sns.barplot(x=feature_imp.index, y=feature_imp)
# Add labels to your graph
plt.xlabel('Features')
plt.ylabel('Feature Importance Score')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, y_pred))

X = X.T[[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15]]
X = X.T
X.shape

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

rf = RandomForestClassifier()

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# grid_search.fit(X_train,Y_train)
# grid_search.best_params_

clf = RandomForestClassifier(bootstrap= True,
 max_depth= 110,
 max_features= 3,
 min_samples_leaf= 3,
 min_samples_split= 8,
 n_estimators= 300)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, y_pred))

X_test = mat['testSpike']
X_test.shape

l = 2
X = numpy.array([])
X = numpy.mean(X_test[:, 0: 1], axis = 1)[numpy.newaxis].T
for i in range(2, len(X_test[0])+1):
  if l != prev_data:
#     print(l,i)
    a = numpy.mean(X_test[:, i-l: i], axis = 1)[numpy.newaxis].T
    X = numpy.append(X, a, axis = 1)
    l+=1
  else:
    a = numpy.mean(X_test[:, i-prev_data: i], axis = 1)[numpy.newaxis].T
    X = numpy.append(X, a, axis = 1)
X = X[[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15]]
X = X.T
X.shape
# print(X)

# X_test = scaler.transform(X)
# decodedState = clf.predict(X_test)
# scipy.io.savemat('/content/drive/My Drive/4830_final_proj/result_rf.mat', mdict={'decodedState': decodedState})

