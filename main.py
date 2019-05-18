#! /usr/bin/env python3
'''
How to RUN the Code:
cd PATH_TO_THIS_FILE'S_DIRECTORY
chmod +x main.py
./main.py
'''

import numpy
import scipy.io

#Load the mat file
mat = scipy.io.loadmat('ELEC4830_Final_project.mat')

#Remove NaN values from the training set
Y_raw = mat['trainState']
noNan = numpy.where(numpy.isnan(Y_raw) == False)
Y = Y_raw.take(noNan[1])
Y_raw.shape, Y.shape

#Average the historic data for every non NaN value in the training set
X_raw =  mat['trainSpike']
# X = X_raw[:,noNan[1]]
X = numpy.array([])
prev_data = 10
X = numpy.mean(X_raw[:, noNan[1][0]-prev_data:noNan[1][0]], axis = 1)[numpy.newaxis].T
for i in range(1, len(noNan[1])):
    a = numpy.mean(X_raw[:, noNan[1][i]-prev_data:noNan[1][i]], axis = 1)[numpy.newaxis].T
    X = numpy.append(X, a, axis = 1)
X_raw.shape, X.shape

X = X.T
Y = Y.T
X.shape, Y.shape

from sklearn.model_selection import train_test_split

#Split the training data to make a training set and a validation set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

X_train.shape, Y_train.shape

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier

error = []

# Calculating error for K values between 1 and 60
for i in range(1, 60):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    pred_i = knn.predict(X_test)
    error.append(numpy.mean(pred_i != Y_test))

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 60), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')

classifier = KNeighborsClassifier(n_neighbors=30)  
classifier.fit(X_train, Y_train)

#Accuracy of the model on the validation set
y_pred = classifier.predict(X_test)  
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(Y_test, y_pred))  
print(classification_report(Y_test, y_pred))

# import matplotlib.pyplot as plt
# fig=plt.figure(figsize=(200, 10), dpi= 80, facecolor='w', edgecolor='k')
# plt.plot(Y_test, 'X')
# plt.plot(y_pred, '*')
# plt.show()

#Load the testSpike data
X_test = mat['testSpike']
X_test.shape

#Pre-processing of the testSpike data
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
X = X.T
X.shape
# print(X)

#Predict output and Save
# X_test = scaler.transform(X)
# decodedState = classifier.predict(X_test)
# scipy.io.savemat('/content/drive/My Drive/4830_final_proj/result_knn.mat', mdict={'decodedState': decodedState})
