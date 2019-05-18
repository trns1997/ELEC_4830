import pandas as pd
import numpy
import scipy.io

#Load the mat file
mat = scipy.io.loadmat('ELEC4830_Final_project.mat')

#Remove NaN values from the training set
Y_raw = mat['trainState']
noNan = np.where(numpy.isnan(Y_raw) == False)
Y = Y_raw.take(noNan[1])
Y_raw.shape, Y.shape

#Average the historic data for every non NaN value in the training set
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

#Split the training data to make a training set and a validation set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

X_train.shape, Y_train.shape

# !pip install /content/drive/My\ Drive/xgboost-0.81-py2.py3-none-manylinux1_x86_64.whl

from xgboost import XGBClassifier

!pip install scikit-optimize

from skopt import BayesSearchCV

#Define search parameters for classifier optimization
xgb_opt = BayesSearchCV(
    XGBClassifier(tree_method='gpu_hist'),
    { 
        'max_depth': (3, 15),
        'gamma': (1e-6, 1e+1, 'log-uniform'),
        'learning_rate': (0.01, 0.4, 'log-uniform'),
        'min_child_weight': (1, 10),
        'subsample': (0.5, 1.0, 'log-uniform'),
        'colsample_bytree': (0.5, 1.0, 'log-uniform'),
        'n_estimators': (100, 1000)
    },
    n_iter=32,
    random_state=42,
    cv=3
)

xgb_opt.fit(X_train, Y_train)

xgb_opt.score(X_train, Y_train)

#Accuracy of the model on the validation set
y_pred = xgb_opt.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

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
# # print(X)

#Predict output and Save
# X_test = scaler.transform(X)
# decodedState = xgb_opt.predict(X_test)
# scipy.io.savemat('/content/drive/My Drive/4830_final_proj/result_xgb.mat', mdict={'decodedState': decodedState})

