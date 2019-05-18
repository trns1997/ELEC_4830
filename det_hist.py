import numpy
import scipy.io
import pandas as pd

#Load the mat file
mat = scipy.io.loadmat('ELEC4830_Final_project.mat')

Y_raw = mat['trainState']

#Find the indices of the non NaN values from the training set
noNan = numpy.where(numpy.isnan(Y_raw) == False)

#Find average distance between 2 consecutive non NaN values
counter = numpy.array([97])
for i in range(5, len(noNan[1]), 5):
  counter = numpy.append(counter, noNan[1][i] - noNan[1][i-1])
  
print("Average Gap:",numpy.mean(counter))