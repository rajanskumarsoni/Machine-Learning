

#Question 1(i)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import genfromtxt
my_data = pd.read_csv('/Datasets/Dataset1.csv', delimiter=',')

# scatter plot of data
def plot_the_data(my_data):
  x = my_data[:,0]
  y = my_data[:,1]
  plt.scatter(x,y)
  plt.title("Scatter plot of dataset-1")
  plt.xlabel("Dimension x")
  plt.ylabel("dimension y")
  plt.show()
  
plot_the_data(my_data)
  
#Question 2(ii)
#FINDING MEAN  AND VARIANCE
#counting number of data points
def count_data_points(my_data):
  number_of_data_points = 0
  for data in my_data:
    number_of_data_points = number_of_data_points + 1
  return number_of_data_points

# finding mean
def find_mean(my_data,number_of_data_points):
  sum_col_0 = 0
  sum_col_1 = 0
  for data in my_data:
    sum_col_0 = sum_col_0 + data[0]
    sum_col_1 = sum_col_1 + data[1]
  mean = [float(sum_col_0)/number_of_data_points,float(sum_col_1)/number_of_data_points ]
  return mean

# finding covariance
def covariance(X,Y,number_of_data_points,mean):
  sum_for_covariance = 0
  for x,y in zip(X,Y):
    sum_for_covariance = sum_for_covariance + (x-mean[0])*(y-mean[1])
  # print(sum_for_covariance)
  covariance_XY = float(sum_for_covariance)/number_of_data_points
  return covariance_XY

# finding covariance matrix
def find_covariance_matrix(my_data,number_of_data_points,mean):


  covariance_XY = covariance(my_data[:,0],my_data[:,1],number_of_data_points,mean)
  variance_X = covariance(my_data[:,0],my_data[:,0],number_of_data_points,mean)
  variance_Y = covariance(my_data[:,1],my_data[:,1],number_of_data_points,mean)

  covariance_matrix = [[variance_X,covariance_XY],
                      [covariance_XY,variance_Y]]
  return covariance_matrix
  
# finding parameters
def find_parameters(my_data):
  number_of_data_points = count_data_points(my_data)
  mean = find_mean(my_data,number_of_data_points)
  covariance_matrix = find_covariance_matrix(my_data,number_of_data_points,mean)
  return mean,covariance_matrix


parameters = find_parameters(my_data)
print(parameters)



#Question 3(3)
#Log likelihood of the given data on parameters
import math
def likelihood_function(data,parameter):
  mean = parameter[0]
  covariance_matrix = parameter[1]
  inverse_covariance_matrix = np.linalg.inv(covariance_matrix)
  totalsum = 0
  for i  in range(len(data)):
    x = np.array(np.subtract(data[i],mean)).T
    y = np.array(np.subtract(data[i],mean))
    z = np.dot(x,inverse_covariance_matrix)

    totalsum = totalsum + np.dot(z,y)
  a = np.power(2*np.pi,2)*np.linalg.det(covariance_matrix)
  constant_term = len(data)*np.log(a)
  likelihood = -0.5*(totalsum + constant_term)
  return likelihood


likelihood = likelihood_function(my_data,parameters)
print(likelihood)

#Question 1(iv)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def plot_log_likelihood(my_data):
  fig = plt.figure(figsize=(15,10))
  ax = fig.gca(projection='3d')

  # Make data.
  X = np.arange(-10, 10.5, 0.5)
  Y = np.arange(-10, 10.5, 0.5)
  XX, YY = np.meshgrid(X, Y)
  covariance = [[1,0],[0,1]]
  myi = -1000000000
  
  z_final = []
  a = 0
  b = 0
  for i in X:
    z_semi= []
    for j in Y:
      mean = [i,j]
      
      parameters = (mean,covariance)
      z = likelihood_function(my_data,parameters)
      if(myi<z):
        a = i
        b =j
        myi = z

      z_semi.append(z)
    z_final.append(z_semi)
  
  print(a,b,myi)
    
      
  # Plot the surface.
  surf = ax.plot_surface(np.array(XX), np.array(YY), np.array(z_final), cmap=cm.coolwarm,
                         linewidth=0, antialiased=False)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title("Log Likelihood ")
  plt.xlabel("x component of mean")
  plt.ylabel("y component of mean")

  plt.show()


plot_log_likelihood(my_data)

