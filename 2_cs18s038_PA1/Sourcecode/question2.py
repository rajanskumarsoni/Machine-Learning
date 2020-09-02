
#Question 2(1)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import genfromtxt
my_data = pd.read_csv('/Datasets/Dataset2.csv', delimiter=',')

mydata = my_data
# print(my_data)
from sklearn.cluster import KMeans
import math
import sys

sys.setrecursionlimit(2000)


def findkmeans(mydata,clusters):
  maxvalue = np.amax(mydata)
  minvalue = np.amin(mydata)
  distancelimit = .001
  centers = np.arange(minvalue,maxvalue,(maxvalue-minvalue)/clusters)*1.0
  sum_distance_betweeen_centers = 45
  count  = 0
  while(sum_distance_betweeen_centers>distancelimit):
    
    
    count = count +1
    binarylist = []
    for i in range(len(mydata)):
      distance = []
      for k in range(clusters):
        distance.append((mydata[i]-centers[k])*(mydata[i]-centers[k]))
      r = np.zeros(clusters)
      p = np.argmin(distance)
    
      r[p] = mydata[i]
      
      binarylist.append(r)
    
    binarylist = np.array(binarylist)
    
    
    new_center = []
    for i in range(clusters):
      a = sum(binarylist[:,i])
      b = np.count_nonzero(binarylist[:,i])
      new_center.append(a/b)
    sum_distance_betweeen_centers = 0
    for i in range(clusters):
      sum_distance_betweeen_centers = sum_distance_betweeen_centers  + (centers[i]-new_center[i])*(centers[i]-new_center[i])
   
    centers = new_center
    
  return centers


def likelihood_function(data_point,mean,variance):
  
  a = pow(variance*2*math.pi, 1/2.0)
  normal_part = 1.0/(a)
  p = (data_point-mean)**2
  exponent_part = math.exp(-p/(2*variance))
  r = normal_part*exponent_part
  
  return r


def initialization(mydata,clusters):
  means = findkmeans(mydata,clusters)
  variance = np.random.randint(10,100, size=clusters)*1.0
  
  prior = variance/sum(variance)
  return means, variance, prior
  
def log_likelihood_data(mydata, mean,variance, prior,clusters) :
  log_likelihood_sum = 0
  for i in range(len(mydata)):
    sum = 0
    for j in range(clusters):
      sum =sum + prior[j]*likelihood_function(mydata[i],mean[j],variance[j])
    log_likelihood_sum = log_likelihood_sum  + np.log(sum)
  return log_likelihood_sum
    
def calculate_new_parameter(mydata, mean, variance, prior,clusters):
    for i in range(len(variance)):
      if variance[i]<.0001:
        variance[i] = 5.0
    
 
    matrix_of_responsibilities = []
    for datapoint in mydata:
      responsibility_of_each_class_for_datapoint= []
      for k in range(clusters):
        responsibility = likelihood_function(datapoint,mean[k],variance[k])*prior[k]
        responsibility_of_each_class_for_datapoint.append(responsibility)
      suma = np.sum(responsibility_of_each_class_for_datapoint)
      matrix_of_responsibilities.append(responsibility_of_each_class_for_datapoint/suma)
    
   

    #finding mean of each cluster
    vector_of_reponsibility_of_each_cluster = np.sum(matrix_of_responsibilities,axis=0)
    mean_vector_of_clusters = []
    for i in range(clusters):
      sum =0
      for j in range(len(mydata)):
        sum = sum + mydata[j]*matrix_of_responsibilities[j][i]
      mean_vector_of_clusters.append(sum/vector_of_reponsibility_of_each_cluster[i])
    

    #findinding variance
    variance_vector_of_clusters = []
    for i in range(clusters):
      sum = 0
      for j in range(len(mydata)):
        sum = sum + matrix_of_responsibilities[j][i]*(my_data[j]-mean_vector_of_clusters[i])*(my_data[j]-mean_vector_of_clusters[i])
        
      variance_vector_of_clusters.append(sum/vector_of_reponsibility_of_each_cluster[i])
    
    

    #prior of clusters
    prior_vector_of_clusters = []
    for i in range(clusters):
      x = vector_of_reponsibility_of_each_cluster[i]/len(mydata)
      prior_vector_of_clusters.append(x)
    

    return mean_vector_of_clusters, variance_vector_of_clusters,prior_vector_of_clusters
  
def shouldIstop(before,after,convergence_limit):
  difference = after - before
  if convergence_limit  > difference:
    return True
  else :
    return False

count = 0
hoodlist = []
countlist = []
def EMAlgorithm(mydata, mean,variance, prior,clusters,convergence_limit):
  global  count
  global hoodlist
  global countlist
  count = count +1
  before = log_likelihood_data(mydata, mean,variance, prior,clusters)
  hoodlist.append(before)
  countlist.append(count)
  new_means ,new_variance, new_prior = calculate_new_parameter(mydata, mean, variance, prior,clusters)
  after = log_likelihood_data(mydata, new_means,new_variance, new_prior,clusters)
  if shouldIstop(before,after,convergence_limit) :
      hoodlist.append(after)
      countlist.append(count+1)
      return new_means ,new_variance, new_prior
  else:
      return EMAlgorithm(mydata, new_means,new_variance, new_prior,clusters,convergence_limit)
      
      
      
def run_the_algorithm(mydata,clusters,convergence_limit):
  means, variance, prior = initialization(mydata,clusters)
  new_means ,new_variance, new_prior = EMAlgorithm(mydata, means,variance, prior,clusters,convergence_limit)
  return new_means ,new_variance, new_prior
  


#question2(ii)
clusters = 10
convergence_limit = .001

means = []
variances = []
priors = []
for cluster in range(clusters):
 
    new_means ,new_variance, new_prior = run_the_algorithm(mydata,cluster +1,convergence_limit)
    means.append(new_means)
    variances.append(new_variance)
    priors.append(new_prior)


#question2(iii)
import matplotlib.pyplot as plt

x_axis = []
y_axis = []
for i in range(clusters):
  s = log_likelihood_data(mydata, means[i],variances[i], priors[i],i +1)
  y_axis.append(s)
  x_axis.append(i+1)
print(x_axis)
plt.title("Plot of Log Likelihood of data with respect to number of clusters")
plt.xlabel("no. of clusters")
plt.ylabel("Log likelihood")
plt.plot(x_axis,y_axis)
plt.show()