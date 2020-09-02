#!/usr/bin/env python
# coding: utf-8

# if 1 then spam other wise non-spam

# In[2]:


# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
import sys
import os
import numpy as np
import re, string, unicodedata
import pickle
import nltk
import random

from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join
from bs4 import BeautifulSoup
from operator import mul
from functools import reduce
from collections import Counter


# from oauth2client.client import GoogleCredentials
# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)
# from google.colab import drive as dr
# dr.mount('/content/drive', force_remount=True)


# In[3]:


from tempfile import TemporaryFile 
# outfile = '/content/drive/My Drive/PrmlAssignment/PRML_assignment3/PRML_Assignment3/fe_mat_all.npy'
# savedata = np.load(outfile, allow_pickle=True)
# savedata = np.array(savedata)
#print(savedata.shape)


# In[ ]:

#generative
# np.random.seed(2332)
# print(train)
def training(trainData,alpha):
    train_x =trainData[:,0:-1]
    train_y = np.squeeze(trainData[:,-1:])
    # print(train_y.shape)

    spam = []
    non_spam = []
    for i in range(len(train_x)):
        if int(train_y[i]) == 1:
            spam.append(train_x[i])
        else :
            non_spam.append(train_x[i])
    spam = np.array(spam)
    non_spam = np.array(non_spam)
    print("non",non_spam.shape)
    
    spam_sum = np.sum(spam)
    print("spam",spam.shape)
    
    non_spam_sum= np.sum(non_spam)
    prob_spam = len(spam)/(len(train_x)*1.0)
    prob_non_spam = len(non_spam)/(len(train_x)*1.0)
    probability_of_feature_spam_class = np.add(np.sum(spam, axis=0),alpha)/(spam_sum + alpha*(len(train_x)))
    probability_of_feature_non_spam_class = np.add(np.sum(non_spam, axis = 0),alpha)/(non_spam_sum + alpha*(len(train_x)))
    return prob_spam,prob_non_spam,probability_of_feature_spam_class,probability_of_feature_non_spam_class

# prob_spam,prob_non_spam,probability_of_feature_spam_class,probability_of_feature_non_spam_class = training(train,alpha)
        
def testing(testData,prob_spam,prob_non_spam,probability_of_feature_spam_class,probability_of_feature_non_spam_class):
    
    prediction = []
    
    for data in testData:
        # prod_spam = prob_spam
        # prod_non_spam = prob_non_spam
        prod_spam = np.log(prob_spam)
        prod_non_spam = np.log(prob_non_spam)

        for a,i in zip(data,np.arange(len(probability_of_feature_spam_class))):
            # if int(a)!= 0:
                prod_spam= prod_spam  + int(a)*np.log(probability_of_feature_spam_class[i])
                prod_non_spam = prod_non_spam + int(a)*np.log(probability_of_feature_non_spam_class[i])
                # prod_spam= prod_spam * np.power(probability_of_feature_spam_class[i],int(a))
                # prod_non_spam = prod_non_spam*np.power(probability_of_feature_non_spam_class[i],int(a))
        if(prod_spam>prod_non_spam):
            prediction.append(1)
        else:
            prediction.append(0)
        
    return prediction



# testing(test_x,prob_spam,prob_non_spam,probability_of_feature_spam_class,probability_of_feature_non_spam_class)        

# In[5]:


# accuracy = -1
# for i in range(1):
#     for i in range(6):
#         np.random.shuffle(savedata)

    
#     #trainData = savedata[0:30000,:]
#     #testData = savedata[30001:,:]
#     test_x = testData
#     #test_y = np.squeeze(testData[:,-1:])
#     alpha = random.uniform(.001, 1.)
#     prob_spam,prob_non_spam,probability_of_feature_spam_class,probability_of_feature_non_spam_class = training(trainData,alpha)
#     p = testing(test_x,prob_spam,prob_non_spam,probability_of_feature_spam_class,probability_of_feature_non_spam_class) 
#     count = 0
#     for i in range(len(test_x)):
#         if int(p[i]) == int(test_y[i]):
#             count = count +1
#     acc = count/len(test_x)
#     print("accuracy",count/len(test_x))
#     if acc>accuracy:
#         accuracy = acc 
#         print(accuracy)
#         outfile1 = '/content/drive/My Drive/PrmlAssignment/PRML_assignment3/PRML_Assignment3/feature_spam_parameter.npy'
#         np.save(outfile1, probability_of_feature_spam_class)
#         outfile2 = '/content/drive/My Drive/PrmlAssignment/PRML_assignment3/PRML_Assignment3/feature_non_spam_paramter.npy'
#         np.save(outfile2, probability_of_feature_non_spam_class)
#         outfile3 = '/content/drive/My Drive/PrmlAssignment/PRML_assignment3/PRML_Assignment3/class_spam_paramter.npy'
#         np.save(outfile3, prob_spam)
#         outfile4 = '/content/drive/My Drive/PrmlAssignment/PRML_assignment3/PRML_Assignment3/class_non_spam_paramter.npy'
#         np.save(outfile4, prob_non_spam)
#         outfile5 = '/content/drive/My Drive/PrmlAssignment/PRML_assignment3/PRML_Assignment3/trainData.npy'
#         np.save(outfile5, trainData)
#         outfile6 = '/content/drive/My Drive/PrmlAssignment/PRML_assignment3/PRML_Assignment3/testData.npy'
#         np.save(outfile6, testData)

   


# In[6]:



# outfile1 =  '/content/drive/My Drive/PrmlAssignment/PRML_assignment3/PRML_Assignment3/feature_spam_parameter.npy'
# probability_of_feature_spam_class = np.load(outfile1, allow_pickle=True)
# probability_of_feature_spam_class = np.array(probability_of_feature_spam_class)
# outfile2 =  '/content/drive/My Drive/PrmlAssignment/PRML_assignment3/PRML_Assignment3/feature_non_spam_paramter.npy'
# probability_of_feature_non_spam_class = np.load(outfile2, allow_pickle=True)
# probability_of_feature_non_spam_class = np.array(probability_of_feature_non_spam_class)
# outfile3 =  '/content/drive/My Drive/PrmlAssignment/PRML_assignment3/PRML_Assignment3/class_spam_paramter.npy'
# prob_spam = np.load(outfile3, allow_pickle=True)
# prob_spam = np.array(prob_spam)
# outfile4 =  '/content/drive/My Drive/PrmlAssignment/PRML_assignment3/PRML_Assignment3/class_non_spam_paramter.npy'
# prob_non_spam = np.load(outfile4, allow_pickle=True)
# prob_non_spam = np.array(prob_non_spam)
# outfile5 = '/content/drive/My Drive/PrmlAssignment/PRML_assignment3/PRML_Assignment3/trainData.npy'
# traindata = np.load(outfile5, allow_pickle=True)
# outfile6 = '/content/drive/My Drive/PrmlAssignment/PRML_assignment3/PRML_Assignment3/testData.npy'
# testData = np.load(outfile6, allow_pickle=True)
# train_x = traindata[:,0:-1]
# train_y = np.squeeze(traindata[:,-1:])
# test_x = testData[:,0:-1]
# test_y = np.squeeze(testData[:,-1:])
# counts = 0
# p = testing(train_x,prob_spam,prob_non_spam,probability_of_feature_spam_class,probability_of_feature_non_spam_class) 
# for i in range(len(train_x)):
#     if int(p[i]) == int(train_y[i]):
#         counts = counts +1

# print("train_acc",counts/len(train_x))
# counts = 0
# p = testing(test_x,prob_spam,prob_non_spam,probability_of_feature_spam_class,probability_of_feature_non_spam_class) 
# for i in range(len(test_x)):
#     if int(p[i]) == int(test_y[i]):
#         counts = counts +1

# print("test_acc",counts/len(test_x))




#nltk.download('stopwords')

# Regex defining what to consider as a word
word_regex = re.compile("[a-zA-Z']+(?:-[a-zA-Z']+)?")


cachedStopWords = stopwords.words("english") 


# Read test data.
def get_messages_test(folder):
    messages = {}

    cnt=0
    encoding = sys.stdout.encoding
    filenames = [f for f in listdir(folder) if isfile(join(folder, f))]

# Step through all files in folder
    for filename in filenames:
        path = folder + "/" + filename

    # Read file, ignoring invalid
        with open(path, encoding=encoding, errors="ignore") as message_file:
        # Add message to dict of messages
            messages[cnt] = message_file.read()
            cnt+=1

    return messages,cnt


def extract_features_test(emails,dictionary,cnt): 
    features_matrix = np.zeros((cnt,4000))
    for ii in range(cnt):
        words = emails[ii].split()

        for word in words:
            wordID = 0
            for i,d in enumerate(dictionary):
                if d[0] == word:
                    wordID = i
                    features_matrix[ii,wordID] = words.count(word)

    return features_matrix

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def remove_words_contains_numbers(text):
    return re.sub(r'\w*\d\w*', '', text).strip()

def remove_punctuation(text):
    result = text.translate(str.maketrans('', '', string.punctuation))
    #result = text.translate(string.dir("",""), string.punctuation)
    return result

def remove_blankspaces(text):
    result =  " ".join(text.split())
    return result

def lower(text):
    return text.lower()

def remove_stopwords(text):
    result = ' '.join([word for word in text.split() if word not in cachedStopWords])
    return result


def denoise_text(emails,cnt):
    for i in range(cnt):
        text = emails[i]
        text = strip_html(text)
        text = remove_between_square_brackets(text)
        text = remove_punctuation(text)
        text = remove_words_contains_numbers(text)
        text = remove_blankspaces(text)
        text = lower(text)
        text = remove_stopwords(text)
        emails[i]=text
    return emails

def test_data():
    folder_name = "test"
    emails,cnt = get_messages_test(folder_name)
    
    #print(len(emails))
    emails = denoise_text(emails,cnt)

    # load dict
    d = np.load("dict_enron_all.npy")
    test_feature = extract_features_test(emails,d,cnt)
    return test_feature

if __name__ == "__main__":

    outfile1 =  'feature_spam_parameter.npy'
    probability_of_feature_spam_class = np.load(outfile1, allow_pickle=True)
    probability_of_feature_spam_class = np.array(probability_of_feature_spam_class)
    outfile2 =  'feature_non_spam_paramter.npy'
    probability_of_feature_non_spam_class = np.load(outfile2, allow_pickle=True)
    probability_of_feature_non_spam_class = np.array(probability_of_feature_non_spam_class)
    outfile3 =  'class_spam_paramter.npy'
    prob_spam = np.load(outfile3, allow_pickle=True)
    prob_spam = np.array(prob_spam)
    outfile4 =  'class_non_spam_paramter.npy'
    prob_non_spam = np.load(outfile4, allow_pickle=True)
    prob_non_spam = np.array(prob_non_spam)


    test_feature = test_data()
    result = testing(test_feature,prob_spam,prob_non_spam,probability_of_feature_spam_class,probability_of_feature_non_spam_class)
    print(result)



















# In[7]:


'''

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(trainData[:,0:-1], np.squeeze(trainData[:,-1:]))

p = clf.predict(train_x)
counts = 0
for i in range(len(train_x)):
    if int(p[i]) == int(train_y[i]):
        counts = counts +1

print("train_acc",counts/len(train_x))
p = clf.predict(test_x)
counts = 0
for i in range(len(test_x)):
    if int(p[i]) == int(test_y[i]):
        counts = counts +1

print("test_acc",counts/len(test_x))


# In[8]:


alpha


# In[9]:


#discriminative
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial', max_iter = 500).fit(trainData[:,0:-1], np.squeeze(trainData[:,-1:]))
p = clf.predict(train_x)
counts = 0
for i in range(len(train_x)):
    if int(p[i]) == int(train_y[i]):
        counts = counts +1

print("train_acc",counts/len(train_x))
p = clf.predict(test_x)
counts = 0
for i in range(len(test_x)):
    if int(p[i]) == int(test_y[i]):
        counts = counts +1

print("test_acc",counts/len(test_x))


# In[10]:


#discrimanative
from sklearn.svm import SVC
clf = SVC(gamma='auto',kernel='linear')
clf.fit(trainData[:,0:-1], np.squeeze(trainData[:,-1:])) 
p = clf.predict(train_x)
counts = 0
for i in range(len(train_x)):
    if int(p[i]) == int(train_y[i]):
        counts = counts +1

print("train_acc",counts/len(train_x))
p = clf.predict(test_x)
counts = 0
for i in range(len(test_x)):
    if int(p[i]) == int(test_y[i]):
        counts = counts +1

print("test_acc",counts/len(test_x))


# In[11]:


#Discriminative
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(trainData[:,0:-1], np.squeeze(trainData[:,-1:])) 
p = clf.predict(train_x)
counts = 0
for i in range(len(train_x)):
    if int(p[i]) == int(train_y[i]):
        counts = counts +1

print("train_acc",counts/len(train_x))
p = clf.predict(test_x)
counts = 0
for i in range(len(test_x)):
    if int(p[i]) == int(test_y[i]):
        counts = counts +1

print("test_acc",counts/len(test_x))


# In[12]:


#discriminative
from sklearn.linear_model import Perceptron

clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(trainData[:,0:-1], np.squeeze(trainData[:,-1:])) 
p = clf.predict(train_x)
counts = 0
for i in range(len(train_x)):
    if int(p[i]) == int(train_y[i]):
        counts = counts +1

print("train_acc",counts/len(train_x))
p = clf.predict(test_x)
counts = 0
for i in range(len(test_x)):
    if int(p[i]) == int(test_y[i]):
        counts = counts +1

print("test_acc",counts/len(test_x))


# In[13]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis()
clf.fit(trainData[:,0:-1], np.squeeze(trainData[:,-1:])) 
p = clf.predict(train_x)
counts = 0
for i in range(len(train_x)):
    if int(p[i]) == int(train_y[i]):
        counts = counts +1

print("test_acc",counts/len(train_x))
p = clf.predict(test_x)
counts = 0
for i in range(len(test_x)):
    if int(p[i]) == int(test_y[i]):
        counts = counts +1

print("test_acc",counts/len(test_x))

'''