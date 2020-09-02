import sys
import os
import numpy as np
import re, string, unicodedata
import pickle
import nltk
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join
from bs4 import BeautifulSoup
from operator import mul
from functools import reduce
from collections import Counter

#nltk.download('stopwords')

# Regex defining what to consider as a word
word_regex = re.compile("[a-zA-Z']+(?:-[a-zA-Z']+)?")


cachedStopWords = stopwords.words("english")

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)


def make_Dictionary(emails,cnt):    
    all_words = []       
    for i in range(cnt):
        words = emails[i].split()
        all_words += words
    dictionary = Counter(all_words)
    list_to_remove = dictionary.keys()
    
    for item in list(list_to_remove):
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(4000)
    
     
    
    return dictionary
 

# Return a dict of messages with filenames for keys
# Return a dict of messages with filenames for keys
def get_messages(c_folder):
    messages = {}
    label={}
    cnt=0

    for x in (c_folder):

        folder1 = x+"/ham"
        folder2 = x+"/spam"
        for folder in [folder1,folder2]:
            encoding = sys.stdout.encoding
            filenames = [f for f in listdir(folder) if isfile(join(folder, f))]
        
        # Step through all files in folder
            for filename in filenames:
                path = folder + "/" + filename
        
            # Read file, ignoring invalid
                with open(path, encoding=encoding, errors="ignore") as message_file:
                # Add message to dict of messages
                    messages[cnt] = message_file.read()
                    if folder == folder1:
                        label[cnt]=0
                    else:
                        label[cnt]=1
                    cnt+=1

    return messages,label,cnt

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


def extract_features(emails,dictionary,cnt,label): 
    features_matrix = np.zeros((cnt,4000))
    for ii in range(cnt):
        words = emails[ii].split()

        for word in words:
            wordID = 0
            for i,d in enumerate(dictionary):
                if d[0] == word:
                    wordID = i
                    features_matrix[ii,wordID] = words.count(word)
    l=[]
    for key, value in label.items():
        l.append(value)
    #print(len(l))
    features_matrix = np.column_stack((features_matrix,np.array(l)) )
    return features_matrix

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
    
    print(len(emails))
    emails = denoise_text(emails,cnt)

    # load dict
    d = np.load("dict_enron_all.npy")
    test_feature = extract_features_test(emails,d,cnt)
    return test_feature


if __name__ == "__main__":
    # folder_name = ['enron1','enron2','enron3','enron4','enron5','enron6']
    # emails,label,cnt = get_messages(folder_name)
    # print(len(label))
    # emails = denoise_text(emails,cnt)
    # print(cnt)
    # save_obj(emails,"messages_all") 
    # save_obj(label,"label_all")

    # d = make_Dictionary(emails,cnt)
    # np.save('dict_enron_all.npy',d)
    # feature = extract_features(emails,d,cnt,label)
    # print(np.shape(feature))
    # np.save("fe_mat_all.npy",feature)

    test_feature = test_data()
    print(test_feature.shape)
    #print(np.sum(test_feature,axis=1))