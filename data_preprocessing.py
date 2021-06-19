# this is for twitter data cleaning!

# ## Import Libraries
# Loading all libraries to be used
import copy
import numpy as np
import re
import nltk
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import autocorrect

'''
CHECKLIST
remove all @
remove all urls
remove all non alphanumeric stuff
convert to lowercase
remove stopwords
'''


def preprocessing(tweet1):
    # print(tweet1)
    tweet1 = str(tweet1)
    tweet1 = tweet1.replace("\n", ' ')
    tweet1 = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)', '',
                    tweet1, flags=re.MULTILINE)  # to remove links that start with HTTP/HTTPS in the tweet
    # print(tweet1)
    tweet1 = re.sub(r'[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)', '', tweet1,
                    flags=re.MULTILINE)
    # print(tweet1)

    tweet2 = re.findall('@[A-Za-z]+[A-Za-z0-9-_]+', tweet1)
    # tweet2 = re.findall('@[A-Z][^A-Z]*', tweet1)
    # print(tweet2)
    if (len(tweet2) != 0):
        for i in range(len(tweet2)):
            tw = tweet2[i]
            tweet3 = re.findall('[A-Z][^A-Z]*', tw)
            t = ''
            for a in tweet3:
                t = t + a + " "
            tweet4 = t
            tweet1 = tweet1.replace(tweet2[i], tweet4)
    tweet1 = ' '.join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", '', tweet1).split())  # to remove #, @
    # tweet1 = ' '.join(re.sub("(@[A-Za-z0–9]+)|([0-9A-Za-z \t])|(\w+:\/\/\S+)",'',tweet1).split()) # to remove #, @
    # print("after removing @", tweet1)
    # tweet1 = " ".join(segment(tweet1))
    # print(tweet1)
    # print(tweet1)
    spelle = autocorrect.Speller('en')
    # tweet1 = ' '.join([spelle(w) for w in tweet1.split()]) # to check and correct spelling

    # print(tweet1)
    tweet1 = re.sub(r'\d', '', tweet1)  # to remove digits
    # print(tweet1)
    tweet1 = tweet1.lower()  # to lower the tweets
    return tweet1


# a = "@Brad_Pitt @WinonaRyder \n\nis amazing i LOVE her hahaha loooove bestie https://joudelshawa.com"
# print(preprocessing(a))

subset = pd.read_csv('all_data.csv',lineterminator='\n')
subset = subset[0:20]
subset['Tweet Preprocessed'] = subset['Tweet'].apply(preprocessing)


subset.to_csv('sample.csv',index=False)



