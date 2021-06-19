import pickle

import nltk
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import copy


subset = pd.read_csv('sample_1000_shuffled.csv',lineterminator='\n')

# tweets = subset['Tweet Preprocessed'].values.tolist()
# categories = subset['Category'].values.tolist()

res = []

for column in subset.columns:
    # Storing the rows of a column
    # into a temporary list
    subset[column] = subset[column].values.astype('U')
    li = subset[column].tolist()
    # appending the temporary list
    res.append(li)
tweets = res[1]
categories = res[0]

# print(res)

# print(tweets)
# print(categories)

# '''splitting data in 3 parts'''
title_tr, title_te, category_tr, category_te = train_test_split(tweets, categories)
title_tr, title_de, category_tr, category_de = train_test_split(title_tr,category_tr)
print("Training: ",len(title_tr))
print("Developement: ",len(title_de),)
print("Testing: ",len(title_te))

'''skipped word cloud stuff'''

# "vectorization of data using bag of words"
tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
stop_words = nltk.corpus.stopwords.words("english")
vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize, stop_words=stop_words)

# title_tr = title_tr
# x = v.fit_transform(df['Review'].values.astype('U'))  ## Even astype(str) would work
vectorizer.fit(iter(title_tr))
Xtr = vectorizer.transform(iter(title_tr))
Xde = vectorizer.transform(iter(title_de))
Xte = vectorizer.transform(iter(title_te))

encoder = LabelEncoder()
encoder.fit(category_tr)
Ytr = encoder.transform(category_tr)
Yde = encoder.transform(category_de)
Yte = encoder.transform(category_te)

''''feature reduction !!!'''
# print("Number of features before reduction : ", Xte.shape[1])
# selection = VarianceThreshold(threshold=0.001)
# Xtr_whole = copy.deepcopy(Xtr)
# Ytr_whole = copy.deepcopy(Ytr)
# selection.fit(Xtr)
# Xtr = selection.transform(Xtr)
# Xde = selection.transform(Xde)
# Xte = selection.transform(Xte)
# print("Number of features after reduction : ", Xtr.shape[1])

'''''sampling data'''
# count number of diff labels in dataset and plot
labels = list(set(Ytr))
counts = []
for label in labels:
    counts.append(np.count_nonzero(Ytr == label))
plt.pie(counts, labels=labels, autopct='%1.1f%%')
# plt.show()

''''data not uniformly distributed so oversampling!!'''
sm = SMOTE(random_state=42)
Xtr, Ytr = sm.fit_resample(Xtr, Ytr)
labels = list(set(Ytr))
counts = []
for label in labels:
    counts.append(np.count_nonzero(Ytr == label))
plt.pie(counts, labels=labels, autopct='%1.1f%%')
# plt.show()
# print(title_tr)
# print(Xtr)
# Xte = selection.transform(Xte)
# Yte = selection.transform(Yte)

from sklearn.feature_selection import SelectKBest

# # This models a statistical test known as ANOVA
#
from sklearn.feature_selection import f_classif

k_best = SelectKBest(f_classif, k=3089)

k_best.fit_transform(Xte, Yte)

Xte = k_best.transform(Xte)
# Yte = k_best.transform(Yte)

print("Number of features after reduction : ", Xte.shape[1])

# load the model from disk
loaded_model = pickle.load(open("finalized_model.pkl", 'rb'))
print(loaded_model.predict(Xte))
result = loaded_model.score(Xte, Yte)
print(result)