# machine learning model on news category dataset
import copy
import numpy as np
import matplotlib.pyplot as plt
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
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

subset = pd.read_csv('all_shuffled.csv',lineterminator='\n')

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
title_tr, title_te, category_tr, category_te = train_test_split(tweets, categories, random_state=0)
title_tr, title_de, category_tr, category_de = train_test_split(title_tr, category_tr, random_state=0)
print("Training: ", len(title_tr))
print("Development: ", len(title_de),)
print("Testing: ", len(title_te))


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
print("Number of features before reduction : ", Xtr.shape[1], Xde.shape[1], Xte.shape[1])
selection = VarianceThreshold(threshold=0.001)
Xtr_whole = copy.deepcopy(Xtr)
Ytr_whole = copy.deepcopy(Ytr)
selection.fit(Xtr)
Xtr = selection.transform(Xtr)
Xde = selection.transform(Xde)
Xte = selection.transform(Xte)
print("Number of features after reduction : ", Xtr.shape[1], Xde.shape[1], Xte.shape[1])

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

''''training models!!!'''
highest_prec_model = ''
highest_prec_val = 0


# baseline model (dummy stratified classifier)
print("dummy classifier - stratified")
dc = DummyClassifier(strategy="stratified")
dc.fit(Xtr, Ytr)
pred = dc.predict(Xde)
print(classification_report(Yde, pred, target_names=encoder.classes_))
if (dc.score(Xde, Yde) > highest_prec_val):
    highest_prec_val = dc.score(Xde,Yde)
    highest_prec_model = 'dummy classifier'

# decision tree
print("decision tree")
dt = DecisionTreeClassifier(random_state=5)
dt.fit(Xtr, Ytr)
pred = dt.predict(Xde)
print(classification_report(Yde, pred, target_names=encoder.classes_))
if (dt.score(Xde, Yde) > highest_prec_val):
    highest_prec_val = dt.score(Xde,Yde)
    highest_prec_model = 'decision tree'

# random forest
print("random forest")
rf = RandomForestClassifier(n_estimators=40)
rf.fit(Xtr, Ytr)
pred = rf.predict(Xde)
print(classification_report(Yde, pred, target_names=encoder.classes_))
if (rf.score(Xde, Yde) > highest_prec_val):
    highest_prec_val = rf.score(Xde, Yde)
    highest_prec_model = 'random forest'

# multinomial naive bayesian
print("multinomial naive bayesian")
nb = MultinomialNB()
nb.fit(Xtr, Ytr)
pred = nb.predict(Xde)
print(classification_report(Yde, pred, target_names=encoder.classes_))
if (nb.score(Xde, Yde) > highest_prec_val):
    highest_prec_val = nb.score(Xde, Yde)
    highest_prec_model = 'naive bayes'

# support vector classification
print("support vector classification")
from sklearn.svm import SVC
svc = SVC()
svc.fit(Xtr, Ytr)
pred = svc.predict(Xde)
print(classification_report(Yde, pred, target_names=encoder.classes_))
if (svc.score(Xde, Yde) > highest_prec_val):
    highest_prec_val = svc.score(Xde, Yde)
    highest_prec_model = 'svc'

# multilayered perceptron
print("multilayered perceptron")
mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 20), random_state=1, max_iter=400)
mlp.fit(Xtr, Ytr)
pred = mlp.predict(Xde)
print(classification_report(Yde, pred, target_names=encoder.classes_))
if (mlp.score(Xde,Yde) > highest_prec_val):
    highest_prec_val = mlp.score(Xde, Yde)
    highest_prec_model = 'multilayered perceptron'

''''ADD BEST MODEL DEPENDING ON RESULTS'''
#maybe multinomial naive bayes??
'''predicting test data'''
pred = nb.predict(Xte)
print(classification_report(Yte, pred, target_names=encoder.classes_))
sns.heatmap(confusion_matrix(Yte, pred))


'''why multinomial naive bayes is good'''
nb1 = MultinomialNB()
nb1.fit(Xtr_whole, Ytr_whole)
coefs = nb1.coef_
target_names = encoder.classes_

# added this part idk
reverse_vocabulary = {}
vocabulary = vectorizer.vocabulary_

for word in vocabulary:
    index = vocabulary[word]
    reverse_vocabulary[index] = word
#

for i in range(len(target_names)):
    words = []
    for j in coefs[i].argsort()[-20:]:
        words.append(reverse_vocabulary[j])
    print (target_names[i], '-', words, "\n")
# should print out all the coefficents of the features
# and then print the top 20 words based on its weight.


print("final results")
print(highest_prec_val, "with", highest_prec_model)

# save the model to disk
filename = 'decisiontree.pkl'
pickle.dump(dt, open(filename, 'wb'))

filename = 'dummyclassifier.pkl'
pickle.dump(dc, open(filename, 'wb'))

filename = 'randomforest.pkl'
pickle.dump(rf, open(filename, 'wb'))

filename = 'svc.pkl'
pickle.dump(svc, open(filename, 'wb'))

filename = 'naivebayesian.pkl'
pickle.dump(nb, open(filename, 'wb'))

filename = 'perceptron.pkl'
pickle.dump(mlp, open(filename, 'wb'))

filename = 'selection.pkl'
pickle.dump(selection, open(filename, 'wb'))

filename = 'vectorizer.pkl'
pickle.dump(vectorizer, open(filename, 'wb'))

filename = 'encoder.pkl'
pickle.dump(encoder, open(filename, 'wb'))