import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import heapq
from operator import itemgetter

subset = pd.read_csv('twitter_data.csv',lineterminator='\n')
# subset = pd.read_csv('sample.csv',lineterminator='\n')

res = []

for column in subset.columns:
    # Storing the rows of a column into a temporary list
    subset[column] = subset[column].values.astype('U')
    li = subset[column].tolist()
    # appending the temporary list
    res.append(li)
tweets = res[0]
# tweets = res[1]
# categories = res[0]

'''vectorization of data using bag of words'''
vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))

X = vectorizer.transform(iter(tweets))

encoder = pickle.load(open("encoder.pkl", 'rb'))
# Y = encoder.transform(categories)

''''feature reduction !!!'''
# load selection from disk
selection = pickle.load(open("selection.pkl", 'rb'))
print("Number of features before reduction : ", X.shape[1])
X = selection.transform(X)
print("Number of features after reduction : ", X.shape[1])

# load the model from disk
loaded_model = pickle.load(open("decisiontree.pkl", 'rb'))
# result = loaded_model.score(X, Y)   # just change to predict when needed & don't have Y
# print(result)

pred = loaded_model.predict(X)
labels = list(set(pred))
counts = []
for label in labels:
    counts.append(np.count_nonzero(pred == label))

diction = {}
'''Print categories of test set (assume random forest is best)'''
print("Topics: ")
for num_label, label in enumerate(labels):
    # print(str((encoder.inverse_transform(labels)[num_label]) + r": " + str(pred.tolist().count(label))))
    # print('{:>12}  {:>12}'.format(str((encoder.inverse_transform(labels)[num_label]), str(pred.tolist().count(label)))))
    print('%-20s  %-4s' % ((encoder.inverse_transform(labels)[num_label]) + ':', (pred.tolist().count(label))))
    diction[str((encoder.inverse_transform(labels)[num_label]))] = pred.tolist().count(label)

num_of_labels = len(labels)


topitems = heapq.nlargest(7, diction.items(), key=itemgetter(1))  # Use .iteritems() on Py2
topitemsasdict = dict(topitems)

# print(topitemsasdict)

topics = topitemsasdict.keys()
values = topitemsasdict.values()

# visualizing
# labels = Nmaxelements(counts,7)
labels = values
# print(labels)
colors_religion=['#F3BEB2','#F3D5B2','#FFFB9A','#C7F3B2','#B2F3EF','#B2CCF3','#EAAFFF']
plt.pie(labels, colors=colors_religion,frame=True)
centre_circle = plt.Circle((0,0),0.8,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.legend(topics, loc="center")
plt.savefig('topics_twitter_graph.png', transparent=True)