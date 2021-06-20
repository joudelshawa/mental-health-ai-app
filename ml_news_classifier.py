import pickle
import pandas as pd

subset = pd.read_csv('all_shuffled.csv',lineterminator='\n')

res = []

for column in subset.columns:
    # Storing the rows of a column into a temporary list
    subset[column] = subset[column].values.astype('U')
    li = subset[column].tolist()
    # appending the temporary list
    res.append(li)
tweets = res[1]
categories = res[0]

'''vectorization of data using bag of words'''
vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))

X = vectorizer.transform(iter(tweets))

encoder = pickle.load(open("encoder.pkl", 'rb'))
Y = encoder.transform(categories)

''''feature reduction !!!'''
# load selection from disk
selection = pickle.load(open("selection.pkl", 'rb'))
print("Number of features before reduction : ", X.shape[1])
X = selection.transform(X)
print("Number of features after reduction : ", X.shape[1])

# load the model from disk
loaded_model = pickle.load(open("randomforest.pkl", 'rb'))
result = loaded_model.score(X, Y)   # just change to predict when needed & don't have Y
print(result)
