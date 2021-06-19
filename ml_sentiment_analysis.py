import textblob            #to import
from textblob import TextBlob
import pandas as pd

tweets_processed = pd.read_csv('sample_shuffled.csv',lineterminator='\n')

# tweets = subset['Tweet Preprocessed'].values.tolist()
# categories = subset['Category'].values.tolist()

res = []

# for column in subset.columns:
#     # Storing the rows of a column
#     # into a temporary list
#     li = subset[column].tolist()
#     # appending the temporary list
#     res.append(li)
# tweets_processed = res[1]

# Add polarities and subkectivities into the DataFrame by using TextBlob
tweets_processed["Polarity"] = tweets_processed["Tweet Preprocessed"].apply(lambda word:TextBlob(word).sentiment.polarity)
tweets_processed["Subjectivity"] = tweets_processed["Tweet Preprocessed"].apply(lambda word:TextBlob(word).sentiment.subjectivity)

# Display the Polarity and Subjectivity columns
# display(tweets_processed[["Polarity","Subjectivity"]].head(10))

# Define a function to classify polarities
def analyse_polarity(polarity):
    if polarity > 0:
        return "Positive"
    if polarity == 0:
        return "Neutral"
    if polarity < 0:
        return "Negative"
# Apply the funtion on Polarity column and add the results into a new column
tweets_processed["Polarity Scores"] = tweets_processed["Polarity"].apply(analyse_polarity)
tweets_processed = tweets_processed.drop('Category', axis=1)

# Display the Polarity and Subjectivity Analysis
print(tweets_processed.head(10))

# Print the value counts of the Label column
print(tweets_processed[["Polarity Scores"]].value_counts())

amount_neg = len(tweets_processed[tweets_processed["Polarity Scores"] == "Negative"])
amount_pos = len(tweets_processed[tweets_processed["Polarity Scores"] == "Positive"])
amount_neutral = len(tweets_processed[tweets_processed["Polarity Scores"] == "Neutral"])

# print(amount_neg, amount_pos, amount_neutral)
percent_neg = (amount_neg/len(tweets_processed))*100
percent_pos = (amount_pos/len(tweets_processed))*100
percent_neutral = (amount_neutral/len(tweets_processed))*100

print(percent_neg, percent_pos, percent_neutral)

#print(tweets_processed.rename(columns={'Label':'Polarity Scores'}, inplace=True))
#This is the Label

# print(tweets_processed[["Polarity Scores"] == "Positive"])