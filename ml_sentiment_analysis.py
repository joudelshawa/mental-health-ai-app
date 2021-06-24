import textblob            #to import
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
tweets_processed = pd.read_csv('twitter_data.csv',lineterminator='\n')

# tweets = subset['Tweet Preprocessed'].values.tolist()
# categories = subset['Category'].values.tolist()

# res = []

# for column in subset.columns:
#     # Storing the rows of a column
#     # into a temporary list
#     li = subset[column].tolist()
#     # appending the temporary list
#     res.append(li)
# tweets_processed = res[1]
print(tweets_processed)

# Add polarities and subkectivities into the DataFrame by using TextBlob
tweets_processed["Polarity"] = tweets_processed["Tweet\r"].apply(lambda word:TextBlob(word).sentiment.polarity)
tweets_processed["Subjectivity"] = tweets_processed["Tweet\r"].apply(lambda word:TextBlob(word).sentiment.subjectivity)

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
# tweets_processed = tweets_processed.drop('Category', axis=1)

# Display the Polarity and Subjectivity Analysis
print(tweets_processed.head(10))

# Print the value counts of the Label column
print(tweets_processed[["Polarity Scores"]].value_counts())
print("\n")


# def percents(x):
#     return (x/len(tweets_processed))*100
#
# tweets_processed["Polarity Percentages"] = tweets_processed["Polarity Scores"].apply(percents)
# print(tweets_processed[["Polarity Percentages"]].value_counts())
print("Polarity Percentages:")
amount_neg = len(tweets_processed[tweets_processed["Polarity Scores"] == "Negative"])
amount_pos = len(tweets_processed[tweets_processed["Polarity Scores"] == "Positive"])
amount_neutral = len(tweets_processed[tweets_processed["Polarity Scores"] == "Neutral"])

# print(amount_neg, amount_pos, amount_neutral)
percent_neg = (amount_neg/len(tweets_processed))*100
percent_pos = (amount_pos/len(tweets_processed))*100
percent_neutral = (amount_neutral/len(tweets_processed))*100

print(f"Positive:\t{percent_pos}%")
print(f"Neutral:\t{percent_neutral}%")
print(f"Negative:\t{percent_neg}%")

# visualizing
values = [percent_pos,percent_neg, percent_neutral]
colors_religion=['#B2F3B5','#FFBEAF','#FFFB9A']
plt.pie(values, colors=colors_religion,frame=True)
centre_circle = plt.Circle((0,0),0.8,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
labels = ["Positive", "Negative", "Neutral"]
plt.legend(labels, loc="center")
plt.savefig('sentiment_twitter_graph.png', transparent=True)