import json
import csv

# with open('D:/OneDrive - The University of Western Ontario/Desktop/newsheadlineclassify/news_headline_category_dataset.json') as json_file:
#     jsondata = json.load(json_file)

data_file = open('D:/OneDrive - The University of Western Ontario/Desktop/newsheadlineclassify/jsonoutput.csv', 'w', newline='')
csv_writer = csv.writer(data_file)


tweets = []
for line in open('D:/OneDrive - The University of Western Ontario/Desktop/newsheadlineclassify/news_headline_category_dataset.json', 'r'):
    tweets.append(json.loads(line))

count = 0
for data in tweets:
    if count == 0:
        header = data.keys()
        csv_writer.writerow(header)
        count += 1
    try:
        csv_writer.writerow(data.values())
    except:
        print("this aint work")



data_file.close()