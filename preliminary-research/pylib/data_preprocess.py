import pandas as pd
import numpy as np
!pip install wget
!wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
!wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv
!wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv
ge1 = pd.read_csv("data/full_dataset/goemotions_1.csv")
ge2 = pd.read_csv("data/full_dataset/goemotions_2.csv")
ge3 = pd.read_csv("data/full_dataset/goemotions_3.csv")
data=[ge1, ge2, ge3]
dataset=pd.concat(data)
dataset.head()