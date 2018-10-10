#Data Processing package
import  pandas as pd
import numpy as np
import re

#Visiualization package
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#language Processing package
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import RegexpTokenizer
import tokenize
from nltk.stem.snowball import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#feature extraction package
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#ML algorithm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import itertools

def prepare_data(file_dir):
    df = pd.read_csv(file_dir)

    #drop rows contain null value in lyrics and genre
    df = df.dropna(subset=['lyrics'])
    df = df[df.lyrics != ""]
    df = df[df.genre != ""]

    # decoding
    df.genre.str.decode('utf-8')
    df.lyrics.str.decode('utf-8')

    # strip excessive blank and convert all the string to lower type
    df.lyrics = df.lyrics.str.strip().str.lower()
    df.genre = df.genre.str.strip().str.lower()

    # get rid of \n,-,none char string
    df.lyrics.replace("\n", " ", regex=True, inplace=True)
    df.lyrics.replace("-", " ", regex=True, inplace=True)
    df.lyrics.replace("[^a-zA-Z0-9_\ ]", "", regex=True, inplace=True)
    df.lyrics.replace('"', "", regex=True, inplace=True)

    df.genre.replace("\n", " ", regex=True, inplace=True)
    df.genre.replace("-", " ", regex=True, inplace=True)
    df.genre.replace("[^a-zA-Z0-9_\ ]", "", regex=True, inplace=True)
    df.genre.replace('"', "", regex=True, inplace=True)

    return df
def data_summary(df):
    genre_list=list(set(df.genre))
    year_list=list(set(df.year))

    return genre_list,year_list,df.shape

