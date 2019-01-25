
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression#optional
from sklearn.naive_bayes import MultinomialNB#optional
from sklearn.tree import DecisionTreeClassifier#optional
from sklearn.feature_extraction.text import CountVectorizer
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

raw = pd.read_csv('SMSSpamCollection.csv', 
                 delimiter = '\t+',
                 header = None, 
                 engine = 'python',
                 names = ("label", "message"),
                 encoding = 'latin-1', 
                 dtype = {'category': str, 'text': str} )

def csv_to_score(text_col, label_col):
   
    """This function takes two inputs: 
        1. the column of a pandas dataframe that contains the text, and 
        2. the column of the dataframe that has the correct label. 
        This function simply cleans the data and runs and evaluates one machine learning algorithm.
        It is good for choosing a general method such as logistic regression, tree methods, or naiive bayes
        to use as a start. After deciding on a simple method, ensemble methods and other tools can be added"""
    
    
    def text_process(text):
    
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
        words = ""

        for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
        return words

    text_feat = text_col.copy()
    text_feat = text_feat.apply(text_process)

    features_train, features_test, labels_train, labels_test = train_test_split(text_feat, label_col, test_size=0.25, random_state=80)

    nb_pipeline = Pipeline(steps=[
      ('tfidf', TfidfVectorizer(stop_words = "english")),
      ('nbclassifier', MultinomialNB())#***** can substitute for another algorithm *****
    ])

    nb_pipeline.fit(features_train, labels_train)

    pred = nb_pred_values = nb_pipeline.predict(features_test)
    #print(classification_report(labels_test, pred)))
    #return(confusion_matrix(labels_test, pred))
    return nb_pipeline.score(features_test,labels_test)


# In[ ]:


import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
from collections import Counter
from nltk.corpus import words

def add_features(text_col, dataframe):
    
    """
    This function takes the column of raw text and the dataframe it's contained in and outputs additional feature columns:
        1. number of sentences
        2. averge length of words
        3. proportion of english words
        4. boolean of whether the text is duplicated in another row
    """
    def number_sent(text):
        sentence_endings = r"[.,?,!]"
        return(len(re.split(sentence_endings,text)))
    #apply to dataframe
    dataframe['num_sent'] = text_col.apply(number_sent)
    
    s = set("abcdefghijklmnopqrstuwxyzABCDEFGHIJKLMNOPQUSTUVWXYZ")
    def avg_word_len(text):#gets the average number of letters in words
        num_words = len(text.split())
        letters = 0
        for w in text.split():
            if w.isalpha():
                letters += len(w)
        for l in w:
                if l in s:letters += 1
        return letters/num_words
    dataframe['avg_len'] = text_col.apply(avg_word_len)
    
    def prop_english(text):#finds the proportion of words used that are in the english dictionary
        engwords = words.words()
        english = 0
        num_words = len(text.split())
        for word in text.split():
            if word in engwords:
                english += 1
        return english/num_words
    dataframe['prop_eng'] = text_col.apply(prop_english)
    
    dataframe['is_dup'] = text_col.duplicated(keep=False)#tests whether a message is repeated or not
    return dataframe
add_features(raw["message"], raw)


# In[ ]:


from nltk.corpus import stopwords
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.probability import FreqDist 
import itertools
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

def top_words(text_col):
    
    """ 
    
    This function creates a visualization of the most frequent 25 terms in the document after cleaning it. 
    It takes in the column of raw text that you want to work with. Any subsets need to be defined before running the function.
    
    """
    
    def text_process(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
        words = ""

        for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
        return words

    text_col = text_col.apply(text_process)
    
    def split_to_words(text):
        words = text.split()
        return words 
    text_col = text_col.apply(split_to_words)
    
     #creates a list of words
    text_col = pd.Series.tolist(text_col)
    merged = list(itertools.chain(*text_col))
    merged = [x for x in merged if len(x) > 3]#words greater than 3 letters

    fdist = FreqDist(merged) #gets frequencies and as word and frequency for graphing
    x = fdist.most_common(25)
    word, frequency = zip(*x)
    indices = np.arange(len(x))
    
    plt.bar(indices, frequency, color='b')
    plt.xticks(indices, word, rotation='vertical')
    plt.tight_layout()
    return plt.show()

