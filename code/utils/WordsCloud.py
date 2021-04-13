"""
WordCloud

This file can create word cloud based on the TF-IDF frequency of a 
corpus

@see config.py

@author José Antonio García-Díaz <joseantonio.garcia8@um.es>
@author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import string
import pandas as pd
import matplotlib.pyplot as plt
import os

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from wordcloud import WordCloud


class WordsCloud ():
    """
    WordCloudGenerator
    
    """

    def __init__ (self, dataset, max_words = 100):
        """
        @param dataset dataset
        @param max_words int
        @param text_column String
        """
        super().__init__()
        
        self.dataset = dataset
        self.max_words = max_words

    def generate (self):
        """
        """
        
        # @var df DataFrame
        df = self.dataset.get ()
        
        
        # @var vectorizer Create vectorizer Based on TFIDF
        vectorizer = TfidfVectorizer (stop_words = stopwords.words ('spanish'), max_features = self.max_words)
        
        
        # @var vecs Create vectorizer Based on TFIDF
        vecs = vectorizer.fit_transform (df['tweet'])
        
        
        # @var feature_names
        feature_names = vectorizer.get_feature_names ()
    
    
        # @var df_frequencies
        df_frequencies = pd.DataFrame (vecs.todense ().tolist (), columns = feature_names)
        
        
        # Generate a word cloud
        wordcloud = WordCloud (background_color = 'white', max_words = self.max_words).generate_from_frequencies (df_frequencies.T.sum (axis = 1))
    
    
        # Generate the image
        plt.imshow (wordcloud, interpolation = 'bilinear')
        plt.axis ("off")
        
        return plt
    