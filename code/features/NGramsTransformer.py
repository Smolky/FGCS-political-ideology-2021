import pandas as pd
import pickle
import os.path

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator 
from nltk.corpus import stopwords

class NGramsTransformer (BaseEstimator, TransformerMixin):
    """
    Obtain word-n-gram and char-n-gram features
    
    Inspired in
    @link https://github.com/pan-webis-de/daneshvar18/blob/5542895062f2404fd5b5a07493ff098132308457/pan18ap/train_model.py

    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
    """

    def __init__ (self, cache_file = '', field = 'tweet_clean_lowercase', max_features = 100):
        """
        @param model String (see Config)
        @param cache_file String
        """
        super().__init__()
        
        self.cache_file = cache_file
        self.field = field
        self.columns = None
        self.ngrams_vectorizer = None
        self.max_features = max_features
    
    def fit (self, X, y = None ):

        # @var word vectorizer TfidfVectorizer Generate a new word vectorizer
        word_vectorizer = TfidfVectorizer (
            analyzer = 'word',
            lowercase = True,
            min_df = 2,
            ngram_range = (1, 3),
            max_features = self.max_features,
            sublinear_tf = True
        )
        
        # @var word character_vectorizer TfidfVectorizer Generate a new character vectorizer
        character_vectorizer = TfidfVectorizer (
            analyzer = 'char',
            lowercase = True,
            min_df = 2,
            ngram_range = (3, 5),
            max_features = self.max_features,
            sublinear_tf = True
        )
        
        
        # Store the vectorizer
        self.ngrams_vectorizer = Pipeline ([
            ('features', FeatureUnion ([
                ('word_ngram', word_vectorizer),
                ('char_ngram', character_vectorizer),
            ])),
            ('reduction', TruncatedSVD (n_components = 100))
        ])
        
        
        # @var texts String Get the texts for fiting
        texts = X[self.field].astype (str)

        
        # Fit on the data
        self.ngrams_vectorizer.fit (texts)

        
        return self
    
    
    def transform (self, X, **transform_params):
    
        # Return tokens from cache
        if self.cache_file and os.path.exists (self.cache_file):
            return pd.read_csv (self.cache_file, header = 0, sep = ",")
        

        # @var features_data
        features_data = self.ngrams_vectorizer.transform (X[self.field].astype (str))
        
        
        # Retrieve the tokens for the subsets we are dealing with
        features = pd.DataFrame (data = features_data)
        
        
        # Create a dataframe with the features
        features.columns = self.get_feature_names ()
        
        
        # Store on disk to prevent recalculating again
        if self.cache_file:
            features.to_csv (self.cache_file, index = False)
            
        return features
    
    
    def get_feature_names (self):
        """Calculate the feature names based on the words """
        return ['ng_' + str (x) for x in range (self.ngrams_vectorizer.named_steps['reduction'].get_params ()['n_components'])]
    
    
    def get_vectorizer (self):
        """
        @return tokenizer
        """
        return self.ngrams_vectorizer
    

    def save_vectorizer_on_disk (self, vectorizer_filename):
        """
        Store vectorizer for further use
        
        @param vectorizer_filename string
        """
        os.makedirs (os.path.dirname (vectorizer_filename), exist_ok = True)
        
        with open (vectorizer_filename, 'wb') as handle:
            pickle.dump ({
                'vectorizer': self.ngrams_vectorizer
            }, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    
    def load_vectorizer_from_disk (self, vectorizer_filename):
        """
        Load the vectorizer from disk
        
        @param vectorizer_filename string
        """
        with open (vectorizer_filename, 'rb') as handle:
            data = pickle.load (handle)
            self.ngrams_vectorizer = data['vectorizer']
