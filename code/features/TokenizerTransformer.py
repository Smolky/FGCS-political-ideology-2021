import pandas as pd
import pickle
import os.path

from tensorflow import keras
from keras import backend as K
from pathlib import Path
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator 

class TokenizerTransformer (BaseEstimator, TransformerMixin):
    """
    Obtain sentences tokenized

    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
    """

    def __init__ (self, cache_file = '', field = 'tweet', max_words = None):
        """
        @param model String (see Config)
        @param cache_file String
        """
        super().__init__()
        
        self.cache_file = cache_file
        self.field = field
        self.max_words = max_words
        self.columns = None
        self.tokenizer = None
        self.maxlen = None
        
    
    # Return self nothing else to do here
    def fit (self, X, y = None ):
    
        # @var Tokenizer When fiting, we create a new tokenizer
        self.tokenizer = keras.preprocessing.text.Tokenizer (num_words = self.max_words, oov_token = True)
        
        
        # Fit on the data
        self.tokenizer.fit_on_texts (X[self.field])
        
        
        # @var maxlen int Get the max-len size
        self.maxlen = max (len (l) for l in self.tokenizer.texts_to_sequences (X[self.field]))
        
        return self 
        
    def transform (self, X, **transform_params):
    
        # Return tokens from cache
        if self.cache_file and os.path.exists (self.cache_file):
            return pd.read_csv (self.cache_file, header = 0, sep = ",")
            
        
        # Retrieve the tokens for the subsets we are dealing with
        features = pd.DataFrame (data = keras.preprocessing.sequence.pad_sequences (
            self.tokenizer.texts_to_sequences (X[self.field]), 
            padding = 'pre', 
            maxlen = self.maxlen
        ))
        
        
        # Create a dataframe with the features
        features.columns = self.get_feature_names ()
        
        
        # Store on disk to prevent recalculating again
        if self.cache_file:
            features.to_csv (self.cache_file, index = False)
            
        return features

    def get_feature_names (self):
        """Calculate the feature names based on maxlen"""
        return ['we_' + str (x) for x in range (1, self.maxlen + 1)]


    def get_tokenizer (self):
        """
        @return tokenizer
        """
        return self.tokenizer
    

    def save_tokernizer_on_disk (self, token_filename):
        """
        Store tokenizer for further use
        
        @param token_filename string
        """
        os.makedirs (os.path.dirname (token_filename), exist_ok = True)
        with open (token_filename, 'wb') as handle:
            pickle.dump ({"tokenizer": self.tokenizer, "maxlen": self.maxlen}, handle, protocol = pickle.HIGHEST_PROTOCOL)
    

    def load_tokenizer_from_disk (self, token_filename):
        """
        Load the tokenizer from disk
        
        @param token_filename string
        """
        with open (token_filename, 'rb') as handle:
            data = pickle.load (handle)
            self.tokenizer = data['tokenizer']
            self.maxlen = data['maxlen'] * 1
