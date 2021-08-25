"""
Generate BERT sentence vectors

@see config.py

@link https://www.sbert.net/docs/quickstart.html

@author José Antonio García-Díaz <joseantonio.garcia8@um.es>
@author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import csv
import os.path
import pandas as pd
import fasttext

from tqdm import tqdm
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator 


class SentenceEmbeddingsTransformer (BaseEstimator, TransformerMixin):
    """
    SentenceEmbeddingsTransformer 
    
    Uses FastText sentence embeddings
    """
    
    def __init__ (self, model, cache_file = '', field = 'tweet'):
        """
        @param model String
        @param field String
        @param cache_file String
        """
        super ().__init__()
        
        self.model = model
        self.field = field
        self.cache_file = cache_file


    def transform (self, X, **transform_params):

        # Return vectors from cache
        if self.cache_file and os.path.exists (self.cache_file):
            return pd.read_csv (self.cache_file, header = 0, sep = ",")

    
        # @var model Load fastText model
        model = fasttext.load_model (self.model)
        
        
        # Avoid NaN
        X[self.field] = X[self.field].fillna ('')
        
        
        # Create and register a new `tqdm` instance with `pandas`
        # @link https://stackoverflow.com/questions/18603270/progress-indicator-during-pandas-operations
        tqdm.pandas ()
        
        
        # Get vectors. In order to get reference to the original document
        # we maintain the Tweet ID
        features = X[self.field].progress_apply (lambda document: ','.join (str (x) for x in model.get_sentence_vector (document)))
        features = features.str.split (pat = ',', expand = True)
        features.columns = self.get_feature_names () 
        
        
        # Store
        if self.cache_file:
            features.to_csv (self.cache_file, index = False)
        
        
        # Return
        return features
        
    def fit (self, X, y = None, **fit_params):
        return self
        
    def get_feature_names (self):
        return ['se_' + str (x) for x in range (1, 301)]
