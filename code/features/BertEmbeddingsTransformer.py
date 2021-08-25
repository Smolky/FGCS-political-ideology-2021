import sys
import csv
import os.path
import io
import numpy as np
import pandas as pd
import transformers
import torch
from pathlib import Path

from tqdm import tqdm

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator 


class BertEmbeddingsTransformer (BaseEstimator, TransformerMixin):
    """
    Generate BERT sentence vectors

    @see config.py

    @link https://www.sbert.net/docs/quickstart.html
    """
    
    def __init__ (self, model, cache_file = "", field = "tweet"):
        """
        @param model String
        @param field String
        @param cache_file String
        """
        super ().__init__()
        
        self.model = model
        self.field = field
        self.cache_file = cache_file
        self.number_of_features = 768
    
    
    def mean_pooling (self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        
        @link https://www.sbert.net/examples/applications/computing-embeddings/README.html
        @link https://gist.github.com/haayanau/e7ca837b9503afbf68d1407bed633619
        """
        
        #First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        
        input_mask_expanded = attention_mask.unsqueeze (-1).expand (token_embeddings.size ()).float ()
        sum_embeddings = torch.sum (token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp (input_mask_expanded.sum (1), min = 1e-9)
        
        return sum_embeddings / sum_mask


    def transform (self, X, **transform_params):
    
        # Return vectors from cache
        if self.cache_file and os.path.exists (self.cache_file):
            return pd.read_csv (self.cache_file, header = 0, sep = ",")
    
        
        # Load AutoModel from huggingface model repository
        tokenizer = transformers.AutoTokenizer.from_pretrained (self.model)
        
        
        # @var model 
        model = transformers.AutoModel.from_pretrained (self.model)
        
        
        def get_bert_embeddings (df):
            """
            @param df DataFrame
            """
            
            # @var encoded_inputs Get data. Note fillna to avoid None and empty strings
            encoded_inputs = tokenizer (
                df[self.field].fillna ('').tolist (), 
                padding = True, 
                truncation = True, 
                max_length = 256, 
                return_tensors = 'pt'
            )
            
            
            # Compute token embeddings
            with torch.no_grad ():
                model_outputs = model (**encoded_inputs)
                sentence_embeddings = self.mean_pooling (model_outputs, encoded_inputs['attention_mask'])
                return pd.DataFrame (sentence_embeddings.numpy ())
        
        
        # @var frames List of DataFrames
        frames = []
        
        
        # Iterate on batches
        for chunk in tqdm (np.array_split (X, min ([100, len (X)]))):
            frames.append (get_bert_embeddings (chunk))
        
        
        # @var features DataFrame Concat frames in row axis
        features = pd.concat (frames)
        
        
        # Assign column names
        features.columns = self.get_feature_names ()
        

        # Store
        if self.cache_file:
            features.to_csv (self.cache_file, index = False)
        
        
        # Return vectors
        return features
        
        
    def fit (self, X, y = None, **fit_params):
        return self
        
    def get_feature_names (self):
        return ['be_' + str (x) for x in range (1, self.number_of_features + 1)]
