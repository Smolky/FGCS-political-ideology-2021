import sys
import csv
import itertools
import subprocess
import os.path
import io
import config
import dask as da
import numpy as np
import pandas as pd
import tempfile
import shlex

import dask.dataframe as ddf
import multiprocessing as mp

from dask.diagnostics import ProgressBar

from io import StringIO
from pathlib import Path
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator 
from sklearn import preprocessing

class LinguisticFeaturesTransformer (BaseEstimator, TransformerMixin):
    """
    lfTransformer
    
    This class allows to extract linguistic features from a dataframe
    by using UMUTextStats
    """
    
    def __init__ (self, model = 'default.xml', cache_file = '', field = 'tweet', label = 'label'):
        """
        @param model String (see Config)
        @param cache_file String
        """
        super().__init__()
        
        self.model = model
        self.cache_file = cache_file
        self.field = field
        self.label = label
        self.columns = None
        self.cache = None
        
    def transform (self, X, **transform_params):
        
        if self.cache is not None:
            return self.cache
        
        # Return vectors from cache
        if self.cache_file and os.path.exists (self.cache_file):
            return pd.read_csv (self.cache_file, header = 0, sep = ",")
            
            
        # To correctly encode backslashes for the main label
        X[self.field] = X[self.field].str.replace ("\\", "\\\\")
        
        
        # Reduce columns to save space
        X = X[[self.field, self.label, 'tagged_pos', 'tagged_ner']]
        
        
        def get_lf_batch (X):
        
            # Temporal file
            with tempfile.NamedTemporaryFile () as temp:
            
                # @var is_fake_data Boolean
                # @link https://docs.dask.org/en/latest/dataframe-api.html#dask.dataframe.DataFrame.map_partitions
                is_fake_data = 'foo' == X[self.label].iloc[0];
                
                
                # @var save_progress_file String
                save_progress_file = self.cache_file.replace ('.csv', '-' + str (X.index[0]) + '-' + str(X.index[-1]) + '.csv') if not is_fake_data and self.cache_file else ""
                
                
                # IF the file exists, it was already computed
                if save_progress_file and os.path.exists (save_progress_file):
                    return pd.read_csv (save_progress_file, header = 0, sep = ",")
                
                
                # Get the name
                temp_file = temp.name + '.csv'
                
                
                # Store the file temporally to process with UMUTextStats
                X.to_csv (temp_file, index = False, quoting = csv.QUOTE_ALL)
                
                
                # Create command
                cmd = config.umutextstats_api_endpoint + " --source-provider=files --file=" + temp_file + " --format=csv --umutextstats-config=" + self.model
                
                
                # Get results from the command line
                data = subprocess.run (shlex.split (cmd), stdout = subprocess.PIPE, check = True, timeout = None)
                data = data.stdout.decode ("utf-8") 

                
            # Extract the values
            rows = [x.split (',') for x in data.split ('\n')[1:-1]]
            
            
            # Get columns for the first time
            if not self.columns:
                self.columns = [x for x in data.split ('\n')[0].split (',')][:-1]
            
            
            
            # Get final dataframe
            features = [row[:-1] for row in rows]
            features = list (np.float_ (features))
            features = pd.DataFrame (features, columns = self.columns, index = X.index)
            
            
            # Store on disk to prevent recalculating again
            if save_progress_file:
                features.to_csv (save_progress_file, index = False)
            
            
            # Return vectors
            return features
        
        
        # @var npartitions int The number of partitions (not lower than 1)
        npartitions = (len (X) // 1000) | 1
        
        
        # Divide the dataframe for paralelization
        df_dask = ddf.from_pandas (X, npartitions = npartitions)

        
        # @var operation For paralelisation
        operation = df_dask.map_partitions (get_lf_batch)
        
        
        # Show progressbar while we are fetching the results...
        with ProgressBar ():
            features = operation.compute ()
        
        
        # Create a dataframe with the features
        features.columns = self.get_feature_names ()
        
        
        # Store on disk to prevent recalculating again
        if self.cache_file:
            features.to_csv (self.cache_file, index = False)
        
        
        return features
        
    
    def fit (self, X, y = None, **fit_params):
        return self
        
    def get_feature_names (self):
    
        # Return vectors from cache
        if self.cache_file and os.path.exists (self.cache_file):
            return pd.read_csv (self.cache_file, header = 0, sep = ",", nrows = 1).columns
    
        return self.columns

