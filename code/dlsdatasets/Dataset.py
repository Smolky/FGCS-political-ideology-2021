"""
Obtain the datasets and place them at the dataset folders

This is a base implementation of the datasets for the majority
of cases. However, you can extend this class to adapt to your 
needs. 

A DatasetResolver class is also provided

To run, this code needs environment variables for the authentication
of UMUCorpusClassifier platform.

@see config.py

@author José Antonio García-Díaz <joseantonio.garcia8@um.es>
@author Ricardo Colomo-Palacios <ricardo.colomo-palacios@hiof.no>
@author Rafael Valencia-Garcia <valencia@um.es>
"""

import json
import requests
import sys
import csv
import itertools
import netrc
import os.path
import config
import bootstrap
import argparse
import io
import pandas as pd
import numpy as np
import math
import string
import multiprocessing
import fasttext
import time

from pathlib import Path
from sklearn.model_selection import train_test_split
from preprocessText import PreProcessText
from tqdm import tqdm
from functools import partial
from requests.auth import AuthBase


def animated_loading ():
    """
    Perfrom an animated loading while downloading the corpus
    """
    chars = "/—\|"
    for char in itertools.cycle(['|', '/', '-', '\\']):
        sys.stdout.write ('\r' + 'retrieving corpus from UMUCorpusClassifier...'+char)
        time.sleep (.1)
        sys.stdout.flush () 
        

class Dataset ():
    """
    Dataset
    """

    # The dataframe
    df = None

    
    def __init__ (self, dataset, options, corpus = '', task = '', refresh = False):
        """
        Constructor
        
        @param dataset string
        @param corpus string|null
        @param task string|null
        @param options dict
        @param refresh boolean
        """
        self.dataset = dataset
        self.corpus = corpus
        self.task = task
        self.options = options
        self.refresh = refresh
        self.is_merged = False
        
        
        # Handle exceptions
        """
        if not self.task and 'tasks' in self.options:
            raise Exception ('this dataset requires a task')
            sys.exit ()
        """
        
        if self.task and self.task not in self.options['tasks']:
            raise Exception ('this task does not exists')
            sys.exit ()
        
        
        # @var filename string Determine the filename of the dataset
        self.filename = self.get_working_dir ('dataset.csv')
    
    
    def set (self, df):
        """
        @param df
        """
        self.df = df
    
    
    def get_options (self):
        """
        @return dict
        """
        return self.options
    
    
    def save_on_disk (self, df):
        """ 
        Save on disk 
        
        Stores the dataframe on disk for further use
        
        @param df
        
        """
        os.makedirs (os.path.dirname (self.filename), exist_ok = True)
        df.to_csv (self.filename, index = False, quoting = csv.QUOTE_ALL)
    
    def get_scoring_metric (self):
        """ 
        get_scoring_metric
        
        Returns the most appropiate metric
        
        @link https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        
        @return string 
        """
        
        # First, check if we have defined our custom metric for my task
        if self.task and 'scoring' in self.options.tasks[self.task]:
            return self.options.tasks[self.task]['scoring']
        
        # Second, check if we have defined a custom metric globally
        if 'scoring' in self.options:
            return self.options['scoring']
        
        # If no, try to determine according to the task type
        if 'classification' == self.get_task_type ():
            if 'balanced' in self.options:
                return 'accuracy' if self.options['balanced'] else 'f1'
            else:
                return 'f1'
        else:
            return 'neg_mean_squared_error'
    
    
    def get (self):
        """
        Retrieve the Pandas Dataframe from disk
        
        @return DataFrame
        """

        if self.df is None:
            
            # Get the dataset as a dataframe from cache
            if not self.refresh and os.path.isfile (self.filename):
                self.df = pd.read_csv (self.filename, header = 0, sep = ",")
            
            # Compile and store into cache
            else:
                self.df = self.compile ()
        
        
        # Adapt the dataset to the specific task we are dealing with
        if self.task:
            self.df = self.getDFFromTask (self.task, self.df)
        
        
        # Fine tune the dataframe
        self.df = self.finetune_df (self.df)
        
        
        return self.df
    
    
    def finetune_df (self, df):
        """
        Apply some techniques to get the dataframe with correct column types
        
        @param df
        
        @return df
        """
        
        # Categorical variables
        for column in self.get_columns_to_categorical ():
            if column in df.columns:
                df[column] = df[column].astype ('category')
        
        
        # Label
        if 'classification' == self.get_task_type () and 'label' in df.columns:
            df['label'] = df['label'].astype ('category')
        
        
        # Update dataframe types
        # Fields
        for column in ['tweet', 'tagged_pos', 'tagged_ner', 'tweet_clean', 'tweet_clean_lowercase']:
            if column in df.columns:
                df[column] = df[column].fillna ('')
                df[column] = df[column].astype (str)
        
        
        # Return
        return df
    
    
    def get_working_dir (self, *files):
        """
        Returns the working dir for this dataset. Your can prepend filenames too
        
        @param files string|list
        """
        
        # @var path String
        path = os.path.join (config.directories['assets'], self.dataset, self.corpus, *files)
        
        
        # @var directory_path String
        directory_path = os.path.dirname (path)
        
        
        # Ensure directory exists
        os.makedirs (directory_path, exist_ok = True)
        
        
        # return path
        return path
    
    
    def calculate_language (self, field = "tweet", threshold = 0.75):
        """
        @param field String What it is the column that contains the texts in which language has to be infered
        @param threshold int Values lower than this threshold are considered unkown
        """
        
        # @var model This multi language model is used to get the lang
        model = fasttext.load_model (os.path.join (config.directories['pretrained'], 'lid.176.bin'))
        
        
        def func (tweet):
            
            # prediction
            language, affinity = model.predict (tweet, k = 1)
            
            
            # @var first_language String
            language = language[0].replace ('__label__', '')
            
            
            # @var first_language_affinity int
            affinity = affinity[0]
            
            
            # Return the language if exceedes the threshold
            return language if affinity > threshold else None
        
        
        # Register TQDM to work with Pandas
        tqdm ().pandas ()
        
        
        # Assign the language
        self.df['language'] = self.df[field].progress_apply (func)
        
        
        return self.df
    
    
    def compile (self):
        """ 
        Compiles the dataset 
        """
        
        # @var auth_token String
        auth_token = _get_auth_token (config.umucorpusclassifier_api_endpoint, config.certificate)
        
        
        # @var preprocessing Dict
        preserve_text_properties = [
            'lettercase', 
            'mentions', 
            'emojis', 
            'punctuation', 
            'digits', 
            'msg_language',
            'misspellings',
            'elongation'
        ];
        
        
        # var export_fields
        export_fields = [
            'twitter_id', 'twitter_created_at', 'tweet', 'corpus', 
            'user', 'agreement', 'votes', 'score', 'label'
        ]
        
        
        # @request_payload Dir
        request_payload = {
            'export-format': 'csv',
            'size': sys.maxsize,
            'corpora[]': ','.join (str(x) for x in self.options['ids']),
            'preprocessing[]': preserve_text_properties,
            'fields[]': export_fields
        }
        
        
        # Attach strategy (if specified)
        if ('strategy' in self.options):
            request_payload['strategy'] = self.options['strategy'];
        
        
        # Attach if the corpus is balanced
        if ('balanced' in self.options):
            request_payload['balanced'] = True;

        if ('filter-date-start' in self.options):
            request_payload['filter-date-start'] = self.options['filter-date-start'];

        if ('filter-date-end' in self.options):
            request_payload['filter-date-end'] = self.options['filter-date-end'];
        
        
        # Create the loading
        loading = multiprocessing.Process (target = animated_loading, args=())
        loading.start ()
        
        
        # @var reponse Response
        response = requests.post (
            url = config.umucorpusclassifier_api_endpoint + 'admin/export-corpus.csv', 
            json = request_payload, 
            verify = config.certificate,
            auth = _PLNAuth (auth_token)
        )
        
        
        # Stop loading
        loading.terminate ()
        
        
        # Check response
        if (response.status_code == 401):
            print ("Authentication failed: " + str (response.status_code))
            print (response.text)
            print (request_payload)
            sys.exit ()
            
        if (response.status_code != 200):
            print ("Request failed: " + str (response.status_code))
            print (response.text)
            print (request_payload)
            sys.exit ()
        
        
        # Store compiled dataset to disk
        s = response.content
        
        
        # @var df DataFrame
        df = pd.read_csv (io.StringIO (s.decode ('utf-8')))
        
        
        # Store this data on disk
        self.save_on_disk (df)
        
        
        # ...
        return df
    
    
    def get_num_labels (self):
        """
        Get the number of labels for classification. 
        
        This returns 1 if the classification is performed in a regression task
        
        @return int
        """
        return 1 if self.get_task_type () == 'regression' else len (self.df['label'].dropna ().unique ())
    
    
    def get_true_labels (self):
        """
        @return Dict  Get true labels as string
        """
        return dict (enumerate (self.df['label'].astype ('category').cat.categories))
    
    
    def get_available_labels (self):
        """
        @return Series
        """
        return sorted (self.df['label'].unique ())
    
    
    def get_dataset_language (self):
        """ Returns the language of the documents of the corpus. Default is Spanish """
        return self.options['language'] if 'language' in self.options else 'es'
    
    
    def get_train_size (self):
        """ Returns the train_size """
        return self.options['train_size'] if 'train_size' in self.options else 0.8
    
    
    def get_val_size (self):
        """ Returns the val_size """
        return self.options['val_size'] if 'val_size' in self.options else 0.25
    
    
    def is_imabalanced (self, threshold = .15):
        """ 
        Determines if the dataframe is imbalanced or not 
        
        @param threshold float
        """
        
        if self.get_task_type () == 'regression':
            return False
            
        if 'balanced' in self.options:
            return not self.options['balanced']
        
        # @var standard_deviation float To determine the deviation of the percentages
        standard_deviation = np.mean (self.df['label'].value_counts (normalize = True).std ())
        
        return standard_deviation > threshold
    
    
    
    def preprocess (self, df, pipeline = [], field = 'tweet'):
        """
        
        Apply generic preprocessing to the documents
        
        @param self
        @param df Dataframe
        @param pipeline List
        """
        
        # Preprocess text
        preprocess = PreProcessText ()
        
        
        # Custom pipeline
        if pipeline:
            for pipe in tqdm (pipeline, desc = "preprocessing"):
                df[field] = getattr (preprocess, pipe)(df[field])
            return df


        # @var preprocessing_steps
        preprocessing_steps = [
            'remove_urls', 'remove_hashtags', 'remove_mentions', 'remove_digits', 'remove_whitespaces', 
            'remove_elongations', 'remove_emojis', 'remove_quotations', 'remove_punctuation', 
            'remove_whitespaces', 'strip'
        ];
        
        
        # Preprocess text before tokenize
        if self.get_dataset_language () == 'es':
            df[field] = preprocess.expand_acronyms (df[field], preprocess.msg_language)
            df[field] = preprocess.expand_acronyms (df[field], preprocess.acronyms)
        
        for pipe in tqdm (preprocessing_steps):
            df[field] = getattr (preprocess, pipe)(df[field])
    
        return df
    
    
    def get_split (self, df, key):
        """
        Split dataset
        Split the dataframes into training, validation and testing (0, 1, 2)
        
        @param df
        @param key string|int
        """
        
        # @var split__column String 
        # To check if the split is supplied directly
        # Note that this has changed and __split is the only valid field. In case 
        # that you have different subtasks with different splits, please fix it 
        # getDFFromTask from the specific dataset class you are using
        split_field = '__split'
        
        
        # Check if the dataset has a custom split field that indicates which instances 
        # are for training, development, or testing
        if split_field in df.columns:
            return df[df[split_field] == key]
        
        
        # Short names
        if key == 'all':
            return df
        
        if key == 'train':
            key = 0

        if key == 'val':
            key = 1

        if key == 'test':
            key = 2

        
        # @var splits Stratified split
        splits = [None, None, None]
        
        
        # Determine the strategy for stratify
        stratify_strategy = df[self.options['stratify']] if 'stratify' in self.options else None
        splits[0], splits[2] = train_test_split (df, train_size = self.get_train_size (), random_state = bootstrap.seed, stratify = stratify_strategy)
        
        stratify_strategy = splits[0][self.options['stratify']] if 'stratify' in self.options else None
        splits[0], splits[1] = train_test_split (splits[0], test_size = self.get_val_size (), random_state = bootstrap.seed, stratify = stratify_strategy)
        
        
        # Fine tune the split
        split_df = self.finetune_df (splits[key])
        
        
        # Return the selected split
        return split_df
        
        
    def get_task_type (self):
        """ 
        Determine the task type
        
        return String
        """
        return self.options['tasks'][self.task]['type'] if 'tasks' in self.options and self.task and 'type' in self.options['tasks'][self.task] else (self.options['type'] if 'type' in self.options else 'classification')
        
        
    def get_primary_key (self, task):
        """ 
        Determine the primary key of the dataset. 
        
        It depends of if the documents are merged by author (user) or not (twitter_id)
        """
    
        if 'tasks' in self.options and task:
            return 'user' if self.options['tasks'][task]['merge'] else 'twitter_id'
        elif 'merge' in self.options:
            return 'user' if self.options['merge'] else 'twitter_id'
        else:
            return 'twitter_id'
            
    def get_task_options (self):
        """
        Returns the options of the dataset according to the 
        current task
        
        @return Dict
        """
        
        # @var options_without_task Dict
        options_without_task = {key: options for key, options in self.options.items () if key not in ['tasks']}
        
        if self.task:
            return {**self.options['tasks'][self.task], **options_without_task}
        else:
            return options_without_task
            
    def getDFFromTask (self, task, df):
        """
        Adjust the dataframe for an specific task
        """
        
        # Adjust the label
        df['label'] = df[self.options['tasks'][task]['label']]
        
        
        # Merge tweets
        if 'merge' in self.options['tasks'][task] and self.options['tasks'][task]['merge']:
            self.is_merged = True
            

        
        return df
        
        
    def get_columns_to_group_by_user (self):
        """
        This function returns a list of columns needed to maintain when grouping a dataset
        by users
        
        Extend this in subclasses
        
        @return List
        """
        return ['user', 'label', '__split']
        
        
    def get_columns_to_categorical (self):
        """
        Determines which columns should be converted in categorical values
        
        Extend this in subclasses
        
        @return List
        """
        if 'classification' == self.get_task_type ():
            return ['label', '__split']
        else:
            return ['__split']


class _PLNAuth (AuthBase):
    """
       PLNAuth
       
       We use a custom authentication because the default authentication
       was adding Basic authentication and do not allowed to manually 
       introduced the token
       
       @link https://requests.readthedocs.io/en/master/user/advanced/#custom-authentication
    """

    def __init__(self, username):
        self.username = username

    def __call__(self, r):
        r.headers['Authorization'] = self.username
        return r
        
        
def _get_auth_token (api_endpoint, certificate):
    """
    @param api_endpoint
    @param certificate
    """

    # Read from the .netrc file in your home directory
    secrets = netrc.netrc ()
    email, account, password = secrets.authenticators ('collaborativehealth.inf.um.es')
    
    
    # @var reponse Response
    response = requests.post (
        api_endpoint + 'login', 
        json={'email': email, 'password': password}, 
        verify = certificate
    )
    
    
    # Transform to JSON
    response = response.json ()
    
    
    # ...
    return str (response['data']['token'])
    