import csv
import sys
import string
import numpy as np
import pandas as pd
import os
import xml.etree.ElementTree as ET

from tqdm import tqdm
from .Dataset import Dataset
from scipy import stats


class DatasetPANAuthorProfiling2019 (Dataset):
    """
    DatasetPANAuthorProfiling2019
    
    @extends Dataset
    """

    def __init__ (self, dataset, options, corpus = '', task = '', refresh = False):
        """
        @inherit
        """
        Dataset.__init__ (self, dataset, options, corpus, task, refresh)
    
    def compile (self):
        
        # Load dataframes
        dfs = []
        for index, dataframe in enumerate (['author-profiling-2019-training-es.txt', 'author-profiling-2019-test-es.txt']):
            
            # Open file
            df_split = pd.read_csv (self.get_working_dir (dataframe), delimiter = ":::", header = None)
            
            
            # Determine training and testing splits
            df_split = df_split.assign (is_train = index == 0)
            df_split = df_split.assign (is_test = index == 1)
            
            
            # Set the text file which contains the files
            df_split = df_split.assign (folder = "author-profiling-2019-training-es" if index == 0 else "author-profiling-2019-test-es")
            
            
            # Merge
            dfs.append (df_split)
        
        
        # Concat and assign
        df = pd.concat (dfs, ignore_index = True)
        df.columns = ['user', 'nature', 'gender', 'is_train', 'is_test', 'folder'] 

        
        # @var new_rows Rows for the new dataframe
        new_rows = []
        
        
        # Iterate over the dataframe
        for index, row in tqdm (df.iterrows (), total = df.shape[0]):
            
            # Get the file that contains the tweets
            text_file_path = self.get_working_dir (row['folder'], row['user'] + ".xml")
            
            
            # Open file
            xml_data = open (text_file_path, 'r').read ()
            
            
            # Transform from XML
            root = ET.XML (xml_data)
            
            
            # Iterate...
            for i, child in enumerate (root):
                for j, subchild in enumerate (child):
                    new_rows.append (row.append (pd.Series ([subchild.text.replace ('\n', '. ')])))

            
            
        # Assign the tweets
        df = pd.DataFrame (new_rows)
        df.columns = ['user', 'nature', 'gender', 'is_train', 'is_test', 'folder', 'tweet'] 
        
        
        # Create fake twitter_id
        df['twitter_id'] = np.arange (len (df))
        
        
        # Divide the dataset into training, evaluation, and testing
        df = df.assign (__split = 'train')
        df = df.assign (is_val = False)
        
        
        # @var val_users List 
        # 
        # To keep the balance, we are going to select a subset of the users 
        # from the training dataset for training and validation
        # This step is important in order to do not mix the same user
        # in the validation and training set in case you want to merge
        # tweets
        val_users = df['user'] \
            .drop_duplicates () \
            .sample (frac = self.options['val_size'], replace = False) \
            .to_list ()
        
        
        # As we are spliting the dataset with frac, we are adding some random 
        # to the training and dataset splits
        df.loc[df['user'].isin (val_users), 'is_val'] = True
        df.loc[df['is_val'], 'is_train'] = False
        df.loc[df['is_train'], '__split'] = 'train'
        df.loc[df['is_val'], '__split'] = 'val'
        df.loc[df['is_test'], '__split'] = 'test'
        
        
        # Drop non-needed columns
        df = df.drop (['folder', 'is_train', 'is_val', 'is_test'], axis = 1)
        
        
        # Set a default label field
        df['label'] = df['gender']
        
        
        # Store this data on disk
        self.save_on_disk (df)
        
        
        # Return
        return df
        
        
    def getDFFromTask (self, task, df):
        """
        Adjust the dataframe for an specific task
        """
        
        # Adjust the label
        # In case of subtask 1 (nature), male and famles are encoded as human
        if task in ['nature_user_level', 'nature_tweet_level']:
            df['label'] = np.select ([
                df['label'].eq ('male') | df['label'].eq ('female'),
                df['label'].eq ('bot')
            ], ['human', 'bot'], default = 'human')
        
        
        # In case of subtask 2 (gender), bots are removed from the dataframe
        # Note that we reindex the dataframe because the features will be 
        # reindex too
        if task in ['gender_user_level', 'gender_tweet_level']:
            df = df.drop (df[(df['label'] == 'bot')].index).reset_index (drop = True)
            
        
        # Merge tweets
        self.is_merged = 'merge' in self.options['tasks'][task] and self.options['tasks'][task]['merge']
            

        
        return df