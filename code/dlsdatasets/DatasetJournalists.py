"""
    DatasetJournalists
    
    This dataset includes heuristics to discard users and tweets 
    
    @see config.py
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import config
import fasttext
import bootstrap
import re
import regex
import csv
import sys
import math
import string
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from .Dataset import Dataset
from scipy import stats


class DatasetJournalists (Dataset):
    """
    DatasetJournalists
    
    @extends Dataset
    """

    # The dataframe
    df = None
    
    
    # Set the limits of tweets per user
    # Users with less than tweets_min_limit will be discarded
    # Less relevant tweets (tweets per user minus tweets_max_limit) will be discarded
    tweets_min_limit = 100
    tweets_max_limit = 200
    tweets_per_task1 = 166
    
    
    def __init__ (self, dataset, options, corpus = '', task = '', refresh = False):
        """
        @inherit
        """
        Dataset.__init__ (self, dataset, options, corpus, task, refresh)
    
    
    def compile (self):
        """
        This process filters the tweets to perform author profile
        """
        
        # @var political_accounts DataFrame
        political_accounts = pd.read_csv (self.get_working_dir ('helpers', 'accounts.csv'))
        
        
        # Retrieve the dataframe from UMUCorpusClassifier
        self.df = super ().compile ()
        
        
        # Statistics. 
        # ---------------------------------------------------------------------
        # Note that we store this file for further analysis before removing any 
        # user nor tweet
        # (1) Store the number of tweets per user
        self.df \
            .groupby (['user']).size () \
            .to_csv (self.get_working_dir ('helpers', 'stats-tweets-by-user.csv'))

        
        # (2) Store tweets per month
        self.df \
            .assign (month = pd.to_datetime (self.df['twitter_created_at']).dt.to_period ('M')) \
            .groupby (['user', 'month']).size () \
            .to_csv (self.get_working_dir ('helpers', 'stats-journalists-by-month.csv'))
            
        
        # Update tweet datatype to ensure it is understood as string
        self.df.tweet = self.df.tweet.astype (str)
        
        
        # Remove tweets that are from media and news (contains "|", vía, a través de, ...)
        self.df = self.df.loc[~self.df['tweet'].str.contains (r'((a través de|v[íi]a)) @+[[A-Za-zÑñ\d\_]+$|\||#endirecto'), :]
        
        
        # Get a clean version of the tweet, that we will use to infer the language
        # in which the tweets were written
        self.df['tweet_clean'] = self.df['tweet']
        self.df = self.preprocess (self.df, field = 'tweet_clean')
        
        
        # Guess the language for each tweet using the cleaned tweet version
        self.df = self.calculate_language (field = 'tweet_clean')
        
        
        # (3) Store tweets per language and author
        # Note that we store this file for further analysis
        # before removing any user nor tweet
        self.df.groupby (['user', 'language']).size () \
            .to_csv (self.get_working_dir ('helpers', 'stats-journalists-by-language.csv'))
        
        
        # Remove tweets which inferred language is not Spanish and
        # remove the language column because it is not needed anymore
        self.df = self.df.drop (self.df[self.df.language != 'es'].index)
        self.df = self.df.drop (['language'], axis = 1)
        
        
        # Strip users which did not have so many tweets
        user_count_series = self.df.groupby (['user']).size ()
        
        
        # @var df_users DataFrame Get users to filter as a dataframe with their counts
        df_users = user_count_series.reset_index (name = 'counts')
        
        
        # @var threshold float Use the MAD value as threshold
        threshold = round (stats.median_abs_deviation (user_count_series, scale = 2 / math.sqrt (3)))
        
        
        # We are generating a blacklist of users to remove
        # Then, we remove from this list those users which the number 
        # of tweets are higher than the threshold
        df_users = df_users.drop (df_users[df_users['counts'] >= self.tweets_min_limit].index)
        
        
        # Then remove those users with few tweets from the corpus
        self.df = self.df.drop (self.df[self.df.user.isin (df_users['user'])].index)
        
        
        # Get topics
        # Note that we do this before the annonymisation
        # This file contains some relevant topics of the tweets
        hashtags_df = pd.read_csv (self.get_working_dir ('helpers', 'topics-helper.csv'), header = 0, sep = ",")
        
        
        # @var hashtag_regex String
        hashtag_regex = '|'.join ([hashtags_df[col].dropna ().str.cat (sep = '|') for col in hashtags_df.columns]).replace (' ', '\s?')
        
        
        # Detect if the tweet is relevant according the presence of hashtags
        self.df = self.df.assign (is_relevant = lambda x: x['tweet'].str.contains (hashtag_regex, regex = True, flags = re.IGNORECASE))
        
        
        # @var df_representative DataFrame Get a new version of the dataframe to select the best tweets
        df_representative = self.df.assign (month = pd.to_datetime (self.df['twitter_created_at']).dt.month)
        
        
        # Store the selected IDs
        twitter_ids = {}

        
        # Iterate over users
        for user in tqdm (df_representative.user.unique ()):

            # Create a dataframe for this user
            df_user = df_representative.loc[self.df['user'] == user]
            
            
            # Calculate the number of words and discard tweets with few words
            df_user = df_user \
                .assign (num_words = lambda x: (x['tweet_clean'].str.split ().str.len ())) \
                .sort_values (by = ['is_relevant', 'num_words'], ascending = [False, False])
            
            
            # Create the list for this user
            twitter_ids[user] = []
            
            
            # Until we fill the bucket for each user...
            skip_to_next_user = False
            
            
            # ...
            while (not skip_to_next_user):
                
                # To avoid infinite loops
                do_i_collect_any = False
                
                
                # For each month...
                for month in range (1, 12):
                    
                    # Create a smaller dataframe only for this month
                    # and without the already selected tweets
                    df_month = df_user.loc[df_user['month'] == month]
                    df_month = df_month.loc[~df_month['twitter_id'].isin (twitter_ids[user])]
                    
                    
                    # If there are any available tweets, pick it; if not, skip to the next month
                    if df_month.shape[0] > 0:
                        do_i_collect_any = True
                        twitter_ids[user].append (df_month['twitter_id'].iloc[0])
                        skip_to_next_user = len (twitter_ids[user]) >= self.tweets_max_limit
                
                if (not do_i_collect_any):
                    skip_to_next_user = True
                    
        
        # Filter those tweets from the dataset
        for user in df_representative.user.unique ():
            self.df = self.df.drop (self.df[(self.df.user == user) & (self.df.twitter_id.isin (twitter_ids[user]) == False)].index)

        
        # Anonimize the dataset
        # First, we select the accounts
        # @todo User the same politicians from policorpus
        journalist_accounts = self.df['user'] \
            .drop_duplicates () \
            .sample (frac = 1, random_state = bootstrap.seed).reset_index (drop = True)

        
        # Create a temporal column with the name of the politician
        self.df = self.df.assign (user_temporal = self.df['user'])
        
        
        # Replace the user, both in the label and the text
        # Replace political in tweets
        for index, account in tqdm (enumerate (political_accounts['user'].tolist ())):
            mask = "@user" + str (index + 1)
            self.df['tweet'] = self.df['tweet'].str.replace ("@" + account, mask, flags = re.I)
            
        
        # Replace journalist in the user name 
        for index, account in tqdm (journalist_accounts.items ()):
            mask = "@journalist" + str (index + 1)
            self.df['user_temporal'] = self.df['user_temporal'].str.replace (account, mask, flags = re.I)
        
        
        self.df[['user', 'user_temporal']] \
            .drop_duplicates () \
            .to_csv (self.get_working_dir ('helpers', 'journalist-accounts.csv'), index = False, quoting = csv.QUOTE_ALL)

        
        # Reassign label
        self.df['label'] = self.df['user_temporal']
        self.df = self.df.drop (['user_temporal'], axis = 1)

        
        # Sort the results grouped by user, and then by date
        self.df['twitter_created_at'] = pd.to_datetime (self.df['twitter_created_at'], format = '%Y/%m/%d %H:%M:%S')
        self.df = self.df.sort_values (by = ['user', 'twitter_created_at'], ascending = [True, True])
        
        
        # @var helper_df DataFrame 
        # This file contains the grounding truth of the politicians and we will use it 
        # to attach data as gender, age range and political affiliation to the 
        # dataset
        helper_df = pd.read_csv (self.get_working_dir ('helpers', 'journalists-helper.csv'), header = 0, sep = ",")
        
        
        # Merge both dataframes based on their twitter account
        # Ideological binary        
        self.df = self.df.assign (ideological_multiclass = np.nan)
        self.df = pd.merge (left = self.df, right = helper_df[['user', 'ideological_binary']], on = 'user')
        
        
        # Ideological multiclass 
        self.df = self.df.assign (ideological_multiclass = np.nan)
        self.df.loc[self.df['affiliation'].isin (self.multiclass_moderate_left), 'ideological_multiclass'] = 'moderate_left'
        self.df.loc[self.df['affiliation'].isin (self.multiclass_left), 'ideological_multiclass'] = 'left'
        self.df.loc[self.df['affiliation'].isin (self.multiclass_right), 'ideological_multiclass'] = 'right'
        self.df.loc[self.df['affiliation'].isin (self.multiclass_moderate_right), 'ideological_multiclass'] = 'moderate_right'

        
        # Select the tweets that belong to author profiling
        # ---------------------------------------------------------------------
        # For the author profiling task we want to know who are the authors with more 
        # tweets
        df_tweets_by_user = self.df.groupby (['user']).size () \
            .reset_index (name = 'counts') \
            .sort_values (by = ['counts'], ascending = False) \
            .reset_index (drop = True)
        
        
        # @var min_number_of_tweets int Get the minimum number of tweets
        min_number_of_tweets = int (df_tweets_by_user.tail (1)['counts'])
        
        
        # Get dataframes split for author profiling
        df_task1_train = self.df.loc[self.df['user'].isin (df_tweets_by_user['user'].head (self.tweets_per_task1))]
        df_task1_val_test = self.df.loc[self.df['user'].isin (df_tweets_by_user['user'].tail (df_tweets_by_user['user'].shape[0] - self.tweets_per_task1))]

        
        # @var rest_of_the_users Get the rest of the users that were not selected by training and we sample the half of them
        rest_of_the_users = pd.Series (df_task1_val_test['user'].unique ()).sample (frac = 0.5, random_state = bootstrap.seed)
        
        
        # Get validation and testing by the sampling of the rest of the users
        df_task1_val = df_task1_val_test.loc[df_task1_val_test['user'].isin (rest_of_the_users)]
        df_task1_test = df_task1_val_test.loc[~df_task1_val_test['user'].isin (rest_of_the_users)]
        
        
        # Sample task 1
        for _df in [df_task1_train, df_task1_val, df_task1_test]:
            _df = _df.groupby ('user').head (min_number_of_tweets)
        
        
        # Create new columns to indicate which tweets belong to 
        # train, val, and test for each task
        self.df = self.df.assign (__split_author_profiling = np.nan)
        
        
        # Now we are set as True those indexes in the main DataFrame
        # that belong to train, val, and test for both tasks
        self.df['__split_author_profiling'][df_task1_train.index] = 'train'
        self.df['__split_author_profiling'][df_task1_val.index] = 'val'
        self.df['__split_author_profiling'][df_task1_test.index] = 'test'
        
        
        # Some texts do not belong neither to Task 1 nor Task 2 and they can 
        # be safely droped
        self.df = self.df.drop (self.df[(self.df['__split_author_profiling'].isnull ())].index)
        
        
        # Sample and reorder tweets to ensure that not all the tweets of the same authors 
        # are near other
        self.df = self.df.sample (frac = 1, random_state = bootstrap.seed).reset_index ()
        
        
        # Get the grounding truth
        self.df.groupby (['label']).first () \
               .drop (['tweet', 'twitter_id', 'twitter_created_at', 'user', 'is_relevant'], axis = 1) \
               .rename (columns = {"label": "user"}) \
               .to_csv (self.get_working_dir ('deploy-dataset', 'journalists-2020-truth.csv'), index = False, quoting = csv.QUOTE_ALL)
        
        
        # Generate files to store the tweets and their IDs
        for task, label in enumerate (['__split_author_profiling']):
        
            # Generate a file per group
            for dataset_split in ['train', 'val', 'test']:
                
                # Get the names of the files
                file_ids = open (self.get_working_dir ('deploy-dataset', 'policorpus-journalists-2020-task-' + str (task + 1) + '-' + dataset_split + '-ids.xml'), 'w')
                file_tweets = open (self.get_working_dir ('deploy-dataset', 'policorpus-journalists-2020-task-' + str (task + 1) + '-' + dataset_split + '-tweets.xml'), 'w')
                
                
                # Reference the dataframe
                temp_df = self.df.loc[self.df[label] == dataset_split]
                
                
                # For each user in the dataset
                for anonymized in temp_df['label'].unique ():
                
                    # Save tweet ids and text
                    file_ids.write ("<author id='" + anonymized + "'>\n")
                    file_tweets.write ("<author id='" + anonymized + "'>\n")

                    
                    # Write their tweets
                    for twitter_id in temp_df.loc[temp_df['label'] == anonymized]['twitter_id']:
                        file_ids.write ("\t<document>" + str (twitter_id) + "</document>\n")
                        
                    for tweet in temp_df.loc[temp_df['label'] == anonymized]['tweet']:
                        file_tweets.write ("\t<document><![CDATA[" + str (tweet) + "]]></document>\n")

                    
                    # Close author tag
                    file_ids.write ("</author>\n\n")
                    file_tweets.write ("</author>\n\n")
            
            
            # Close
            file_ids.close () 
            file_tweets.close () 
        
        
        # Reorder and select columns for the daily use dataframe
        self.df = self.df[[
            'index', 'twitter_id', 'twitter_created_at', 'user', 'label',
            'ideological_binary', 'ideological_multiclass', 
            '__split_author_profiling', 'tweet'
        ]]

        
        # Store in disk after filters
        self.save_on_disk (self.df)
        
        
        # Return the dataframe
        return self.df
        
        

    def getDFFromTask (self, task, df):
        
        """
        @inherit
        """
        
        # @var task_options dict
        task_options = self.options['tasks'][self.task]
        
        
        # Adjust the label
        self.df['label'] = self.df[task_options['label']]

        
        # @fix
        # The split field has changed in the merged version and it is possible it 
        # is not appear anymore. One option is to add to the merged dataset
        if task_options['split'] in df.columns:
            self.df['__split'] = self.df[task_options['split']]
        
        
        return df



    def get_columns_to_group_by_user (self):
        """
        @inherit}
        """
        
        # @var columns
        columns = super ().get_columns_to_group_by_user ()
        
        
        # Attach specific columns
        columns += ['ideological_binary', 'ideological_multiclass']
        
        
        # return columns to group
        return columns
        
        
    def get_columns_to_categorical (self):
        """ 
        {@inherit}
        """
        return [
            '__split_author_profiling', 'ideological_binary', 'ideological_multiclass'
        ]
