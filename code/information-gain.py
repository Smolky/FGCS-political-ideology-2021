"""
    Information Gain per class
    
    This class calculates the Information Gain (Mutual Info) of a dataset
    and uses it to select the most discrimatory features
    
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from features.FeatureResolver import FeatureResolver
from sklearn.pipeline import Pipeline, FeatureUnion
from features.TokenizerTransformer import TokenizerTransformer


def main ():

    # var parser
    parser = DefaultParser (description = 'Calculates the Information Gain (Mutual Info) per class and obtains the best LF')
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')


    # @var df Ensure if we already had the data processed
    df = dataset.get ()
    
    
    # @var task_type String
    task_type = dataset.get_task_type ()
    
    
    # @var df_train DataFrame
    df_train = dataset.get_split (df, 'train')
    df_train = df_train[df_train['label'].notna()]
    
    
    # @var feature_resolver FeatureResolver
    feature_resolver = FeatureResolver (dataset)
    
    
    # @var feature_file String
    feature_file = feature_resolver.get_suggested_cache_file ('lf', task_type)
    
    
    # @var features_cache String The file where the features are stored
    features_cache = dataset.get_working_dir (args.task, feature_file)
    
    
    # If the feautures are not found, get the default one
    if not Path (features_cache).is_file ():
        raise Exception ('features lf file are not avaiable')
        sys.exit ()
        
    
    # @var transformer Transformer
    transformer = feature_resolver.get ('lf', cache_file = features_cache)
    
    
    # @var features_df DataFrame
    features_df = transformer.transform ([])
    
    
    # @var linguistic_features List
    linguistic_features = features_df.columns.to_list ()
        
    
    # Keep only the training features
    features_df = features_df[features_df.index.isin (df_train.index)].reindex (df_train.index)
    
    
    # Attach label
    features_df = features_df.assign (label = df_train['label'])

    
    
    # @var unique_labels Series Bind to the label
    unique_labels = dataset.get_available_labels ()
    
    
    # @var X
    X = features_df.loc[:, features_df.columns != 'label']
    
    
    # @var mi 
    if 'classification' == task_type:
        mi = mutual_info_classif (X = X, y = df_train['label']).reshape (-1, 1)
        
    elif 'regression':
        mi = mutual_info_regression (X = X, y = df_train['label']).reshape (-1, 1)
    
    
    # @var best_features_indexes List
    best_features_indexes = pd.DataFrame (mi, 
        columns = ['Coefficient'], 
        index = linguistic_features
    )
    
    
    # Best indexes and categories
    best_features_indexes = best_features_indexes.assign (category = best_features_indexes.index)
    best_features_indexes['category'] = best_features_indexes['category'].apply (lambda x: x.split ('-')[0])
    best_features_indexes['category'] = best_features_indexes['category'].apply (lambda x: x.split ('_')[0])
    
    
    
    # Best features for regression tasks
    if 'regression' == task_type:

        print ("by dataset")
        print ("----------")
        best_features_indexes.index = linguistic_features
        print ("top")
        print (best_features_indexes.sort_values (by = 'Coefficient', ascending = False).head (20).to_csv (float_format = '%.5f'))
        
        print ("worst")
        print (best_features_indexes.sort_values (by = 'Coefficient', ascending = True).head (10).to_csv (float_format = '%.5f'))

    
    
    # Best features for classification tasks
    if 'classification' == task_type:
    
        # @var average_features_per_label List
        average_features_per_label = [features_df.loc[df_train.loc[features_df['label'] == label].index].mean ().to_frame ().T for label in unique_labels]
        
        
        # Merge features by label
        features_df_merged = pd \
            .concat (average_features_per_label) \
            .reset_index (drop = True) \
            .assign (label = unique_labels) \
            .set_index ('label') \
            .transpose ()
        
        
        # Attach coefficient to the features
        features_df_merged = features_df_merged.assign (Coefficient = best_features_indexes['Coefficient'].values)


        # Remove those features without coefficient
        features_df_merged = features_df_merged.loc[features_df_merged['Coefficient'] > 0]
        best_features_indexes = best_features_indexes.loc[best_features_indexes['Coefficient'] > 0]
        
        
        # Assign indexes
        best_features_indexes.index = features_df_merged.index
        
        
        
        print ("by dataset")
        print ("----------")
        
        print ("top")
        print (best_features_indexes.sort_values (by = 'Coefficient', ascending = False).head (10).to_csv (float_format = '%.5f'))
        
        print ("worst")
        print (best_features_indexes.sort_values (by = 'Coefficient', ascending = True).head (10).to_csv (float_format = '%.5f'))

        
        # Results merged by label
        print ("by label")
        print ("----------")
        print ("top")
        print (features_df_merged.sort_values (by = 'Coefficient', ascending = False).head (10)[unique_labels].to_csv (float_format = '%.5f'))
        
        print ("worst")
        print (features_df_merged.sort_values (by = 'Coefficient', ascending = True).head (10)[unique_labels].to_csv (float_format = '%.5f'))


        # @var ig_path_by_label String
        ig_path = dataset.get_working_dir (dataset.task, 'information-gain', 'by-coefficient.csv')


        # @var ig_name String
        ig_name = dataset.get_working_dir (dataset.task, 'information-gain', 'by-name.csv')
        
        
        # @var ig_path_by_label String
        ig_path_by_label = dataset.get_working_dir (dataset.task, 'information-gain', 'by-label-all.csv')


        # @var ig_path_by_label_name String
        ig_path_by_label_name = dataset.get_working_dir (dataset.task, 'information-gain', 'by-label-and-name-all.csv')


        # @var ig_path_by_category_best String
        ig_path_by_category_best = dataset.get_working_dir (dataset.task, 'information-gain', 'by-category-5.csv')


        # @var ig_path_by_category_all String
        ig_path_by_category_all = dataset.get_working_dir (dataset.task, 'information-gain', 'by-category-all.csv')


        # Save all features ranked by coefficient
        best_features_indexes.sort_values (by = 'Coefficient', ascending = False).to_csv (ig_path, float_format = '%.5f')


        # Save all features ranked by coefficient
        best_features_indexes.sort_index (ascending = True).to_csv (ig_name, float_format = '%.5f')


        # Save all features by label ranked by name
        features_df_merged.sort_index (ascending = True)[unique_labels].to_csv (ig_path_by_label_name, float_format = '%.5f')

        
        # Save all features by label ranked by coefficient
        features_df_merged.sort_values (by = 'Coefficient', ascending = False)[unique_labels].to_csv (ig_path_by_label, float_format = '%.5f')
        
        
        # Group features by category
        best_features_indexes.sort_values (
            by = ['category', 'Coefficient'], 
            ascending = [True, False]
        ).groupby ('category').head (5).to_csv (ig_path_by_category_best, float_format = '%.5f')
        
        
        # Group features by category
        best_features_indexes.sort_values (
            by = ['category', 'Coefficient'], 
            ascending = [True, False]
        ).groupby ('category').head (1000).to_csv (ig_path_by_category_all, float_format = '%.5f')
        
        
        

if __name__ == "__main__":
    main ()
