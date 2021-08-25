"""
    Generate tables
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import glob
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
import dlsmodels.utils as models_utils
import json

from pathlib import Path

from sklearn import preprocessing

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from features.FeatureResolver import FeatureResolver
from dlsmodels.ModelResolver import ModelResolver
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def main ():

    # var parser
    parser = DefaultParser (description = 'Generate tables achieved by feature set')
    
    
    # Add model
    parser.add_argument ('--model', 
        dest = 'model', 
        default = 'machine-learning', 
        help = 'Select the family or algorithms to evaluate', 
        choices = ['machine-learning', 'deep-learning', 'transformers']
    )
    
    
    # Add features
    parser.add_argument ('--features', 
        dest = 'features', 
        default = 'all', 
        help = 'Select the family or features to evaluate', 
        choices = ['all', 'lf', 'se', 'be', 'we']
    )
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')


    # @var df Ensure if we already had the data processed
    df = dataset.get ()
    
    
    # @var model Model
    model = model_resolver.get (args.model)
    model.set_dataset (dataset)
    model.is_merged (dataset.is_merged)
    
    
    # @var feature_resolver FeatureResolver
    feature_resolver = FeatureResolver (dataset)
    
    
    # Supply test set
    dataset.df = dataset.get_split (df, 'test')

    
    # @var available_features List
    available_features = model.get_available_features () if args.features == 'all' else [args.features]
    
    
    # Iterate over all available features
    for feature_set in available_features:

        # @var features_cache String The file where the features are stored
        if feature_set == 'lf':
            features_cache = dataset.get_working_dir (args.task, feature_set + '_minmax_ig.csv')
        else:
            features_cache = dataset.get_working_dir (args.task, feature_set + '_ig.csv')


        if not Path (features_cache).is_file():
            features_cache = dataset.get_working_dir (args.task, feature_set + '.csv')

            
        # @var transformer
        transformer = feature_resolver.get (feature_set, cache_file = features_cache)
        
        
        # Set the features in the model
        model.set_features (feature_set, transformer)

    
    # @var rows Dict
    rows = {}
    
    
    # @var confusion_matrices Dict
    confusion_matrices = {}
    
    
    # @var available_labels List
    availabe_labels = dataset.get_available_labels ()
    
    
    # @var labels List Contains all the labels, including the weighted and macro average of the f1-score
    labels = availabe_labels + ['weighted avg', 'macro avg']
    

    def callback (feature_key, y_pred, y_real, models_per_architecture):
        
        if models_per_architecture:
        
            # Iterate over models
            for key in models_per_architecture:
                
                # @var report
                report = classification_report (y_true = y_real, y_pred = models_per_architecture[key], output_dict = True, labels = availabe_labels)
                
                
                print (feature_key)
                print (key)
                print (pd.DataFrame (report).transpose ().to_csv (float_format = '%.5f'))
                

                print (feature_key)
                print (key)
                cm = confusion_matrix (y_pred = models_per_architecture[key], y_true = y_real, labels = availabe_labels)
                
                print (cm)
            
        
        # @var report Dict
        report = classification_report (y_true = y_real, y_pred = y_pred, output_dict = True, labels = availabe_labels)
        
        
        # Attach rows
        rows[feature_key.upper ()] = {label: round (100 * report[label]['f1-score'], 4) for label in labels}
        
        
        # @var cm List Confusion matrix
        cm = confusion_matrix (y_pred = y_pred, y_true = y_real, labels = availabe_labels)
        
        
        # Attach
        confusion_matrices[feature_key.upper ()] = cm

    
    # Perform the training...
    model.predict (using_official_test = True, callback = callback)
        
    
    # @var table_df DataFrame
    table_df = pd.DataFrame (rows).rename_axis ('feature set').transpose ()
    
    
    # @var indexes_with_the_maximum_values Series
    indexes_with_the_maximum_values = table_df.idxmax ()
    
    
    # Change column types
    for col in labels:
        table_df[col] = table_df[col].astype (str)
        
    for col in labels:
        value = table_df.loc[indexes_with_the_maximum_values[col]][col]
        table_df.loc[indexes_with_the_maximum_values[col], [col]] = '\\textbf{' + value + '}'
    

    # Rename tables
    new_column_names = {label: ('F1-' + label) for label in availabe_labels}
    new_column_names['weighted avg'] = 'F1-average'
    new_column_names['macro avg'] = 'F1-macro'
    
    table_df = table_df.rename (columns = new_column_names)
    
    
    # @var label_field String
    label_field = dataset.get_task_options ()['label']
    
    
    # @var caption String
    caption = 'Author profiling: ' + label_field.title ()
    
    
    # @var label String
    label = 'table:' + label_field + '-author-profiling'
    
    
    # @var column_format String
    column_format = 'l' + ('r' * len (labels))
    
        
    # @var table_latex String
    table_latex = table_df.to_latex (
        column_format = column_format, 
        col_space = 10, 
        escape = False, 
        caption = caption, 
        label = label
    )
    print (table_latex)
    
    
    # @var best_feature_set String
    best_feature_set = indexes_with_the_maximum_values['weighted avg']
    
    
    # @var best_cm DataFrame
    best_cm = pd.DataFrame (confusion_matrices[best_feature_set], columns = availabe_labels, index = availabe_labels)
    
    
    # @var column_format String
    column_format = 'l' + ('r' * len (labels))
    
    
    # @var best_cm_latex String
    best_cm_latex = best_cm.to_latex (column_format = column_format)
    print (best_cm_latex)


    # Hyperparameters
    if args.model == 'deep-learning':
    
        # @var hyperparameters_filename String    
        hyperparameters_filename = dataset.get_working_dir (dataset.task, 'models', 'deep-learning', best_feature_set.lower (), 'hyperparameters.csv')
        
        
        # @var hyperparameters_df DataFrame
        hyperparameters_df = pd.read_csv (hyperparameters_filename).sort_values ('val_f1_micro').head (1)


        # @var columns List
        columns = ['architecture', 'shape', 'num_layers', 'first_neuron', 'dropout', 'lr', 'batch_size', 'activation']


        print (hyperparameters_df[columns].T)

    



if __name__ == "__main__":
    main ()
    
    
    