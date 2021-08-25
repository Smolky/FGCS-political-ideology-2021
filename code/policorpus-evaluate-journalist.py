"""
    PoliCorpus evaluate journalists
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""


import os
import sys
import argparse
import pandas as pd
import csv
import pickle

from pathlib import Path
from tqdm import tqdm

from dlsdatasets.DatasetResolver import DatasetResolver
from dlsmodels.ModelResolver import ModelResolver
from features.FeatureResolver import FeatureResolver
from utils.Parser import DefaultParser
from sklearn.metrics import classification_report


def main ():

    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()
    
    
    # var parser
    parser = DefaultParser (description = 'Evaluate journalists with the PoliCorpus model')


    # Add model
    parser.add_argument ('--model', 
        dest = 'model', 
        default = model_resolver.get_default_choice (), 
        help = 'Select the family of algorithms to evaluate', 
        choices = model_resolver.get_choices ()
    )
    
    
    # @var choices List of list 
    choices = FeatureResolver.get_feature_combinations (['lf', 'se', 'be', 'we', 'ne', 'cf', 'bf', 'pr', 'ng'])
    
    
    # Add features
    parser.add_argument ('--features', 
        dest = 'features', 
        default = 'all', 
        help = 'Select the family or features to evaluate', 
        choices = ['all'] + ['-'.join (choice) for choice in choices]
    )
    
    
    # Add features
    parser.add_argument ('--architecture', 
        dest = 'architecture', 
        default = '', 
        help = 'Determines the architecture to evaluate', 
        choices = ['', 'dense', 'cnn', 'bigru', 'gru', 'lstm', 'bilstm']
    )
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var architecture String
    architecture = args.architecture
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var policorpus_dataset Dataset
    policorpus_dataset = dataset_resolver.get (
        dataset = 'policorpus', 
        corpus = '2020', 
        task = args.task, 
        refresh = False
    )
    
    
    # Determine if we need to use the merged dataset or not
    policorpus_dataset.filename = policorpus_dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    # @var task_type String
    task_type = policorpus_dataset.get_task_type ()
    
    
    # @var df_policorpus DataFrame
    df_policorpus = policorpus_dataset.get ()
    
    
    # @var journalists_dataset
    journalists_dataset = dataset_resolver.get (
        dataset = 'journalists', 
        corpus = 'policorpus', 
        task = args.task, 
        refresh = False
    )
    journalists_dataset.filename = journalists_dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    # @var df_journalist DataFrame
    df_journalist = journalists_dataset.get ()
    
    
    if '__split' not in df_journalist.columns:
        df_journalist = df_journalist.rename ({
            '__split_author_profiling': '__split'
        }, axis = 'columns')
    

    # @var labels
    labels = policorpus_dataset.get_available_labels ()
    policorpus_dataset.set_available_labels (labels)
    
    
    # This a trick to set the DF of the journalist
    policorpus_dataset.df = df_journalist
    
    
    # @var policorpus_model Model We set the policorpus model in order to use it to predict
    policorpus_model = model_resolver.get (args.model)
    policorpus_model.set_dataset (policorpus_dataset)
    policorpus_model.is_merged (policorpus_dataset.is_merged)
    
    
    # Define the criterion for selecting the best model
    if architecture:
        policorpus_model.set_best_model_criteria ({
            'architecture': architecture
        })
    
    
    # @var group_results_by_user Boolean In some author attribution tasks at tweet level
    #                                    the results should be reported by the mode of the 
    #                                    predictions of users
    group_results_by_user = policorpus_dataset.group_results_by_user ()
    
    
    # @var feature_resolver FeatureResolver We base it on the journalist dataset
    feature_resolver = FeatureResolver (journalists_dataset)
    
    
    # @var feature_combinations List
    feature_combinations = FeatureResolver.get_feature_combinations (policorpus_model.get_available_features ()) if args.features == 'all' else [args.features.split ('-')]

    
    # Configure to use all the dataset
    journalists_dataset.default_split = 'all'


    # Get the results grouped by user according to the task
    if group_results_by_user:
        y_real = df_journalist.groupby (by = 'user', sort = False)['label'].apply (lambda x: x.mode().iloc[0])
        
    else:
        # @var y_real Series
        y_real = df_journalist.dropna (subset = ['label'])['label']


    
    # @var y_real_users Series
    y_real_users = df_journalist.dropna (subset = ['user'])['user'].unique ()
    
    
    def callback (feature_key, y_pred, model_metadata):
        """
        @param feature_key String
        @param y_pred Series
        @param model_metadata Dict
        """
        
        # Reassign y_preds by user, if necessary
        y_pred = df_journalist.assign (y_pred = y_pred) \
            .groupby (by = 'user', sort = False)['y_pred'] \
            .apply (lambda x: x.mode().iloc[0]) if group_results_by_user else y_pred
        
        print ("callback")
        print (len (y_pred))
        print (len (y_real))
        
        
        # @var report DataFrame|None
        report = pd.DataFrame (classification_report (
            y_true = y_real, 
            y_pred = y_pred, 
            digits = 5,
            output_dict = True
        )).T
        
        
        # Adjust the report in a scale from 0 to 100
        report['precision'] = report['precision'].mul (100)
        report['recall'] = report['recall'].mul (100)
        report['f1-score'] = report['f1-score'].mul (100)
    
        print (pd.DataFrame ({
            'user': y_real_users,
            'y_pred': y_pred,
            'y_real': y_real
        }))
        
        print (report.to_csv ())
    
        
    # Load all the available features
    if policorpus_model.has_external_features ():
    
        for features in feature_combinations:
            
            # Indicate which features we are loading
            print ("loading features...")
            
            
            # Load...
            for feature_set in features:

                # @var feature_file String According to the task, we get the ideal file
                feature_file = feature_resolver.get_suggested_cache_file (feature_set, task_type)
                
            
                # @var journalist_features_cache String The file where the features are stored. Note that here we are non
                #                                       selecting the suggested model. We do this on purpose, as we want 
                #                                       to fit and select the features according to the PoliCorpus feature
                #                                       selection process
                journalist_features_cache = journalists_dataset.get_working_dir (args.task, feature_set + '.csv')


                # @var features_journalist Transformer We get the features
                features_journalist = feature_resolver.get (feature_set, cache_file = journalist_features_cache)
                
                
                # For testing...
                print ("...loading " + feature_set + " from " + journalist_features_cache)
                
                
                # @var features_journalist_df Dataframe The features from the journalists
                features_journalist_df = features_journalist.transform ([])
                
                
                # In case on the linguistic features, we need to adjust them
                if feature_set == 'lf':
                
                    # @var variance_selector VarianceSelector
                    with open (policorpus_dataset.get_working_dir (args.task, 'features', 'lf_variance.pkl'), 'rb') as f:
                        variance_selector = pickle.load (f)
                    
                    # @var minmax_transformer MinMax
                    with open (policorpus_dataset.get_working_dir (args.task, 'features', 'lf_minmax.pkl'), 'rb') as f:
                        minmax_transformer = pickle.load (f)

                
                    # @var train_features_with_variance_columns List Get those features without variance
                    train_features_with_variance_columns = list (features_journalist_df.columns[variance_selector.get_support ()])
                    
                    
                    # @var temp_df Dataframe We apply minmax scaler
                    temp_df = pd.DataFrame (minmax_transformer.transform (features_journalist_df), columns = features_journalist_df.columns)
                    
                    
                    # @var policorpus_features_cache String We get the original PoliCorpus featureset in order to get the 
                    #                                       columns to filter. For example, to get the Mutual Information
                    #                                       columns
                    policorpus_features_cache = policorpus_dataset.get_working_dir (args.task, feature_file)
                    

                    # @var features_policorpus Transformer
                    features_policorpus = feature_resolver.get (feature_set, cache_file = policorpus_features_cache)
                    
                    
                    # @var features_policorpus_df Dataframe The features from the policorpus
                    features_policorpus_df = features_policorpus.transform ([])
                    
                    
                    # features_journalist_df DataFrame
                    features_journalist_df = temp_df[features_policorpus_df.columns]
                    
                
                print (features_journalist_df)
                
                
                # Replace cache
                features_journalist.cache = features_journalist_df
                
                
                # Set features
                policorpus_model.set_features (feature_set, features_journalist)
            
            
            # Predict this feature set
            policorpus_model.predict (using_official_test = False, callback = callback)
            
            
            # Clear session
            policorpus_model.clear_session ();

            
    else:
        
        print ("@test")
        
        # Predict this feature set
        policorpus_model.predict (using_official_test = False, callback = callback)


    
    
    
if __name__ == "__main__":
    main ()
