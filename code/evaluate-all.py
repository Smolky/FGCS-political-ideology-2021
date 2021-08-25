"""
    PoliCorpus evaluate all models
    
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
    parser = DefaultParser (description = 'Evaluate all models')


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
    
    # Add features
    parser.add_argument ('--source', 
        dest = 'source', 
        default = 'test', 
        help = 'Determines the source to evaluate', 
        choices = ['all', 'train', 'test', 'val']
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
    
    
    # @var labels
    labels = policorpus_dataset.get_available_labels ()


    # @var full_df DataFrame
    full_df = policorpus_dataset.get ()
    
    
    # Replace the dataset to contain only the test or val-set
    if args.source in ['train', 'val', 'test']:
        policorpus_dataset.default_split = args.source
        
    
    # @var df Ensure if we already had the data processed
    df = policorpus_dataset.get ()
    
    
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


    # @var feature_combinations List
    feature_combinations = FeatureResolver.get_feature_combinations (policorpus_model.get_available_features ()) if args.features == 'all' else [args.features.split ('-')]


    # Get the results grouped by user according to the task
    if group_results_by_user:
        y_real_politicians = df.groupby (by = 'user', sort = False)['label'].apply (lambda x: x.mode().iloc[0])
        
    else:
        # @var y_real_politicians
        y_real_politicians = df['label']


    
    # @var do_journalists Boolean
    do_journalists = 'ideological_binary' in args.task
    
    
    if do_journalists:
    
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
    
    
        # @var feature_resolver FeatureResolver We base it on the journalist dataset
        feature_resolver = FeatureResolver (journalists_dataset)
        
        
        # Configure to use all the dataset
        journalists_dataset.default_split = 'all'
    
    
        # Get the results grouped by user according to the task
        if group_results_by_user:
            y_real = journalists_dataset.get ().dropna (subset = ['label']).groupby (by = 'user', sort = False)['label'].apply (lambda x: x.mode().iloc[0])
            
        else:
            y_real = journalists_dataset.get ().dropna (subset = ['label'])['label']
    
    
    else:
    
        # @var feature_resolver FeatureResolver We base it on the policorpus dataset
        feature_resolver = FeatureResolver (policorpus_dataset)
    

    # @var labels List
    labels = policorpus_dataset.get_available_labels ()
    
    
    
    # @var hyperparameter_df DataFrame|None
    hyperparameter_df = None
    

    # @var politicians_weighted_series List
    politicians_weighted_series = []
    
    
    # @var politicians_macro_series List
    politicians_macro_series = []
    
    
    # Restore
    policorpus_dataset.df = full_df
    
    
    def callback_politicians (feature_key, y_pred, model_metadata):
    
        # Reassign y_preds by user, if necessary
        y_pred = df.assign (y_pred = y_pred) \
            .groupby (by = 'user', sort = False)['y_pred'] \
            .apply (lambda x: x.mode().iloc[0]) if group_results_by_user else y_pred
            
        
        # @var report DataFrame|None
        report = pd.DataFrame (classification_report (
            y_true = y_real_politicians, 
            y_pred = y_pred, 
            digits = 5,
            output_dict = True
        )).T 
        politicians_macro_series.append (report.iloc[-2]['f1-score'])
        politicians_weighted_series.append (report.iloc[-1]['f1-score'])


    # Load all the available features
    for features in feature_combinations:

        # @var feature_key String
        feature_key = '-'.join (features)
        
        
        # @var hyperparameter_df DataFrame
        hyperparameter_df = pd.read_csv (policorpus_dataset.get_working_dir (policorpus_dataset.task, 'models', 'deep-learning', feature_key, 'hyperparameters.csv'))
        hyperparameter_df = hyperparameter_df.rename (columns = {'Unnamed: 0': 'index'})
        
        
        # Testing
        # hyperparameter_df = hyperparameter_df.head (5)
        
        
        # Load the features
        for feature_set in features:

            # @var feature_file String According to the task, we get the ideal file
            feature_file = feature_resolver.get_suggested_cache_file (feature_set, task_type)
            
            
            # @var features_cache String The file where the features are stored
            features_cache = policorpus_dataset.get_working_dir (args.task, feature_file)
            
            
            # If the feautures are not found, get the default one
            if not Path (features_cache, cache_file = "").is_file ():
                features_cache = policorpus_dataset.get_working_dir (args.task, feature_set + '.csv')
            
            
            # Indicate what features are loaded
            print ("\t" + features_cache)
            if not Path (features_cache).is_file ():
                print ("skip...")
                continue
            
            
            # Set features
            policorpus_model.set_features (feature_set, feature_resolver.get (feature_set, cache_file = features_cache))
            
        
        
        # Iterate...
        for index in tqdm (range (len (hyperparameter_df))):
            
            # Set criteria to select the best model
            policorpus_model.set_best_model_criteria ({
                'index': index
            })
            
            
            # Predict this feature set
            policorpus_model.predict (using_official_test = True, callback = callback_politicians)
        
        
        # Clear session
        policorpus_model.clear_session ();    



    if do_journalists:
    
        # @var journalists_weighted_series List
        journalists_weighted_series = []
        
        
        # @var journalists_macro_series List
        journalists_macro_series = []


        # This a trick to set the DF of the journalist
        policorpus_dataset.df = df_journalist
        
        
        def callback_journalists (feature_key, y_pred, model_metadata):
        
            # Reassign y_preds by user, if necessary
            y_pred = df_journalist.assign (y_pred = y_pred) \
                .groupby (by = 'user', sort = False)['y_pred'] \
                .apply (lambda x: x.mode().iloc[0]) if group_results_by_user else y_pred
                
            
            # @var report DataFrame|None
            report = pd.DataFrame (classification_report (
                y_true = y_real, 
                y_pred = y_pred, 
                digits = 5,
                output_dict = True
            )).T 
            journalists_macro_series.append (report.iloc[-2]['f1-score'])
            journalists_weighted_series.append (report.iloc[-1]['f1-score'])
            
        
        # Load all the available features
        for features in feature_combinations:

            # @var feature_key String
            feature_key = '-'.join (features)
            
            
            # Load...
            for feature_set in features:

                # @var feature_file String According to the task, we get the ideal file
                feature_file = feature_resolver.get_suggested_cache_file (feature_set, task_type)
                
            
                # @var journalist_features_cache String The file where the features are stored. Note that here we are non
                #                                       selecting the suggested model. We do this on purpose, as we want 
                #                                       to fit and select the features according to the PoliCorpus feature
                #                                       selection process
                journalist_features_cache = journalists_dataset.get_working_dir (args.task, feature_set + '.csv')


                # @var policorpus_features_cache String We get the original PoliCorpus featureset in order to get the 
                #                                       columns to filter. For example, to get the Mutual Information
                #                                       columns
                policorpus_features_cache = policorpus_dataset.get_working_dir (args.task, feature_file)


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
            
            
            # Iterate...
            for index in tqdm (range (len (hyperparameter_df))):
                
                # Set criteria to select the best model
                policorpus_model.set_best_model_criteria ({
                    'index': index
                })
                
                
                # Predict this feature set
                policorpus_model.predict (using_official_test = False, callback = callback_journalists)
            
            
            # Clear session
            policorpus_model.clear_session ();


    # Assign data
    if do_journalists:
        hyperparameter_df = hyperparameter_df.assign (journalists_macro_series = journalists_macro_series)
        hyperparameter_df = hyperparameter_df.assign (journalists_weighted_series = journalists_weighted_series)
    
    hyperparameter_df = hyperparameter_df.assign (politicians_macro_series = politicians_macro_series)
    hyperparameter_df = hyperparameter_df.assign (politicians_weighted_series = politicians_weighted_series)


    # @var params_filename String
    params_filename = policorpus_dataset.get_working_dir (policorpus_dataset.task, 'models', 'deep-learning', feature_key, 'hyperparameters-full.csv')
    

    # @var hyperparameter_df DataFrame
    hyperparameter_df.to_csv (params_filename, index = True)
    
    features_to_show = ['politicians_macro_series', 'politicians_weighted_series']    
    if do_journalists:
        features_to_show += ['journalists_macro_series', 'journalists_weighted_series']
        
    
    print (hyperparameter_df[features_to_show])
    
    
    
if __name__ == "__main__":
    main ()
