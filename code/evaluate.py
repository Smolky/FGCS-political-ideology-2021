"""
    Evaluate a new text or a test dataset
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Ricardo Colomo-Palacios <ricardo.colomo-palacios@hiof.no>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import argparse
import pandas as pd

from pathlib import Path

from dlsdatasets.DatasetResolver import DatasetResolver
from dlsmodels.ModelResolver import ModelResolver
from features.FeatureResolver import FeatureResolver
from utils.Parser import DefaultParser
from pipeline.Tagger import Tagger


def main ():

    # var parser
    parser = DefaultParser (description = 'Evaluate the dataset')
    
    
    # Add model
    parser.add_argument ('--model', 
        dest = 'model', 
        default = 'machine-learning', 
        help = 'Select the family or algorithms to evaluate', 
        choices = ['machine-learning', 'deep-learning', 'transformers', 'transformers-lf']
    )
    
    
    # Add features
    parser.add_argument ('--features', 
        dest = 'features', 
        default = 'all', 
        help = 'Select the family or features to evaluate', 
        choices = ['all', 'lf', 'se', 'be', 'we']
    )
    
    
    # Add features
    parser.add_argument ('--source', 
        dest = 'source', 
        default = 'test', 
        help = 'Determines the source to evaluate', 
        choices = ['train', 'test', 'val', 'file', 'directinput']
    )
    
    
    # Additional input options for file and direct input
    parser.add_argument ('--file', type = argparse.FileType ('r'))
    parser.add_argument ('--text', dest = 'text', default = '', help = 'Text text to evaluate')
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # Determine external datasets (files and directinput)
    if 'file' == args.source:
        with args.file as file:
            _df = pd.read_csv (file, header = 0)
    
    elif 'directinput' == args.source:
        _df = pd.DataFrame ({'tweet': args.text, 'label': ''}) if args.text else None
    
    
    # @var evaluating_self_dataset
    evaluating_self_dataset = args.source in ['train', 'test', 'val']
    
    
    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    
    
    # Change the filename of the dataset
    if not evaluating_self_dataset:
        
        # Change the filename of the dataset
        dataset.filename = dataset.get_working_dir ("evaluate.csv")
    
        
        # Load the dataset if it has previously preprocessed
        if os.path.isfile (dataset.filename):
            dataset.df = pd.read_csv (dataset.filename, header = 0, sep = ",")
    
        else:
            dataset.df = _df
    
    else:
        # Determine if we need to use the merged dataset or not
        dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
        

    
    # @var df Ensure if we already had the data processed
    df = dataset.get ()
    
    
    # @var model Model
    model = model_resolver.get (args.model)
    model.set_dataset (dataset)
    model.is_merged (dataset.is_merged)


    # @var feature_resolver FeatureResolver
    feature_resolver = FeatureResolver (dataset)
    
    
    # Preprocess tweet, only for new texts
    if not 'tweet_clean' in df.columns:
        df['tweet_clean'] = df['tweet']
        df = dataset.preprocess (df, field = 'tweet_clean')
        dataset.save_on_disk (df)
    
    
    # Tag, only for new texts
    if not 'tagged_ner' in df.columns:
    
        # @var tagger
        tagger = Tagger (dataset.get_dataset_language ())


        # Attach POS and NER info
        df = tagger.get (df, field = 'tweet_clean')
        
        
        # Save the dataset tagged
        dataset.save_on_disk (df)

    
    # Replace the dataset to contain only the test or val-set
    if args.source in ['train', 'val', 'test']:
        dataset.df = dataset.get_split (df, args.source)
    
    
    # @var available_features List
    available_features = model.get_available_features () if args.features == 'all' else [args.features]
    
    
    # Iterate over all available features
    for feature_set in available_features:
    
        # @var transformer Transformer
        if evaluating_self_dataset:
                
            # @var features_cache String The file where the features are stored
            if feature_set == 'lf':
                features_cache = dataset.get_working_dir (args.task, feature_set + '_minmax_ig.csv')
            else:
                features_cache = dataset.get_working_dir (args.task, feature_set + '_ig.csv')


            if not Path (features_cache).is_file():
                features_cache = dataset.get_working_dir (args.task, feature_set + '.csv')

            
            # @var transformer
            transformer = feature_resolver.get (feature_set, cache_file = features_cache)
            
            
            print ("\t" + features_cache)
            
        
        
        # Get the features for the new dataset
        else:
        
            # @var transformer
            transformer = feature_resolver.get (feature_set, cache_file = dataset.get_working_dir ('evaluate-' + feature_set + '.csv'))
            
        
            # Load the tokenizer to ensure we use the same data
            if 'we' == feature_set:
                transformer.load_tokenizer_from_disk (dataset.get_working_dir ('we_tokenizer.pickle')) 
            
            # Get the features
            transformer.transform (df)
    
    
        # Set the features in the model
        model.set_features (feature_set, transformer)

    
    # Perform the training...
    model.predict (using_official_test = evaluating_self_dataset)
    


if __name__ == "__main__":
    main ()
    
    
    