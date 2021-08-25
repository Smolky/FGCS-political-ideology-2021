"""
    Generate LF features per category
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn import preprocessing

from tqdm import tqdm
from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from features.FeatureResolver import FeatureResolver


def main ():

    # var parser
    parser = DefaultParser (description = 'Generate LF categories')
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    
    
    # Determine if we need to use the merged dataset or not
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')


    # @var df Ensure if we already had the data processed
    df = dataset.get ()

    
    # @var feature_resolver FeatureResolver
    feature_resolver = FeatureResolver (dataset)
    

    # @var features_cache String Retrieve the unprocessed features dataframe
    features_cache = dataset.get_working_dir (dataset.task, 'lf.csv')
    
    
    # Skip as those features does not exist
    if not Path (features_cache).is_file ():
        print ("LF features not found")
        sys.exit ()
        
        
    # @var transformer Retrieve the transformer for this feature set
    transformer = feature_resolver.get ('lf', features_cache)
        
        
    # @var features_df DataFrame Retrieves all the features (they were previosly cached)
    features_df = transformer.transform ([]);
    
    
    # @var categories List
    categories = list (set ([column.split ('-')[0] for column in features_df.columns if '-' in column]))    
    
    
    # Select the subset of the features
    for category in tqdm (categories):
        features_df[[col for col in features_df.columns if col.startswith (category + '-')]].to_csv (dataset.get_working_dir ('lf_' + category + '.csv'), index = False)

if __name__ == "__main__":
    main ()