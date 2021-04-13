"""
    Polar Charts Generation
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Ricardo Colomo-Palacios <ricardo.colomo-palacios@hiof.no>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle

from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif


from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from features.FeatureResolver import FeatureResolver
from sklearn.pipeline import Pipeline, FeatureUnion
from features.TokenizerTransformer import TokenizerTransformer


def main ():

    # var parser
    parser = DefaultParser (description = 'Generates polar charts from the LF')
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')


    # @var df Ensure if we already had the data processed
    df = dataset.get ()
    
    
    # @var feature_resolver FeatureResolver
    feature_resolver = FeatureResolver (dataset)
    
    
    # @var features_cache String
    features_cache = dataset.get_working_dir (args.task, 'lf_minmax.csv')
    
    
    # @var transformer
    transformer = feature_resolver.get ('lf', cache_file = features_cache)
    
    
    # @var features_df DataFrame
    features_df = transformer.transform ([]) \
        .assign (label = df['label'])
    
    
    features_df = features_df.replace (0, np.nan).dropna (axis = 1, how = "all")
    
    
    # @var categories 
    categories = list (set ([column.split ('-')[0] for column in features_df.columns if '-' in column]))
    
    
    # @var unique_labels Series Bind to the label
    unique_labels = sorted (df['label'].unique ().to_list ())
    
    
    # @var features_df_per_class Dict
    features_df_per_class = {label: features_df.loc[features_df['label'] == label] for label in unique_labels}
    
    
    # @var results Dict
    results = {label: {} for label in unique_labels}
    
    
    # Fill the data
    for label in unique_labels:
        results[label] = {category: np.mean (features_df_per_class[label][[col for col in features_df.columns if col.startswith (category + '-')]].mean (axis = 1))  for category in categories }

    
    print (pd.DataFrame (results).sort_index ().to_csv ())
        

    
if __name__ == "__main__":
    main ()
    
    
    