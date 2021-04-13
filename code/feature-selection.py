"""
    To perform feature selection
    
    Right now this script only applies MinMaxScaler and a feature selection
    based on IG. These techniques were applied first for the LF; however, 
    we make another tests to another types of features. Note that we do not 
    apply this feature selection on the tokenizer (we)
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Ricardo Colomo-Palacios <ricardo.colomo-palacios@hiof.no>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif

from tqdm import tqdm
from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from features.FeatureResolver import FeatureResolver


def main ():

    # var parser
    parser = DefaultParser (description = 'Feature selection')
    
    
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
    
    
    # @var indexes Dict the the indexes for each split
    indexes = {split: dataset.get_split (df, split).index for split in ['train', 'val', 'test']}
    
    
    # @var feature_resolver FeatureResolver
    feature_resolver = FeatureResolver (dataset)
    
    
    # @var available_features List
    available_features = ['lf', 'be', 'se']
    
    
    # @var scaler MinMaxScaler
    scaler = preprocessing.MinMaxScaler ()
    
    
    # Get every feature set
    for feature_set in (tqdm (available_features)):
    
        # @var features_cache String Retrieve the features dataframe
        features_cache = dataset.get_working_dir (dataset.task, feature_set + '.csv')
        
        
        # @var transformer Retrieve the transformer for this feature set
        transformer = feature_resolver.get (feature_set, features_cache)
        
        
        # @var features_df DataFrame Retrieves all the features (that were previosly cached)
        features_df = transformer.transform ([]);
        
        
        # Fit the scaler on the train set
        scaler.fit (features_df[features_df.index.isin (indexes['train'])])
    
    
        # @var fitted_features_df DataFrame
        fitted_features_df = pd.DataFrame (scaler.fit_transform (features_df), columns = features_df.columns, index = features_df.index)
        
        
        # Save features into disk
        fitted_features_df.to_csv (
            dataset.get_working_dir (dataset.task, feature_set + '_minmax.csv'), 
            index = False,
            float_format = '%.5f'
        )
        
        
        # Note that we perform feature selection over the unnormalised (not minmaxscaler)
        # for the feature selection
        # @var df_selection DataFrame
        df_selection = fitted_features_df if feature_set == 'lf' else features_df
        df_selection = features_df
        
        
        # @var train_labels List
        train_features = df_selection[df_selection.index.isin (indexes['train'])]
        
        
        # @var train_labels Series
        train_labels = df[df.index.isin (indexes['train'])]['label'].astype ('category').cat.codes
        
        
        # @var coeff_df DataFrame Select best features using information gain
        # @link https://stackoverflow.com/questions/64343345/how-to-select-best-features-in-dataframe-using-the-information-gain-measure-in-s
        coeff_df = pd.DataFrame (
            mutual_info_classif (X = train_features, y = train_labels).reshape (-1, 1), 
            columns = ['Coefficient'], 
            index = df_selection.columns
        )
    
    
        # @var best_features_columns List Filter features excluding the lower quartil
        # @link https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame
        best_features_columns = coeff_df[(coeff_df["Coefficient"] > coeff_df["Coefficient"].quantile (0.25))] \
            .index \
            .tolist ()
    
    
        # Save features into disk (information gain with minmax)
        fitted_features_df[best_features_columns].to_csv (
            dataset.get_working_dir (dataset.task, feature_set + '_minmax_ig.csv'), 
            index = False, 
            float_format = '%.5f'
        )
        
        
        # Save features into disk (information gain without minmax)
        features_df[best_features_columns].to_csv (
            dataset.get_working_dir (dataset.task, feature_set + '_ig.csv'), 
            index = False, 
            float_format = '%.5f'
        )
    

if __name__ == "__main__":
    main ()
