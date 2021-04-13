"""
    Train a dataset from specific features
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Ricardo Colomo-Palacios <ricardo.colomo-palacios@hiof.no>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys

from pathlib import Path

from tqdm import tqdm

from dlsdatasets.DatasetResolver import DatasetResolver
from dlsmodels.ModelResolver import ModelResolver
from features.FeatureResolver import FeatureResolver
from utils.Parser import DefaultParser

def main ():
    
    # var parser
    parser = DefaultParser (description = 'Train dataset')
    parser.add_argument ('--model', 
        dest = 'model', 
        default = 'machine-learning', 
        help = 'Select the family or algorithms to evaluate', 
        choices = ['machine-learning', 'deep-learning', 'transformers', 'transformers-lf']
    )
    
    parser.add_argument ('--features', 
        dest = 'features', 
        default = 'all', 
        help = 'Select the family or features to evaluate', 
        choices = ['all', 'lf', 'se', 'be', 'we']
    )

    

    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()

    
    # @var dataset Dataset
    dataset = resolver.get (args.dataset, args.corpus, args.task, False)
    
    
    # Determine if we need to use the merged dataset or not
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    # @var df DataFrame
    df = dataset.get ()

    
    # @var model Model
    model = model_resolver.get (args.model)
    model.set_dataset (dataset)
    model.is_merged (dataset.is_merged)
    
    
    # @var feature_resolver FeatureResolver
    feature_resolver = FeatureResolver (dataset)
    
    
    # @var available_features List
    available_features = model.get_available_features () if args.features == 'all' else [args.features]
    
    
    # Indicate which features we are loading
    print ("loading features...")
    

    # Set the features
    for feature_set in available_features:
    
        # @var features_cache String The file where the features are stored
        if feature_set == 'lf':
            features_cache = dataset.get_working_dir (args.task, feature_set + '_minmax_ig.csv')
        else:
            features_cache = dataset.get_working_dir (args.task, feature_set + '_ig.csv')


        if not Path (features_cache).is_file():
            features_cache = dataset.get_working_dir (args.task, feature_set + '.csv')

        
        print ("\t" + features_cache)
        
        
        # Set features
        model.set_features (feature_set, feature_resolver.get (feature_set, features_cache))


    # @var using_official_test boolean
    using_official_test = True if ('evaluate_with_test' in dataset.options and dataset.options['evaluate_with_test']) else False

    
    # Perform the training...
    model.train (args.force, using_official_test)
    
    

if __name__ == "__main__":
    main ()