"""
    Generate the linguistic features with UMUTextStats
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import config
import bootstrap
import os.path

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser

from features.LinguisticFeaturesTransformer import LinguisticFeaturesTransformer


def main ():
    
    # var parser
    parser = DefaultParser (description = 'Generate the linguistic features')
    

    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var dataset Dataset
    dataset = resolver.get (args.dataset, args.corpus, args.task, args.force)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    # @var df Dataframe
    df = dataset.get ()

    
    # @var lf_transformers LF
    lf_transformers = LinguisticFeaturesTransformer (cache_file = dataset.get_working_dir ('lf.csv'))
    print (lf_transformers.transform (df))



if __name__ == "__main__":
    main ()