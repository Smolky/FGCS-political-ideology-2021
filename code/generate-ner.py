"""
    Compile a dataset and all its features
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import config
import bootstrap
import os.path

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser


from pipeline.Tagger import Tagger


def main ():
    
    # var parser
    parser = DefaultParser (description = 'Generate NER and PoS')
    

    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var dataset Dataset
    dataset = resolver.get (args.dataset, args.corpus, args.task, args.force)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    # @var df Dataframe
    df = dataset.get ()

    
    # @var tagger
    tagger = Tagger (dataset.get_dataset_language ())


    # Attach POS and NER info
    df = tagger.get (df, field = 'tweet_clean')
    
    
    # Save the dataset tagged
    dataset.save_on_disk (df)
    

if __name__ == "__main__":
    main ()