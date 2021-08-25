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


def main ():
    
    # var parser
    parser = DefaultParser (description = 'Generates cleaned versions of the texts')
    

    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var dataset Dataset
    dataset = resolver.get (args.dataset, args.corpus, args.task, args.force)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    
    # @var df Dataframe
    df = dataset.get ()
    

    # Copy tweet
    df['tweet_clean'] = df['tweet']
    
    
    # Preprocess
    df = dataset.preprocess (df, field = 'tweet_clean')


    # Copy clean version to produce the lowercase one
    df['tweet_clean_lowercase'] = df['tweet_clean']
    df = dataset.preprocess (df, pipeline = ['to_lower'], field = 'tweet_clean_lowercase')
    
    print (df['tweet_clean_lowercase'])

    
    # Save the dataset tagged
    dataset.save_on_disk (df)
    

if __name__ == "__main__":
    main ()