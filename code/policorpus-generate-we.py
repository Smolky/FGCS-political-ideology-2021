"""
    Generate BERT embeddings from the fine-tuned model
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from features.TokenizerTransformer import TokenizerTransformer


def main ():

    # var parser
    parser = DefaultParser (description = 'Generate Word Embeddings (WE) features for the PoliCorpus')
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
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
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')


    # @var df The original dataframe
    df = dataset.get ()


    # @var cache_file String
    cache_file = dataset.get_working_dir (args.task, 'we.csv')
    
    
    # @var we_transformers WE
    we_transformers = TokenizerTransformer (
        cache_file = cache_file, 
        field = 'tweet_clean_lowercase'
    )
    
    
    # Get a custom tokenizer
    we_transformers.load_tokenizer_from_disk (policorpus_dataset.get_working_dir (args.task, 'we_tokenizer.pickle'))
    
    print (policorpus_dataset.get_working_dir (args.task, 'we_tokenizer.pickle'))
    
    # Print the embeddings generated
    print (we_transformers.transform (df))  

    
if __name__ == "__main__":
    main ()
