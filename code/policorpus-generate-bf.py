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
from features.BertEmbeddingsTransformer import BertEmbeddingsTransformer


def main ():

    # var parser
    parser = DefaultParser (description = 'Generate BERT embeddings from the finetuned model')
    
    
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


    # @var df El dataframe original (en mi caso es el dataset.csv)
    df = dataset.get ()


    # @var cache_file String
    cache_file = dataset.get_working_dir (dataset.task, 'bf.csv')


    # @var huggingface_model String
    huggingface_model = policorpus_dataset.get_working_dir (dataset.task, 'models', 'bert', 'bert-finetunning')
    
    
    # Create 
    be_transformers = BertEmbeddingsTransformer (
        huggingface_model, 
        cache_file = cache_file, 
        field = 'tweet_clean_lowercase'
    )

    print (be_transformers.transform (df))

    
if __name__ == "__main__":
    main ()
