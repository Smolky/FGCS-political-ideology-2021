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
from tqdm import tqdm

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
    
    
    # @var dataset Dataset
    dataset = dataset_resolver.get (
        dataset = 'journalists', 
        corpus = 'policorpus', 
        task = '', 
        refresh = False
    )
    

    # @var df El dataframe original (en mi caso es el dataset.csv)
    df = dataset.get ()
    
    if 'ideological_multiclass.1' in df.columns:
        df = df.drop ('ideological_multiclass.1', axis = 1)

    if 'ideological_multiclass' in df.columns:
        df = df.drop ('ideological_multiclass', axis = 1)
    


    # @var helper_df DataFrame
    helper_df = pd.read_csv (dataset.get_working_dir ('helpers', 'journalists-helper.csv'))
    
    
    # Merge
    df = pd.merge (left = df, right = helper_df[['user', 'ideological_multiclass']], on = 'user')
    
    if 'ideological_multiclass_x' in df.columns:
        df = df.drop (labels = 'ideological_multiclass_x', axis = 1)
        df = df.rename (columns = {
            'ideological_multiclass_y': 'ideological_multiclass'
        })
    
    
    print (df)
    
    dataset.save_on_disk (df)

        
    
if __name__ == "__main__":
    main ()
