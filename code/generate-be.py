"""
    Compile BERT features (no fine-tunned)
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import config
import bootstrap
import os.path

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser

from features.BertEmbeddingsTransformer import BertEmbeddingsTransformer
from features.TokenizerTransformer import TokenizerTransformer


def main ():
    
    # var parser
    parser = DefaultParser (description = 'Generate fixed BERT embeddings')
    

    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var dataset Dataset
    dataset = resolver.get (args.dataset, args.corpus, args.task, args.force)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    # @var df Dataframe
    df = dataset.get ()


    # @var language String
    language = dataset.get_dataset_language ()
    
    
    # @var huggingface_model String
    huggingface_model = 'dccuchile/bert-base-spanish-wwm-uncased' if language == 'es' else 'bert-base-uncased'
        
    be_transformers = BertEmbeddingsTransformer (huggingface_model, cache_file = dataset.get_working_dir (args.task, 'be.csv'), field = 'tweet_clean_lowercase')
    print (be_transformers.transform (df))

if __name__ == "__main__":
    main ()