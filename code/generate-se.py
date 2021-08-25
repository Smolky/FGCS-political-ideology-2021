"""
    Generate SE FastText
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import config
import bootstrap
import os.path

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser

from features.SentenceEmbeddingsTransformer import SentenceEmbeddingsTransformer


def main ():
    
    # var parser
    parser = DefaultParser (description = 'Generate Sentence Embeddings')
    

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

    
    # @var fasttext_model SE
    fasttext_model = config.pretrained_models[language]['fasttext']['binary']
    se_transformers = SentenceEmbeddingsTransformer (fasttext_model, cache_file = dataset.get_working_dir (args.task, 'se.csv'), field = 'tweet_clean_lowercase')
    print (se_transformers.transform (df))
    

if __name__ == "__main__":
    main ()