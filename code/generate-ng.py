"""
    Generate n-grams features
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import config
import bootstrap
import os.path

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser

from features.NGramsTransformer import NGramsTransformer


def main ():
    
    # var parser
    parser = DefaultParser (description = 'Generate word or character n-grams (NG) features')
    

    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var dataset Dataset
    dataset = resolver.get (args.dataset, args.corpus, args.task, False)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    
    # @var df Dataframe
    df = dataset.get ()
    
    
    # @var train_df DataFrame Get training split
    train_df = dataset.get_split (df, 'train')
    
    
    # @var cache_file String
    cache_file = dataset.get_working_dir (args.task, 'ng.csv')
    
    
    # @var ng_transformers NG
    ng_transformers = NGramsTransformer (
        cache_file = cache_file, 
        field = 'tweet_clean_lowercase'
    )
    ng_transformers.fit (train_df)
    ng_transformers.save_vectorizer_on_disk (dataset.get_working_dir ('ng_vectorizer.pickle'))


    # @var vectorizer VectorizerTransformer
    vectorizer = ng_transformers.get_vectorizer ()
    
    
    # Print the embeddings generated
    print (ng_transformers.transform (df))    
    

    

if __name__ == "__main__":
    main ()