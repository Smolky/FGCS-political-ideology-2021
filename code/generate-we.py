"""
    Generate word embeddings
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import config
import bootstrap
import os.path

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser

from features.TokenizerTransformer import TokenizerTransformer


def main ():
    
    # var parser
    parser = DefaultParser (description = 'Generate Word Embeddings (WE) features')
    

    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var dataset Dataset
    dataset = resolver.get (args.dataset, args.corpus, args.task, args.force)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    
    # @var df Dataframe
    df = dataset.get ()
    
    
    # @var train_df DataFrame Get training split
    train_df = dataset.get_split (df, 'train')
    
    
    # @var cache_file String
    cache_file = dataset.get_working_dir (args.task, 'we.csv')
    
    
    # @var we_transformers WE
    we_transformers = TokenizerTransformer (
        cache_file = cache_file, 
        field = 'tweet_clean_lowercase'
    )
    we_transformers.fit (train_df)
    we_transformers.save_tokernizer_on_disk (dataset.get_working_dir (args.task, 'we_tokenizer.pickle'))


    # @var tokenizer Tokenizer
    tokenizer = we_transformers.get_tokenizer ()
    
    
    # Print the embeddings generated
    print (we_transformers.transform (df))    
    
    print ()
    print ("word counts")
    print ("-----------")
    print (tokenizer.word_counts)
    
    print ()
    print ("document counts")
    print ("-----------")
    print (tokenizer.document_count)
    
    print ()
    print ("word index")
    print ("-----------")
    print (len (tokenizer.word_index))
    
    print ()
    print ("word docs")
    print ("-----------")
    print (len (tokenizer.word_docs))
    

if __name__ == "__main__":
    main ()