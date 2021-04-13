"""
    Compile a dataset and all its features
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Ricardo Colomo-Palacios <ricardo.colomo-palacios@hiof.no>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import config
import bootstrap
import os.path

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from utils.LabelsDistribution import LabelsDistribution
from utils.WordsCloud import WordsCloud
from utils.CorpusStatistics import CorpusStatistics

from features.SentenceEmbeddingsTransformer import SentenceEmbeddingsTransformer
from features.BertEmbeddingsTransformer import BertEmbeddingsTransformer
from features.LinguisticFeaturesTransformer import LinguisticFeaturesTransformer
from features.TokenizerTransformer import TokenizerTransformer

from pipeline.Tagger import Tagger


def main ():
    
    # var parser
    parser = DefaultParser (description = 'Compile dataset')
    

    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var dataset Dataset
    dataset = resolver.get (args.dataset, args.corpus, args.task, args.force)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    # @var df Dataframe
    df = dataset.get ()
    
    
    # @for testing
    # df = df.iloc[0:200, :]
    # df = df[df['user'] == "8d8e558714570287eeab308475a5d12d"]
    # df = df.reset_index ()

    
    # Preprocess the tweets before retrieve the rest of the features
    # We keep lowercase and uppercase versions for several reasons:
    # (1) use case and uncase models
    # (2) use pos tagger with case versions of the texts
    if not 'tweet_clean' in df.columns or not 'tweet_clean_lowercase' in df.columns:
    
        print ()
        print ("Cleaning...")
        df['tweet_clean'] = df['tweet']
        df = dataset.preprocess (df, field = 'tweet_clean')

        df['tweet_clean_lowercase'] = df['tweet_clean']
        df = dataset.preprocess (df, pipeline = ['to_lower'], field = 'tweet_clean_lowercase')
        
        # Save the dataset tagged
        dataset.save_on_disk (df)
    
    
    # @var tags
    if not 'tagged_ner' in df.columns:
    
        print ()
        print ("compile NER and POS...")

    
        # @var tagger
        tagger = Tagger (dataset.get_dataset_language ())
    
    
        # Attach POS and NER info
        df = tagger.get (df, field = 'tweet_clean')
        
        
        # Save the dataset tagged
        dataset.save_on_disk (df)
    
    
    # @var train_df DataFrame Get training split
    train_df = dataset.get_split (df, 'train')
        

    # @var we_transformers WE
    if not os.path.exists (dataset.get_working_dir ('we.csv')):
        we_transformers = TokenizerTransformer (cache_file = dataset.get_working_dir ('we.csv'), field = 'tweet_clean_lowercase')
        we_transformers.fit (train_df)
        print (we_transformers.transform (df))
        we_transformers.save_tokernizer_on_disk (dataset.get_working_dir ('we_tokenizer.pickle'))

    
    # @var lf_transformers LF
    lf_transformers = LinguisticFeaturesTransformer (cache_file = dataset.get_working_dir ('lf.csv'))
    print (lf_transformers.transform (df))


    
    # @var fasttext_model SE
    fasttext_model = config.pretrained_models[dataset.get_dataset_language ()]['fasttext']['binary']
    se_transformers = SentenceEmbeddingsTransformer (fasttext_model, cache_file = dataset.get_working_dir ('se.csv'), field = 'tweet_clean_lowercase')
    print (se_transformers.transform (df))

    
    print ()
    print ("compile BE")

    
    # @var huggingface_model String
    huggingface_model = 'dccuchile/bert-base-spanish-wwm-uncased'
    be_transformers = BertEmbeddingsTransformer (huggingface_model, cache_file = dataset.get_working_dir ('be.csv'), field = 'tweet_clean_lowercase')
    print (be_transformers.transform (df))
    

if __name__ == "__main__":
    main ()