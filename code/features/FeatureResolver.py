import config

from .SentenceEmbeddingsTransformer import SentenceEmbeddingsTransformer
from .BertEmbeddingsTransformer import BertEmbeddingsTransformer
from .LinguisticFeaturesTransformer import LinguisticFeaturesTransformer
from .TokenizerTransformer import TokenizerTransformer

class FeatureResolver ():
    """
    FeatureResolver
    
    Determines which feature set should I load
    """

    def __init__ (self, dataset):
        """
        @param dataset DatasetBase
        """
        self.dataset = dataset
        

    def get (self, features, cache_file):
    
        # @var fasttext_model String
        fasttext_model = config.pretrained_models[self.dataset.get_dataset_language ()]['fasttext']['binary']
        
        
        # @var huggingface_model String
        huggingface_model = 'dccuchile/bert-base-spanish-wwm-uncased'
        
        
        if 'lf' == features:
            return LinguisticFeaturesTransformer (cache_file = cache_file)

        if 'se' == features:
            return SentenceEmbeddingsTransformer (fasttext_model, cache_file = cache_file, field = 'tweet_clean')

        if 'be' == features:
            return BertEmbeddingsTransformer (huggingface_model, cache_file = cache_file, field = 'tweet_clean')

        if 'we' == features:
            return TokenizerTransformer (cache_file = cache_file, field = 'tweet_clean')
