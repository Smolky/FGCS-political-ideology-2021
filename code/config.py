"""
    Configuration of the pathss
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os

from pathlib import Path


# @var certificate String
certificate = str (Path.home ()) + '/certificates/CA.pem'


# @var umucorpusclassifier_api_endpoint String
umucorpusclassifier_api_endpoint = 'https://collaborativehealth.inf.um.es/corpusclassifier/api/'


# @var umutextstats_api_endpoint String
umutextstats_api_endpoint = 'php /home/rafa_pepe/umutextstats/api/umutextstats.php'



# @var base_path String
base_path = Path (os.path.realpath (__file__)).parent.parent 


# @var directories Paths
directories = {
    'datasets': os.path.join (base_path, 'datasets'),
    'postaggers': os.path.join (base_path, 'postaggers'),
    'pretrained': os.path.join (base_path, 'embeddings', 'pretrained'),
    'assets': os.path.join (base_path, 'assets'),
    'cache': os.path.join (base_path, 'cache_dir'),
}


# @var pretrained_models 
pretrained_models = {
    'es': {
        'fasttext': {
            'binary': os.path.join (directories['pretrained'], 'cc.es.300.bin'),
            'vectors': os.path.join (directories['pretrained'], 'cc.es.300.vec'),
        },
        
        'word2vec': {
            'vectors': os.path.join (directories['pretrained'], 'word2vec-sbwc.txt')
        },
        
        'glove': {
            'vectors': os.path.join (directories['pretrained'], 'glove-sbwc.vec')
        },
    },
    
    'en': {
        'fasttext': {
            'binary': os.path.join (directories['pretrained'], 'cc.en.300.bin'),
            'vectors': os.path.join (directories['pretrained'], 'cc.en.300.vec')
        },
        
        'glove': {
            'vectors': os.path.join (directories['pretrained'], 'glove.6b.300d.txt'),
        }
    }
}



pos_taggers = {
    'es': os.path.join (directories['postaggers'], 'spanish-distsim.tagger'),
    'en': os.path.join (directories['postaggers'], 'english-bidirectional-distsim.tagger')
}
