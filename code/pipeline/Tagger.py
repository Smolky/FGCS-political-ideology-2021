"""
    Generate POS Tagger
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import itertools
import os.path
import dask as da
import numpy as np
import pandas as pd
import stanza
import math
import string

import dask.dataframe as ddf
import multiprocessing as mp

import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dask.diagnostics import ProgressBar



class Tagger ():
    """
    Tagger
    """
    
    def __init__ (self, language):
        self.language = language
    
    def get (self, df, field = 'tweet'):
    
        # Configure Stanza
        stanza.download (self.language)
        nlp = stanza.Pipeline (lang = self.language, processors = 'tokenize,mwt,pos,ner')
        
        
        # @var tagged_pos
        tagged_pos = []
        
        
        # @var tagged_ner
        tagged_ner = []
        
        
        # @var f_pos lambda
        f_pos = lambda x: x['text'] + '__(' + (x['xpos'] if 'xpos' in x else x['upos']) + ")" + (('(' + x['feats'] + ')') if 'feats' in x else '') if isinstance (x['id'], int) else ''
        
        
        # @var f_ner lambda
        f_ner = lambda x: x['type'] + '(' + x['text'] + ')'
        
        
        def get_pos_and_ner (document):
            
            # @var sentences
            sentences = nlp (document).sentences
            
            
            # For each sentence we get its token. 
            # In order to avoid using extra memory, all operations are within comprension list
            # (1) check if document if a valid string
            # (2) use NLP to get its sentences, words, and tokens
            # (3) Apply lambda for each word to extract the information we need
            # (4) Join all sentences with a "." sign
            # POS
            return [
                ', '.join ([', '.join ([f_pos (word.to_dict ()) for word in sent.words]) for sent in sentences]),
                ', '.join ([', '.join ([f_ner (ent.to_dict ()) for ent in sent.ents]) for sent in sentences])
            ]
        
        
        # @var int npartitions
        npartitions = mp.cpu_count () - 1 if len (df) >= 100 else 1
        
        
        # Divide the dataframe for paralelization
        df_dask = ddf.from_pandas (df, npartitions = npartitions)
        
        
        # @var operation For paralelisation
        operation = df_dask[field].apply (lambda document: get_pos_and_ner (document) if document else ['', ''])
        
        
        # Show progressbar while we are fetching the results...
        with ProgressBar ():
            response = operation.compute ()
            
        
        # Attach response
        df.loc[:, 'tagged_pos'] = [item[0] for item in response]
        df.loc[:, 'tagged_ner'] = [item[1] for item in response]
        
        
        # Return dataframe
        return df


def main ():
    
    # @var tagger
    tagger = Tagger ('es')
    
    
    df = tagger.get (pd.DataFrame ({'tweet_clean': ['Satán, harto de contestar estúpidas ouijas de adolescentes']}), field = 'tweet_clean')
    print (df)
    
    df = tagger.get (pd.DataFrame ({'tweet_clean': ['Satán, harto de contestar estúpidas ouijas de adolescentes']}), field = 'tweet_clean')
    print (df)

    

if __name__ == "__main__":
    main ()
