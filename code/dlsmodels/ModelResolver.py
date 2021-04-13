"""
ModelResolver

@author José Antonio García-Díaz <joseantonio.garcia8@um.es>
@author Ricardo Colomo-Palacios <ricardo.colomo-palacios@hiof.no>
@author Rafael Valencia-Garcia <valencia@um.es>
"""

import json
import sys
import os.path


from .ClassicMachineLearningTechniques import ClassicMachineLearningTechniques
from .DeepLearningTechniques import DeepLearningTechniques
from .BertModel import BertModel
from .BertModelWithLF import BertModelWithLF


class ModelResolver ():
    """
    ModelResolver
    """
    
    def get (self, model = 'machine-learning'):
        """
        @param model string
        """
        
        if 'transformers' == model:
            return BertModel ()

        if 'transformers-lf' == model:
            return BertModelWithLF ()

            
        if 'machine-learning' == model:
            return ClassicMachineLearningTechniques ()
            
        if 'deep-learning' == model:
            return DeepLearningTechniques ()
            KerasM