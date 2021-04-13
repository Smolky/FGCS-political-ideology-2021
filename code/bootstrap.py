"""
Bootstrapping

Contains a lot of stuff that is reused continously in all the packages

@author José Antonio García-Díaz <joseantonio.garcia8@um.es>
@author Ricardo Colomo-Palacios <ricardo.colomo-palacios@hiof.no>
@author Rafael Valencia-Garcia <valencia@um.es>
"""

# For reproductibility, we set a global seed that will be shared on tensorflow, numpy, etc...
seed = 0 


# Import OS to set the environment variables
import os


# Configure TF and CUDA for reproductilibty and to avoid INFO and WARNING messages
os.environ['PYTHONHASHSEED'] = str (seed)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'
os.environ['OMP_NUM_THREADS'] = '32'
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Import libreries that can generate stochastic behaviour
import random
import numpy as np
import pandas as pd
import torch
import tensorflow


# Seed random generators
# @todo improve https://scikit-learn.org/stable/common_pitfalls.html#randomness
random.seed (seed)
np.random.seed (seed)
tensorflow.compat.v1.set_random_seed (seed)
torch.manual_seed (seed)


# Keras and tensorflow parallelism
session_conf = tensorflow.compat.v1.ConfigProto (intra_op_parallelism_threads = 8, inter_op_parallelism_threads = 8)
sess = tensorflow.compat.v1.Session (graph = tensorflow.compat.v1.get_default_graph (), config = session_conf)
tensorflow.compat.v1.keras.backend.set_session (sess)



# Configure pretty print options
np.set_printoptions (formatter = {'float_kind': "{:.5f}".format})

