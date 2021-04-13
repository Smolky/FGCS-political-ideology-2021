"""
    Utils
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Ricardo Colomo-Palacios <ricardo.colomo-palacios@hiof.no>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import math
import netrc
import requests
import os
import random
import config
import numpy as np
import pandas as pd

from io import StringIO
from requests.auth import AuthBase
from pathlib import Path
from tqdm import tqdm

def get_project_root () -> Path:
    return Path (__file__).parent

def pd_onehot (df, feature):
    """
    pd_onehot
    
    @link https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
    """
    dummies = pd.get_dummies (df[[feature]])
    res = pd.concat([df, dummies], axis=1)
    return (res)
    
    
    
def get_neurons_per_layer (shape, number_of_layers, first_neuron):
    """
    Return the number of neurons per layer for MLP according to a shape
    
    @param shape
    @param number_of_layers
    @param first_neuron
    
    @link https://mikkokotila.github.io/slate/#diamond
    """
    
    # @var neurons_per_layer List Contains a list of the neurons per layer. Used for the shape
    neurons_per_layer = []    
    
    # Calculate the number of neurons per layer
    # In 'funnel' shape, the number of layers is decreasing
    if (shape == 'funnel'):
        for i in range (1, number_of_layers + 1):
            neurons_per_layer.append (round (first_neuron / i))

    
    # In 'rhombus' the first layer equals to 1 and the next layers slightly increase until the middle one 
    # which equals to neuron size
    if (shape == 'rhombus'):
        
        # Init the neurons to 1
        neurons_per_layer = [1] * number_of_layers
        
        
        # Iterate only in the first half of the array
        for first_index in range (round (number_of_layers / 2)):
            
            # @var half int The central point of the array
            half = math.ceil (number_of_layers / 2) - 1
            
            
            # @var distance int The distance between the half and the first index
            distance = abs (first_index - half)
            
            
            # @var mirror_index int The opposite index
            mirror_index = half + distance
            
            
            # Update opposite index with the same value
            neurons_per_layer[first_index] = int (first_neuron / pow (2, distance))
            neurons_per_layer[mirror_index] = neurons_per_layer[first_index]
    
    
        # Ajust even and odd networks
        if number_of_layers % 2 != 0:
            neurons_per_layer[math.floor (number_of_layers / 2)] = first_neuron
        else:
            neurons_per_layer[-1] = neurons_per_layer[-2]


    # In 'lfunnel' the first half of the layers have the value of neuron_max
    if (shape == 'lfunnel'):
    
        # Init the neurons to 1
        neurons_per_layer = [first_neuron] * number_of_layers
        
        
        # Iterate over all the layers
        for index in range (number_of_layers):
            
            # Get half position
            half = math.ceil (number_of_layers / 2) - 2
            
            
            # Set value
            neurons_per_layer[index] = round (first_neuron / (pow (2, index - half))) if index > half else first_neuron
    
    
    # In 'brick' all neurons have the same number of neurons
    if (shape == 'brick'):
        neurons_per_layer = [first_neuron] * number_of_layers
    
    
    # In 'diamond' the shape is similar to rhombus but with open first half
    if (shape == 'diamond'):
        
        # Init the neurons to 1
        neurons_per_layer = [1] * number_of_layers
        
        
        # Iterate over the half of layers
        for first_index in range (round (number_of_layers / 2)):
            
            # @var half int The central point of the array
            half = math.ceil (number_of_layers / 2) - 1
            
            
            # Get the distance
            distance = abs (first_index - half)
            
            
            # @var mirror_index int The opposite index
            mirror_index = half + distance
            
            
            # Update opposite index with the same value
            neurons_per_layer[first_index] = int (first_neuron / 2)
            neurons_per_layer[mirror_index] = int (first_neuron / pow (2, distance))

        # Ajust even and odd networks
        if number_of_layers % 2 != 0:
            neurons_per_layer[math.floor (number_of_layers / 2)] = first_neuron
        else:
            neurons_per_layer[-1] = neurons_per_layer[-2]
            
    
    # In 'triangle' the shape is similar to a funnel but reversed
    if (shape == '3angle'):
    
        # Init the neurons to 1
        for i in range (1, number_of_layers + 1):
            neurons_per_layer.append (round (first_neuron / i))
        neurons_per_layer = neurons_per_layer[::-1]
        
        
    return neurons_per_layer
    
    
"""
   get_embedding_matrix
   
   @param key String The key of the pretained word embedding
   @param tokenizer Tokenizer
   @param dataset Dataset
   @param embedding_dim int
   @param force Boolean
   @param lang String
   
   @todo https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings
   
   @link https://realpython.com/python-keras-text-classification/#your-first-keras-model
"""

def get_embedding_matrix (key, tokenizer, dataset, embedding_dim = 300, force = False, lang = 'es'):
    
    # @var cache_file String
    cache_file = dataset.get_working_dir (dataset.task, key + '.npy')
    
    
    # Restore cache
    if (not force and os.path.isfile (cache_file)):
        return np.load (cache_file)
    
    
    # Adding again 1 because of reserved 0 index
    vocab_size = len (tokenizer.word_index) + 1  
    
    
    # embedding_matrix
    embedding_matrix = np.zeros ((vocab_size, embedding_dim))


    # @var filename String
    filename = config.pretrained_models[lang][key]['vectors']

    
    # Progress bar
    with tqdm (total = os.path.getsize (filename)) as pbar:
    
        # Update progress bar
        pbar.set_description ("Processing %s" % key)
        
    
        # Open vector file
        with open (filename, 'r', encoding = 'utf8', errors = 'ignore') as f:
        
            for line in f:
                
                # Get word and vector
                word, *vector = line.split ()
                
                
                # Check if the word is in the word index
                if word in tokenizer.word_index:
                
                    # Get position
                    idx = tokenizer.word_index[word] 
                    
                    
                    # Set weights
                    embedding_matrix[idx] = np.array (vector, dtype = np.float32)[:embedding_dim]
                
                
                # Update bar
                pbar.update (len (line.encode('utf-8')))

    # Store in cache
    os.makedirs (os.path.dirname (cache_file), exist_ok = True)
    np.save (cache_file, embedding_matrix)


    # Return embedding matrix
    return embedding_matrix

    
    
def print_cm (cm, labels, hide_zeroes = False, hide_diagonal = False, hide_threshold = None):
    """
    pretty print for confusion matrixes
    @link https://gist.github.com/zachguo/10296432
    """
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    
    
    # @link https://stackoverflow.com/questions/21341096/redirect-print-to-string-list
    s = StringIO ()
    sys.stdout = s

    
    # Begin CHANGES
    fst_empty_cell = (columnwidth-3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "
    
    if len(fst_empty_cell) < len (empty_cell):
        fst_empty_cell = " " * (len (empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print ("    " + fst_empty_cell, end=" ")
    # End CHANGES
    
    for label in labels:
        print ("%{0}s".format (columnwidth) % label, end=" ")
        
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print ("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range (len (labels)):
            cell = "%{0}.2f".format (columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float (cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print (cell, end=" ")
        print ()
    
    sys.stdout = sys.__stdout__
    
    return s.getvalue ()
        
def root_mean_squared_error (y_true, y_pred):
    """
    Returns the RMSE
    
    @link https://stackoverflow.com/questions/43855162/rmse-rmsle-loss-function-in-keras
    @param y_true
    @param y_pred
    """
    return K.sqrt (K.mean (K.square (y_pred - y_true)))
    
    
def get_f1 (precision, recall, weight = 1):
    """
    @var precision
    @var recall
    @var weight
    """
    
    if not (precision and recall):
        return 0
    
    precision_by_recall = precision * recall
    precision_plus_recall = precision + recall
    
    return weight * ((2 * precision_by_recall) / (precision_plus_recall)) if precision_plus_recall > 0 else 0
