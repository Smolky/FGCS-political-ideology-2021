import os
import glob
import tensorflow
import talos
import datetime
import sys
import csv
import itertools
import pickle
import time
import sklearn
import numpy as np
import pandas as pd
import bootstrap

from . import utils
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from keras import backend as K
from sklearn.pipeline import FeatureUnion
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterSampler
from sklearn import preprocessing

from .BaseModel import BaseModel
from . import kerasModel

class DeepLearningTechniques (BaseModel):

    # @var all_available_features List
    all_available_features = ['lf', 'se', 'be', 'we']
    

    """
    DeepLearningTechniques
    
    Keras and Talos por hyper-parameter tunning that include 
    recurrent neural networks, convolutional neural networsks, 
    and vanilla multilayer perceptrons
    """

    def train (self, force = False, using_official_test = True):
        """
        @inherit
        """
        
        # @var dataset_options Dict
        dataset_options = self.dataset.get_task_options ()
        
        
        # @var df DataFrame
        df = self.dataset.get ()
        
        
        # @var val_split String 
        # Determine which split do we get based on if we 
        # want to validate the val dataset of the train dataset
        val_split = 'val' if not using_official_test else 'test'
        
        
        # @var train_df DataFrame Get training split
        train_df = self.dataset.get_split (df, 'train')
        
        
        # If using the validation set for training
        if using_official_test:
            train_df = pd.concat ([train_df, self.dataset.get_split (df, 'val')])
            
        
        # @var val_df DataFrame Get validation split
        val_df = self.dataset.get_split (df, val_split)
        
        
        # @var indexes Dict the the indexes for each split
        indexes = {split: subset.index for split, subset in {'train': train_df, 'val': val_df}.items ()}
        
        
        # @var task_type string Determine if we are dealing with a regression or classification problem
        task_type = self.dataset.get_task_type ()
        
        
        # @var is_imabalanced Boolean Determine if the dataset is imbalaced
        is_imabalanced = self.dataset.is_imabalanced ()
        
        
        # @var unique_classes Series
        unique_classes = sorted (train_df['label'].unique ())
        
        
        # @var train_label_counts int
        train_label_counts = train_df['label'].value_counts (sort = False)

        
        # If the problem is classification, then we need to encode the label as numbers instead of using names
        if task_type == 'classification' and len (unique_classes) > 2:
            
            # Create a binarizer
            lb = sklearn.preprocessing.LabelBinarizer ()
            
            
            # Get unique classes
            lb.fit (unique_classes)
            
            
            # Note that we are dealing with binary (one label) or multi-class (one-hot enconding)
            train_df = pd.concat ([train_df, pd.DataFrame (lb.transform (train_df['label']), index = train_df.index, columns = lb.classes_)], axis = 1)
            val_df = pd.concat ([val_df, pd.DataFrame (lb.transform (val_df['label']), index = val_df.index, columns = lb.classes_)], axis = 1)
        
        
        # Encode labels as True|False
        elif task_type == 'classification':
            train_df['label'] = train_df['label'].astype ('category').cat.codes
            val_df['label'] = val_df['label'].astype ('category').cat.codes
        
        
        # @var number_of_classes int
        number_of_classes = self.dataset.get_num_labels ()
        
        
        # @var Tokenizer Tokenizer|None Only defined if we handle word embedding features
        tokenizer = None
        
        
        # @var maxlen int|None Only defined if we handle word embedding features
        maxlen = None
        
        
        # Generate the tokenizer employed 
        if 'we' in self.features:
        
            # Load the tokenizer from disk
            self.features['we'].load_tokenizer_from_disk (self.dataset.get_working_dir (self.dataset.task, 'we_tokenizer.pickle')) 
            
            
            # @var tokenizer Tokenizer
            tokenizer = self.features['we'].get_tokenizer ()
            
            
            # @var maxlen int Get maxlen
            maxlen = int (self.features['we'].maxlen)
            
            
            # Generate data and store them on cache
            for key in ['fasttext', 'glove', 'word2vec']:
                utils.get_embedding_matrix (key = key, tokenizer = tokenizer, dataset = self.dataset)


        # Get the optimizers for hyper-parameter optimisation
        optimizers = [keras.optimizers.Adam]
        if task_type == 'regression':
            optimizers.append (keras.optimizers.RMSprop)
        
        
        # @var feature_combinations List
        feature_combinations = self.get_feature_combinations ()
        
        
        # Get the reduction_metric according to the domain problem
        reduction_metric = 'val_loss' if task_type == 'classification' else 'val_rmse'
        
        
        # @var time_limit Used with TALOS to prevent infinite trainings times
        time_limit = (datetime.datetime.now () + datetime.timedelta (minutes = 60 * 24 * 7)).strftime ("%Y-%m-%d %H:%M")
        
        
        # Get parameters to evaluate
        params_epochs = [1000]
        params_lr = [10e-03, 10e-04]
        params_batch_size = [8, 16, 32, 64]
        params_dropout = [False, 0.1, 0.2, 0.3]
        params_pretrained_embeddings = ['none', 'fasttext', 'glove', 'word2vec']
        
        
        # With imbalaced datasets, we evaluate larger batch_sizes
        # Now create and train your model using the function that was defined earlier. Notice that the model is fit 
        # using a larger than default batch size of 2048, this is important to ensure that each batch has a 
        # decent chance of containing a few positive samples. If the batch size was too small, they would likely have 
        # no real data to learn
        # @link https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
        # @todo Find heuristics to determine this size
        """
        Uncomment this for some problems
        if is_imabalanced:
            params_batch_size = [128, 256, 512]
        """
        
        # Iterate over features
        for feature_combination in feature_combinations:
            
            # @var feature_key String
            feature_key = '-'.join (feature_combination)
            
            
            # @var architectures_to_evaluate List
            architectures_to_evaluate = ['bigru'] if 'we' in feature_key else ['dense']
            
            
            # @var best_model_file String
            best_model_file = self.dataset.get_working_dir (self.dataset.task, 'models', 'deep-learning', feature_key, 'model.h5')
            
            
            # If the file exists, then skip (unless we force to retrain)
            if os.path.isfile (best_model_file):
                if not force:
                    print ("skip " + best_model_file)
                    continue
            
            # Skip merged datasets
            if 'we' in feature_key and self.dataset.is_merged:
                print ("skiping we in a merged dataset")
                continue
            
            
            # @var features_to_show List
            features_to_show = [
                'features', 'architecture', 'shape', 'num_layers', 'first_neuron', 
                'dropout', 'lr', 'batch_size', 'activation'
            ]
            
            
            # Include pretrained word embeddings
            if 'we' in feature_key:
                features_to_show.append ('pretrained_embeddings')
            
            
            # @var metrics_to_show List
            metrics_to_show = []
            
            
            # @var main_metric String The metric that determines which is the best model
            main_metric = None
            
            
            # Classification metrics
            if 'classification' == task_type:
                
                # Imbalanced datasets require f1 score to determine the correct balanced between classes
                if is_imabalanced:
                    metrics_to_show = ['precision', 'val_precision', 'recall', 'val_recall', 'loss', 'val_loss', 'f1', 'val_f1']
                    main_metric = 'val_f1';
                
                # Normalised datasets work fine with accuracy metrics
                elif number_of_classes <= 2:
                    metrics_to_show = ['loss', 'val_loss', 'accuracy', 'val_accuracy']
                    main_metric = 'val_accuracy';
                
                # For multi-classification tasks, we want to show the F1 score
                if number_of_classes > 2:
                    metrics_to_show = ['loss', 'val_loss', 'f1_micro', 'val_f1_micro']
                    main_metric = 'f1_micro', 'val_f1_micro';
                    
            
            # Regression metrics work fine with loss metrics
            else:
                metrics_to_show = ['loss', 'val_loss']
                main_metric = 'val_loss'


            # Special by class
            if 'scoring' in dataset_options:
                main_metric = dataset_options['scoring']
                if dataset_options['scoring'] not in metrics_to_show:
                    metrics_to_show += [dataset_options['scoring']]

            
            # @var initial_bias Float
            initial_bias = None
            if 'classification' == task_type:
                if number_of_classes <= 2:
                    initial_bias = np.log ([train_label_counts[1] / train_label_counts[0]])
            
            
            # Class weight
            # Using class_weights changes the range of the loss and may affect the stability 
            # of the training depending on the optimizer. 
            # Optimizers whose step size is dependent on the magnitude of the gradient, 
            # like optimizers.SGD, may fail. The optimizer used here, optimizers.Adam, 
            # is unaffected by the scaling change. 
            
            # NOTE: Because of the weighting, the total losses are not comparable between the two models.
            # NOTE: Scaling keep the loss to a similar magnitude. The sum of the weights of all examples stays the same.
            # @link https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
            
                
                
            # @link https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
            class_weight = {i: (1 / train_label_counts[i]) * (len (train_df)) / len (train_label_counts) for i in range (0, len (train_label_counts))} \
                if 'classification' == task_type else None
            
            
            # Adjust the input size
            inputs_size = {
                key: pd.DataFrame (self.features[key].transform (df)).shape[1] if key in feature_key else 0 for key in ['lf', 'se', 'be', 'we']
            }
            
            
            # @var parameters_to_evaluate List
            # Note that some of these variables are not hyper-parameters
            # but only features we pass to the function that builds the 
            # Keras Models
            parameters_to_evaluate = []
            
            
            # Shallow neural networks
            if 'dense' in architectures_to_evaluate:
                parameters_to_evaluate.append ({
                    'architecture': ['dense'],
                    'num_layers':  [1, 2],
                    'shape': ['brick'],
                    'activation': ['linear', 'relu', 'sigmoid', 'tanh'],
                    'pretrained_embeddings': params_pretrained_embeddings,
                })
            
            
            # Deep neuronal networks
            if 'dense' in architectures_to_evaluate:
                parameters_to_evaluate.append ({
                    'architecture': ['dense'],
                    'num_layers': [3, 4, 5, 6, 7, 8],
                    'shape': ['funnel', 'rhombus', 'lfunnel', 'brick', 'diamond', '3angle'],
                    'activation': ['sigmoid', 'tanh', 'selu', 'elu'],
                    'pretrained_embeddings': params_pretrained_embeddings,
                })
            
            
            # Convolutional neuronal networks
            if 'cnn' in architectures_to_evaluate:
            
                # Convolutional neural networks
                parameters_to_evaluate.append ({
                    'architecture': ['cnn'],
                    'num_layers': [1, 2],
                    'shape': ['brick'],
                    'kernel_size': [3, 4, 5],
                    'pretrained_embeddings': params_pretrained_embeddings,
                    'activation': ['relu', 'tanh'],
                    'first_neuron': [16, 32, 64],
                    'batch_size': [64],
                    'lr': [10e-03]
                })
                
                
            # Recurrent neuronal networks
            if 'bigru' in architectures_to_evaluate:
                
                # Bidirectional Recurrent neuronal networks
                parameters_to_evaluate.append ({
                    'architecture': ['bigru'],
                    'num_layers': [1, 2],
                    'shape': ['brick'],
                    'pretrained_embeddings': params_pretrained_embeddings,
                    'activation': ['relu'],
                    'first_neuron': [4, 5],
                    'batch_size': [64],
                    'lr': [10e-03]
                })

            # Recurrent neuronal networks
            if 'bilstm' in architectures_to_evaluate:
                
                # Bidirectional Recurrent neuronal networks
                parameters_to_evaluate.append ({
                    'architecture': ['bilstm'],
                    'num_layers': [1, 2],
                    'shape': ['brick'],
                    'pretrained_embeddings': params_pretrained_embeddings,
                    'activation': ['relu'],
                    'first_neuron': [4, 5],
                    'batch_size': [64],
                    'lr': [10e-03]
                })
            
            
            # the optimal size of the hidden layer is usually between the size of the input and size of the output layers
            # @link https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
            # @var min_neurons int
            min_neurons = self.dataset.get_num_labels ()
            
            
            # @var max_neurons int
            max_neurons = sum (inputs_size.values ())

            
            # @var param_first_neuron List
            param_first_neuron = [neurons for neurons in [8, 16, 48, 64, 128, 256, 512, 1024] if neurons >= min_neurons and neurons <= max_neurons]
            param_first_neuron.append (int (round (min_neurons + max_neurons) / 2))
            
            
            # Common parameters for all groups (except if they were defined before)
            for group in parameters_to_evaluate:
                
                if 'optimizer' not in group:
                    group['optimizer'] = optimizers
                    
                if 'first_neuron' not in group:
                    group['first_neuron'] = param_first_neuron
                
                if 'lr' not in group:
                    group['lr'] =  params_lr

                if 'epochs' not in group:
                    group['epochs'] = params_epochs
                    
                if 'batch_size' not in group:
                    group['batch_size'] = params_batch_size
                    
                if 'dropout' not in group:
                    group['dropout'] = params_dropout
                
                
            # @var x Dict of features for each subset
            x = {}
            for subset in ['train', 'val']:
                x[subset] = {}
                for key in feature_combination:
                    features = pd.DataFrame (self.features[key].transform (df))
                    
                    """
                    # Uncomment this for testing "input-indepent baseline"
                    # This changes all the inputs features to 0
                    # The idea is that the network learn worse than with input data
                    # Do not worry if the models learns, it is becasue the "bias"
                    # You could change it as: keras.layers.Dense (use_bias = False)
                    # @link http://karpathy.github.io/2019/04/25/recipe/#2-set-up-the-end-to-end-trainingevaluation-skeleton--get-dumb-baselines
                    for col in features.columns:
                        features[col].values[:] = 0
                    """
                    
                    x[subset]['input_' + key] = features[features.index.isin (indexes[subset])].reindex (indexes[subset])
            
            
            
            # @var y_labels_columns List
            y_labels_columns = lb.classes_ if self.dataset.get_num_labels () > 2 else 'label'
            
            
            # @var y labels
            y = {
                'train': tensorflow.convert_to_tensor (train_df[y_labels_columns].values),
                'val': tensorflow.convert_to_tensor (val_df[y_labels_columns].values)
            }
            
            
            # @var n_iter int How many permutations
            n_iter = 5 if 'we' in feature_combination else 50
            
            
            # @var param_list List
            param_list = []

            
            # Sample uniformly
            for parameters in parameters_to_evaluate:
                param_list += list (ParameterSampler (parameters, n_iter = n_iter))
            
            
            # Sort parameters from shallow networks to deep neuronal networks
            param_list = sorted (param_list, key = lambda combination: (
                combination['num_layers'], 
                combination['first_neuron']
            ))
            
            
            # @var results List
            results = []
            
            
            # @var best_model Model
            best_model = None
            
            
            # @var best_models_per_architecture Dict
            best_models_per_architecture = {architecture: None for architecture in architectures_to_evaluate}
            
            
            # @var metric_of_the_best_model float
            metric_of_the_best_model = float ('-inf')
            
            
            # @var metric_of_the_best_model_per_architecture float
            metric_of_the_best_model_per_architecture = {architecture: float ('-inf') for architecture in architectures_to_evaluate}
            
            
            # Print header
            str_metrics  = ' '.join ([f'{metric: >{max (6, len (metric))}}' for metric in metrics_to_show])
            str_features  = ' '.join ([f'{feature: >{max (6, len (feature))}}' for feature in features_to_show])
            
            
            print ("dataset {dataset} corpus {corpus} task {task}".format (
                dataset = self.dataset.dataset, 
                corpus = self.dataset.corpus,
                task = self.dataset.task
            ))
            print ("Generating {n} iterations".format (n = len (param_list)))
            print ()
            print ("{:>3}".format ("#") + ": " + str_metrics + " " + str_features)

            
            # Iterate over each model
            for counter, params in enumerate (param_list):
            
                # Attach extra stuff
                params['dataset'] =  self.dataset
                params['number_of_classes'] = self.dataset.get_num_labels ()
                params['features'] = '-'.join (feature_combination)
                params['output_bias'] = initial_bias
                params['inputs_size'] = inputs_size
                params['unique_classes'] = unique_classes
                
                
                # Attach word embeddings stuff
                if 'we' in feature_combination:
                    params['tokenizer'] = tokenizer
                    params['maxlen'] = maxlen
                    params['trainable_embedding_layer'] = True
                
                
                # @var model KerasClassifier based on the parameters
                model = kerasModel.create (**params)
                
                
                # Comment
                print (model.summary ())

                
                
                # @var patience int For the early stoppping mechanism
                patience = 100
                
                if params['architecture'] in ['lstm', 'gru', 'bilstm', 'bigru']:
                    patience = 10
                    
                if params['architecture'] in ['cnn']:
                    patience = 25
                    
                
                # @var pbar a TQDM progress bar to log the process
                pbar = tqdm (range (params['epochs']))
                
                
                # @var early_stopping Early Stopping callback
                # @todo Note this: https://github.com/tensorflow/tensorflow/issues/35634
                early_stopping = tensorflow.keras.callbacks.EarlyStopping (
                    monitor = 'val_loss' if task_type == 'classification' else 'val_rmse', 
                    patience = patience,
                    restore_best_weights = True
                )


                # @var train_weights dict
                train_weights = {}
                
                
                # @var val_weights dict
                val_weights = {}
                
                
                # @var train_weights dict
                if 'classification' == task_type and len (unique_classes) > 2:
                    train_weights = train_df['label'].value_counts (normalize = True).to_dict ()
                    val_weights = val_df['label'].value_counts (normalize = True).to_dict ()
                
    
                # @var decay long
                decay = params['lr'] / params['epochs']
                def lr_time_based_decay (epoch, lr):
                    return lr * 1 / (1 + decay * epoch)
                    
                
                # @var lr_scheduler LearningRateScheduler
                # @link https://towardsdatascience.com/learning-rate-schedule-in-practice-an-example-with-keras-and-tensorflow-2-0-2f48b2888a0c
                lr_scheduler = tensorflow.keras.callbacks.LearningRateScheduler (lr_time_based_decay)
                
                
                # @var AUTOTUNE
                AUTOTUNE = tensorflow.data.AUTOTUNE
                
                
                # @var train_dataset Dict
                datasets = {split: tensorflow.data.Dataset.from_tensor_slices ((x[split], y[split])) \
                    .batch (params['batch_size']) \
                    .cache () \
                    .shuffle (params['batch_size'], reshuffle_each_iteration = True) \
                    .prefetch (buffer_size = AUTOTUNE) for split in ['train', 'val']}
                
                
                # @var tqdm_callback TQDMCallback
                tqdm_callback = TQDMCallback (
                    pbar = pbar, 
                    counter = counter, 
                    monitor = main_metric, 
                    params = params, 
                    metrics_to_show = metrics_to_show,
                    features_to_show = features_to_show,
                    train_weights = train_weights,
                    val_weights = val_weights,
                    train_dataset = datasets['train'],
                    val_dataset = datasets['val']
                )
                
                
                # @var log_callback LogResultsCallback
                log_callback = LogResultsCallback (
                    parameters = params,
                    metrics_to_show = metrics_to_show, 
                    features_to_show = features_to_show,
                    results = results,
                    train_weights = train_weights,
                    val_weights = val_weights,
                    train_dataset = datasets['train'],
                    val_dataset = datasets['val']
                )
                
                
                # Reset graph
                tensorflow.compat.v1.reset_default_graph ()
                tensorflow.keras.backend.clear_session ()
                
                
                # @var history
                try:
                    history = model.fit (
                        x = datasets['train'], 
                        validation_data = datasets['val'],
                        epochs = params['epochs'],
                        callbacks = [early_stopping, tqdm_callback, log_callback, lr_scheduler],
                        verbose = 0,
                        class_weight = class_weight
                    )
                
                # Go for the next model
                except:
                    print ("@exception")
                    continue;
                
                
                # Get best model
                if results[-1][main_metric] > metric_of_the_best_model:
                    print ("updating best-1")
                    metric_of_the_best_model = results[-1][main_metric]
                    best_model = model
                    
                if results[-1][main_metric] > metric_of_the_best_model_per_architecture[params['architecture']]:
                    print ("updating best-2")
                    metric_of_the_best_model_per_architecture[params['architecture']] = results[-1][main_metric]
                    best_models_per_architecture[params['architecture']] = model
                    

                
            
            # @var scan_object_data DataFrame
            scan_object_data = pd.DataFrame (results)
            
            
            # Perform one hot encoding for categorical data for better understanding in a dataframe or 
            # csv file
            for feature in ['shape', 'architecture', 'activation', 'pretrained_embeddings']:
                if feature in scan_object_data.columns:
                    scan_object_data = utils.pd_onehot (scan_object_data, feature)
            
            
            # @var params_filename String To store the hyperparameter evaluation
            params_filename = self.dataset.get_working_dir (self.dataset.task, 'models', 'deep-learning', feature_key, 'hyperparameters.csv')
            
            
            # Store hyper-parameter tunning for further analysis
            scan_object_data.to_csv (params_filename, index = False)
            
            
            # Save the best model
            best_model.save (best_model_file)
            
            
            # Save the best model per architecture
            for architecture, best_model_per_architecture in best_models_per_architecture.items ():
            
                if not best_model_per_architecture:
                    print ("skip model for " + architecture)
                    continue;

                # @var _model_file String
                _model_file = self.dataset.get_working_dir (self.dataset.task, 'models', 'deep-learning', feature_key, 'model-' + architecture + '.h5')


                # Save the best model
                best_model_per_architecture.save (_model_file)


    def predict (self, using_official_test = False, callback = None):
        """
        @inherit
        """
        
        # @link https://stackoverflow.com/questions/58814130/tensorflow-2-0-custom-keras-metric-caused-tf-function-retracing-warning/62298471#62298471
        tensorflow.compat.v1.disable_eager_execution ()
        
        
        # @var feature_combinations List
        feature_combinations = self.get_feature_combinations ()
        
        
        # @var df DataFrame
        df = self.dataset.get ()
        
        
        # @var result Dict
        result = {}
            
        
        # Iterate over feature combinations
        for feature_combination in feature_combinations:
        
            # @var feature_key String
            feature_key = '-'.join (feature_combination)
            
            
            # @var model_filename String Retrieve the filepath of the best model
            model_filename = self.dataset.get_working_dir (self.dataset.task, 'models', 'deep-learning', feature_key, 'model.h5')
            
            
            # Ensure the model exists. If not, fill the row with NaNs
            if not os.path.exists (model_filename):
                result[feature_key] = [np.nan] * len (df)
                continue
        
        
            # @var best_model KerasModel
            try:
                best_model = keras.models.load_model (model_filename, compile = False)
            except:
                print ("model could not be loaded. Skip...")
                continue
            
            
            # @var x Dict of features for each subset
            x = {key: pd.DataFrame (self.features[key].transform (df)) for key in feature_combination}
            
            
            # If the supplied dataset contains information of the split, that means that we are dealing
            # only with a subset of the features and we retrieve it accordingly
            if using_official_test:
                x = {'input_' + key: item[item.index.isin (df.index)].reindex (df.index) for key, item in x.items ()}


            # @var true_labels_iterable List
            true_labels_iterable = self.dataset.get_available_labels ()

            
            # @var predictions List
            y_pred = self.get_y_pred (best_model, x, true_labels_iterable)
            
            
            # @var y_pred
            y_real = df['label']

            
            # Results for the final data frame
            result[feature_key] = y_pred
            
            
            # Store classification report
            if using_official_test:
                
                # Classification report
                pd.DataFrame (classification_report (y_true = y_real, y_pred = y_pred, output_dict = True)) \
                    .transpose () \
                    .to_latex (self.dataset.get_working_dir (self.dataset.task, 'results', 'deep-learning', feature_key, 'classification_report.latex'), index = True)
                
        
                # Confusion matrix
                pd.DataFrame (confusion_matrix (y_pred = y_pred, y_true = y_real, labels = self.dataset.get_available_labels ())) \
                    .to_latex (self.dataset.get_working_dir (self.dataset.task, 'results', 'deep-learning', feature_key, 'confusion_matrix.latex'), index = True)
                
                
                # Plot the real confusion matrix
                pd.DataFrame (confusion_matrix (y_pred = y_pred, y_true = y_pred, labels = self.dataset.get_available_labels ())) \
                    .to_latex (self.dataset.get_working_dir (self.dataset.task, 'results', 'deep-learning', feature_key, 'confusion_matrix_truth.latex'), index = True)
                
                
                # @var models_per_architecture
                models_per_architecture = {}
                
                
                # ...
                if 'we' in feature_key:
                
                    # @var files List */
                    files = self.dataset.get_working_dir (self.dataset.task, 'models', 'deep-learning', feature_key) + '/*.h5'
                
                
                    # @var model_files List */
                    model_files = glob.glob (files)
                    
                    
                    # Iterate
                    for models_filepath in model_files:
                        
                        # @var model_basename String
                        model_basename = os.path.basename (models_filepath)
                        
                        
                        # Skip best model
                        if 'model.h5' == model_basename:
                            continue
                        
                        
                        # @var _temp KerasModel
                        _temp = keras.models.load_model (models_filepath, compile = False)
                        
                        print (models_filepath)
                        print ("-----------------------------")
                        print (_temp.summary ())
                        
                        for layer in _temp.layers:
                            print (layer.get_config ())
                        
                        
                        # @var best_model_per_architecture
                        models_per_architecture[model_basename] = self.get_y_pred (_temp, x, true_labels_iterable)
                
                
                # run callback
                if callback:
                    callback (feature_key, y_pred, y_real, models_per_architecture)
                
            
        # Classification report
        result = pd.DataFrame (result)
        
        
        # Attach text and labels
        result['label'] = df.reset_index (drop=True)['label']
        result['text'] = df.reset_index (drop=True)['tweet_clean']
        
        
        # Store the hyper-parameters in disk
        if using_official_test:
            result.to_csv (self.dataset.get_working_dir (self.dataset.task, 'results', 'deep-learning', 'features-comparison.csv'), index = False, quoting = csv.QUOTE_ALL)
        
        

    def get_y_pred (self, model, x, true_labels):
        """
        Calculates the predicted labels
        
        @param model KerasModel
        @param x Dict The features
        @param true_labels List of the true labels
        """
            
        # @var predictions Get the logits of the predictions
        predictions = model.predict (x)
        
        
        # According to the number of labels we discern between binary and multiclass
        if self.dataset.get_num_labels () <= 2:
            predictions = predictions >= 0.5
            predictions = np.squeeze (predictions)
        
        # Multiclass
        else:
            predictions = np.argmax (predictions, axis = 1)
        
        
        # @var y_pred
        return [true_labels[int (prediction)] for prediction in predictions]
        

class LogResultsCallback (tensorflow.keras.callbacks.Callback):
    """
    LogResultsCallback
    
    This callback is used to store the best parameters combinations after one
    training
    
    @param params Dict
    @param results List
    """
    def __init__ (self, parameters = {}, train_dataset = None, val_dataset = None, metrics_to_show = [], features_to_show = [], results = [], train_weights = {}, val_weights = {}):
        self.parameters = parameters
        self.metrics_to_show = metrics_to_show
        self.features_to_show = features_to_show
        self.results = results
        self.train_weights = train_weights
        self.val_weights = val_weights,
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        

    def get_metrics (self, logs = None):

        # @var metrics Dictionary
        metrics = {metric: logs.get (metric) for metric in self.metrics_to_show}
        
        
        # Calculate F1 Score
        if 'f1' in self.metrics_to_show:
            metrics['f1'] = utils.get_f1 (metrics['precision'], metrics['recall'])
        
        # Calculate F1 Score on VAL
        if 'val_f1' in self.metrics_to_show:
            metrics['val_f1'] = utils.get_f1 (metrics['val_precision'], metrics['val_recall'])
        
        if 'f1_micro' in self.metrics_to_show:
            # @var y_true_batch List
            y_true_batches = list (map (lambda x: x[1], self.train_dataset))
            y_true_batches = [np.where (y_true_batch.numpy () > 0.5, 1, 0) if (y_true_batch.ndim == 1 or y_true_batch.shape[1] == 1) else np.argmax (y_true_batch, axis = 1) \
                for y_true_batch in y_true_batches]
            
            # @var y_true List
            y_true = list(itertools.chain(*y_true_batches))

            
            # @var predictions List
            predictions = self.model.predict (self.train_dataset)
            
            
            # @var y_real List
            y_pred = np.where (predictions > 0.5, 1, 0) if (predictions.ndim == 1 or predictions.shape[1] == 1) else np.argmax (predictions, axis = 1)
            
            
            # Attach metric
            metrics['f1_micro'] = f1_score (y_true = y_true, y_pred = y_pred, average = 'micro')    
        
        if 'val_f1_micro' in self.metrics_to_show:
            # @var y_true_batch List
            y_true_batches = list (map (lambda x: x[1], self.val_dataset))
            y_true_batches = [np.where (y_true_batch.numpy () > 0.5, 1, 0) if (y_true_batch.ndim == 1 or y_true_batch.shape[1] == 1) else np.argmax (y_true_batch, axis = 1) \
                for y_true_batch in y_true_batches]
            
            # @var y_true List
            y_true = list(itertools.chain(*y_true_batches))

            
            # @var predictions List
            predictions = self.model.predict (self.val_dataset)
            
            
            # @var y_real List
            y_pred = np.where (predictions > 0.5, 1, 0) if (predictions.ndim == 1 or predictions.shape[1] == 1) else np.argmax (predictions, axis = 1)
            
            
            # Attach metric
            metrics['val_f1_micro'] = f1_score (y_true = y_true, y_pred = y_pred, average = 'micro')        
        
        # round
        metrics = {label: round (metric, 4) if metric else 0 for label, metric in metrics.items ()}
        
        return metrics

        
    def on_train_end (self, logs = None):
        metrics = self.get_metrics (logs)
        features = {feature: self.parameters[feature] for feature in self.features_to_show}
        self.results.append ({**metrics, **features})
        
    
        
class TQDMCallback (tensorflow.keras.callbacks.Callback):
    """
    TQDMCallback
    """

    def __init__ (self, 
        pbar, 
        counter = 0, 
        train_dataset = None,
        val_dataset = None,
        monitor = 'val_accuracy', 
        params = {}, 
        metrics_to_show = [], 
        features_to_show = [],
        train_weights = {},
        val_weights = {}
    ):
        
        # Parent
        super (TQDMCallback, self).__init__()
        
        
        # Store the variable to monitor
        self.monitor = monitor
        self.pbar = pbar;
        self.train_dataset = train_dataset;
        self.val_dataset = val_dataset;
        self.counter = counter;
        self.metrics = metrics_to_show
        self.train_weights = train_weights
        self.val_weights = val_weights
        
        
        # Get all the features String
        self.str_features = ' '.join ([f'{str(params[feature])[:6]: >{max (6, len (feature))}}' for feature in features_to_show])
        

        # @var str_metrics String
        str_metrics = ' '.join ([f'{float ("-inf"): >{max (6, len (metric))}}' for metric in self.metrics])
        
        
        # Start
        self.pbar.set_description ("{:>3}".format (str (self.counter + 1)) + ": " + str_metrics + " " + self.str_features)
        
        
        # Update value
        self.pbar.update (0)

    
    def on_train_end (self, logs = None):
        self.pbar.set_description ("{:>3}".format (str (self.counter + 1)) + ": " + self.metrics_to_show (logs or {}) + " " + self.str_features)
        self.pbar.close ()
        

    def on_epoch_end (self, epoch, logs = {}):
        self.pbar.set_description ("{:>3}".format (str (self.counter + 1)) + ": " + self.metrics_to_show (logs or {}) + " " + self.str_features)
        self.pbar.update (1)

    def metrics_to_show (self, logs = None):

        # @var metrics List
        metrics = {metric: logs.get (metric) for metric in self.metrics}


        # Calculate F1 Score
        if 'f1' in self.metrics:
            metrics['f1'] = utils.get_f1 (metrics['precision'], metrics['recall'])
        
        
        # Calculate F1 Score on VAL
        if 'val_f1' in self.metrics:
            metrics['val_f1'] = utils.get_f1 (metrics['val_precision'], metrics['val_recall'])
        
        
        if 'f1_micro' in self.metrics:
            
            # @var y_true_batch List
            y_true_batches = list (map (lambda x: x[1], self.train_dataset))
            y_true_batches = [np.where (y_true_batch.numpy () > 0.5, 1, 0) if (y_true_batch.ndim == 1 or y_true_batch.shape[1] == 1) else np.argmax (y_true_batch, axis = 1) \
                for y_true_batch in y_true_batches]
            
            # @var y_true List
            y_true = list (itertools.chain (*y_true_batches))

            
            # @var predictions List
            predictions = self.model.predict (self.train_dataset)
            
            
            # @var y_real List
            y_pred = np.where (predictions > 0.5, 1, 0) if (predictions.ndim == 1 or predictions.shape[1] == 1) else np.argmax (predictions, axis = 1)
            
            
            # Attach metric
            metrics['f1_micro'] = f1_score (y_true = y_true, y_pred = y_pred, average = 'micro')
        
        if 'val_f1_micro' in self.metrics:
            
            # @var y_true_batch List
            y_true_batches = list (map (lambda x: x[1], self.val_dataset))
            y_true_batches = [np.where (y_true_batch.numpy () > 0.5, 1, 0) if (y_true_batch.ndim == 1 or y_true_batch.shape[1] == 1) else np.argmax (y_true_batch, axis = 1) \
                for y_true_batch in y_true_batches]
            
            # @var y_true List
            y_true = list (itertools.chain (*y_true_batches))

            
            # @var predictions List
            predictions = self.model.predict (self.val_dataset)
            
            
            # @var y_real List
            y_pred = np.where (predictions > 0.5, 1, 0) if (predictions.ndim == 1 or predictions.shape[1] == 1) else np.argmax (predictions, axis = 1)
            
            
            # Attach metric
            metrics['val_f1_micro'] = f1_score (y_true = y_true, y_pred = y_pred, average = 'micro')
    
        
        # round
        metrics = {label: round (metric, 4) if metric else 0 for label, metric in metrics.items ()}
        
    
        # @var metrics_str String Get all the metrics in a fancy string mode
        metrics_str  = ' '.join ([f'{value: >{max (6, len (metric))},.4f}' for metric, value in metrics.items ()])

    
        return metrics_str