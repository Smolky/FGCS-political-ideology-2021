import os
import glob
import tensorflow
import datetime
import sys
import csv
import json
import itertools
import pickle
import time
import sklearn
import numpy as np
import pandas as pd
import bootstrap
import traceback
import shutil

from . import utils
from contextlib import redirect_stdout
from tqdm import tqdm

from sklearn import preprocessing
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from .BaseModel import BaseModel
from . import kerasModel

class DeepLearningTechniques (BaseModel):
    """
    DeepLearningTechniques
    
    Keras por hyper-parameter tunning that include 
    recurrent neural networks, convolutional neural networsks, 
    and vanilla multilayer perceptrons
    """
    
    # @var all_available_features List
    all_available_features = ['lf', 'se', 'be', 'we', 'ne', 'cf', 'bf', 'pr', 'ng']
    
    
    # @var log Boolean
    log = False
    
    
    def get_main_metric (self):
        """ 
        The metric that determines which is the best model 
        """
        
        # @var task_type String
        task_type = self.dataset.get_task_type ()
        
        
        # @var dataset_options Dict
        dataset_options = self.dataset.get_task_options ()
        
        
        # Special by class
        if 'scoring' in dataset_options:
            return dataset_options['scoring']
            
            
        # Regression metrics
        if 'regression' == task_type:
            return 'val_rmse'

        
        # Multi_label metrics
        if 'multi_label' == task_type:
            return 'val_f1_score'
        
        
        # Classification metrics
        # Imbalanced datasets require f1 score to determine the correct balanced among classes
        # However, we include the accuracy as well in order to see if the model is learning to 
        # classify instances
        if 'classification' == task_type:
            return 'val_f1_score' if self.dataset.is_imabalanced () else 'val_accuracy'


        # Most generic option
        return 'val_loss'
    
    
    def get_metrics_to_show (self):
        """
        @return List
        """
    
        # @var task_type String
        task_type = self.dataset.get_task_type ()    
        
        
        # @var metrics_to_show List Default values
        metrics_to_show = ['best', 'epochs']
        
        
        # Classification metrics
        if 'classification' == task_type:
            metrics_to_show = ['best', 'epochs', 'loss', 'val_loss', 'accuracy', 'val_accuracy']

            
            # Imbalanced datasets require f1 score to determine the correct balanced among classes
            # However, we include the accuracy as well in order to see if the model is learning to 
            # classify instances
            if self.dataset.is_imabalanced ():
                metrics_to_show += ['f1_score', 'val_f1_score']
        
        
        # Regression metrics
        if 'regression' == task_type:
            metrics_to_show += ['rmse', 'val_rmse']
            main_metric = 'val_rmse'

        
        # Multi_label metrics
        if 'multi_label' == task_type:
            metrics_to_show += ['accuracy', 'val_accuracy', 'f1_score', 'val_f1_score']
        
        
        # @var dataset_options Dict
        dataset_options = self.dataset.get_task_options ()
        

        # Special by class
        if 'scoring' in dataset_options:
            if dataset_options['scoring'] not in metrics_to_show:
                metrics_to_show += [dataset_options['scoring']]


        return metrics_to_show
    
    
    def has_external_features (self):
        """
        @inherit
        """        
        return True
        

    def train (self, force = False, using_official_test = True):
        """
        @inherit
        """
        
        # @var dataset_options Dict
        dataset_options = self.dataset.get_task_options ()
        
        
        # @var df DataFrame
        df = self.dataset.get ()
        
        
        # @var task_type string Determine if we are dealing with a regression or classification problem
        task_type = self.dataset.get_task_type ()
        
        
        # Remove NaN for regression tasks
        if task_type == 'regression':
            # df['label'] = df['label'].fillna (0)
            df = df.dropna (subset = ['label'])
        
        
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
        
        
        # @var language String
        language = self.dataset.get_dataset_language ()

        
        # @var indexes Dict the the indexes for each split
        indexes = {split: subset.index for split, subset in {'train': train_df, 'val': val_df}.items ()}
        
        
        # @var available_labels
        available_labels = self.dataset.get_available_labels ()
        
        
        # @var is_imabalanced Boolean Determine if the dataset is imbalaced
        is_imabalanced = self.dataset.is_imabalanced ()
        
        
        # If the problem is classification, then we need to encode the label as numbers instead of using names
        if task_type in 'classification':
            
            # One hot encoding
            if self.dataset.get_num_labels () > 2:
            
                # Create a binarizer
                lb = sklearn.preprocessing.LabelBinarizer ()
                
                
                # Fit the label binarizer
                lb.fit (available_labels)
                
                
                # Note that we are dealing with one-hot enconding for multi-class
                train_df = pd.concat ([train_df, pd.DataFrame (lb.transform (train_df['label']), index = train_df.index, columns = lb.classes_)], axis = 1)
                val_df = pd.concat ([val_df, pd.DataFrame (lb.transform (val_df['label']), index = val_df.index, columns = lb.classes_)], axis = 1)

            
            # Encode labels as True|False for binary labels
            else:
                train_df['label'] = train_df['label'].astype ('category').cat.codes
                val_df['label'] = val_df['label'].astype ('category').cat.codes


        elif task_type in 'multi_label':

            # Create a binarizer
            lb = sklearn.preprocessing.MultiLabelBinarizer ()
            
            
            # Fit the multi-label binarizer
            lb.fit ([available_labels])
            
            
            # Encode the labels and merge to the dataframe
            # Note we use "; " to automatically trim the texts
            train_df = pd.concat ([train_df, pd.DataFrame (lb.transform (train_df['label'].str.split ('; ')), index = train_df.index, columns = lb.classes_)], axis = 1)
            val_df = pd.concat ([val_df, pd.DataFrame (lb.transform (val_df['label'].str.split ('; ')), index = val_df.index, columns = lb.classes_)], axis = 1)

            
        # @var Tokenizer Tokenizer|None Only defined if we handle word embedding features
        tokenizer = None
        
        
        # @var maxlen int|None Only defined if we handle word embedding features
        maxlen = None
        
        
        # @var pretrained_word_embeddings List
        pretrained_word_embeddings = []

        
        # Generate the tokenizer employed 
        if 'we' in self.features:
        
            # Load the tokenizer from disk
            self.features['we'].load_tokenizer_from_disk (self.dataset.get_working_dir (self.dataset.task, 'we_tokenizer.pickle')) 
            
            
            # @var tokenizer Tokenizer
            tokenizer = self.features['we'].get_tokenizer ()
            
            
            # @var maxlen int Get maxlen
            maxlen = int (self.features['we'].maxlen)
            
            
            # @var pretrained_word_embeddings List Generate data and store them on cache
            pretrained_word_embeddings = ['fasttext', 'glove', 'word2vec'] if language == 'es' else ['fasttext']
            
            
            # Get embedding matrix
            for key in pretrained_word_embeddings:
                utils.get_embedding_matrix (
                    key = key, 
                    tokenizer = tokenizer, 
                    dataset = self.dataset,
                    lang = language
                )



        # Get the optimizers for hyper-parameter optimisation
        optimizers = [tensorflow.keras.optimizers.Adam]
        if task_type == 'regression':
            optimizers.append (tensorflow.keras.optimizers.RMSprop)
        
        
        # @var feature_combination Tuple
        feature_combination = self.get_feature_combinations ()
        
        
        # Get the reduction_metric according to the domain problem
        reduction_metric = 'val_loss'
        
        
        # Get parameters to evaluate
        params_epochs = [1000]
        params_lr = [10e-03, 10e-04]
        params_batch_size = [32, 64]
        params_dropout = [False, 0.1, 0.2, 0.3]
        params_pretrained_embeddings = ['none'] + pretrained_word_embeddings
        
        
        
        # With imbalaced datasets, we evaluate larger batch_sizes
        # Now create and train your model using the function that was defined earlier. Notice that the model is fit 
        # using a larger than default batch size of 2048, this is important to ensure that each batch has a 
        # decent chance of containing a few positive samples. If the batch size was too small, they would likely have 
        # no real data to learn
        # @link https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
        # @todo Find heuristics to determine this size
        if is_imabalanced:
            params_batch_size = [128, 256, 512]
            

        # Due to the size level, we reduce the batch size
        if 'user_level' in self.dataset.task:
            params_batch_size = [4, 8, 16]
            params_lr = [10e-03]

        if 'tweet_level' in self.dataset.task:
            params_batch_size = [128, 256, 512]
        
            
            
        # @var feature_key String
        feature_key = '-'.join (feature_combination)
        
        
        # @var architectures_to_evaluate List
        architectures_to_evaluate = ['dense', 'cnn', 'bilstm', 'bigru'] if 'we' in feature_key else ['dense']

        
        # @var best_model_file String
        best_model_file = self.dataset.get_working_dir (self.dataset.task, 'models', 'deep-learning', feature_key, 'model.h5')
        
        
        # If the file exists, then skip (unless we force to retrain)
        if os.path.isfile (best_model_file):
            if not force:
                print ("skip " + best_model_file)
                return
            
        # Skip merged datasets
        if 'we' in feature_key and self.dataset.is_merged:
            print ("skiping we in a merged dataset")
            return
            
            
        # @var logs_path String
        logs_path = self.dataset.get_working_dir (self.dataset.task, 'models', 'deep-learning', feature_key, 'logs')
        
        
        # Delete previous path folder
        shutil.rmtree (logs_path, ignore_errors = True)
        
        
        # @var features_to_show List
        features_to_show = [
            'features', 'architecture', 'shape', 'num_layers', 'first_neuron', 
            'dropout', 'lr', 'batch_size', 'activation'
        ]
        
        
        # Add optimizer
        if 'regression' == task_type:
            features_to_show.append ('optimizer')
        
        
        # Include pretrained word embeddings
        if 'we' in feature_key:
            features_to_show.append ('pretrained_embeddings')
        
        
        
        # @var main_metric String The metric that determines which is the best model
        main_metric = self.get_main_metric ()
        
        
        # @var metrics_to_show List
        metrics_to_show = self.get_metrics_to_show ()

        
        # @var initial_bias Float
        initial_bias = None
        if 'classification' == task_type and self.dataset.get_num_labels () <= 2:
        
            # @var train_label_counts int
            train_label_counts = train_df['label'].value_counts (sort = True).to_list ()
        
            
            # Set initial bias
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
        if 'regression' == task_type:
            class_weights = None
        
        if 'classification' == task_type:
            class_weights = dict (enumerate (class_weight.compute_class_weight (
                class_weight = 'balanced', 
                classes = available_labels, 
                y = self.dataset.get_split (df, 'train')['label']
            )))

        if 'multi_label' == task_type:
            class_weights = dict (enumerate (class_weight.compute_sample_weight (
                class_weight = 'balanced', 
                y = lb.transform (train_df['label'].str.split ('; ')) 
            )))
        
        
        # Adjust the input size
        inputs_size = {
            key: pd.DataFrame (self.features[key].transform (df)).shape[1] if key in feature_key else 0 for key in self.all_available_features
        }
        
        
        # @var parameters_to_evaluate List
        # Note that some of these variables are not hyper-parameters
        # but only features we pass to the function that builds the 
        # Keras Models
        parameters_to_evaluate = []
        
        
        # Shallow neural networks
        if 'dense' in architectures_to_evaluate:
            parameters_to_evaluate.append ({
                'type': ['shallow'],
                'n_iter': [100],
                'architecture': ['dense'],
                'num_layers':  [1, 2],
                'shape': ['brick'],
                'activation': ['linear', 'relu', 'sigmoid', 'tanh'],
                'pretrained_embeddings': params_pretrained_embeddings,
            })
        
        
        # Deep neuronal networks
        if 'dense' in architectures_to_evaluate:
            parameters_to_evaluate.append ({
                'type': ['deep'],
                'n_iter': [10],
                'architecture': ['dense'],
                'num_layers': [3, 4, 5, 6, 7, 8],
                'shape': ['funnel', 'rhombus', 'lfunnel', 'brick', 'diamond', '3angle'],
                'activation': ['sigmoid', 'tanh', 'selu', 'elu'],
                'pretrained_embeddings': params_pretrained_embeddings,
                'lr': [10e-03]
            })
        
        # Convolutional neuronal networks
        if 'cnn' in architectures_to_evaluate:
        
            # Convolutional neural networks
            parameters_to_evaluate.append ({
                'type': ['cnn'],
                'n_iter': [25],
                'architecture': ['cnn'],
                'num_layers': [1, 2],
                'shape': ['brick'],
                'kernel_size': [3, 4, 5],
                'pretrained_embeddings': params_pretrained_embeddings,
                'activation': ['relu', 'tanh'],
                'first_neuron': [16, 32, 64],
                'batch_size': [max (params_batch_size)],
                'lr': [10e-03]
            })
            
            
        # Recurrent neuronal networks
        if 'bigru' in architectures_to_evaluate:
            
            # Bidirectional Recurrent neuronal networks
            parameters_to_evaluate.append ({
                'type': ['rnn'],
                'n_iter': [5],
                'architecture': ['bigru'],
                'num_layers': [1, 2],
                'shape': ['brick'],
                'pretrained_embeddings': params_pretrained_embeddings,
                'activation': ['relu'],
                'first_neuron': [4, 5],
                'batch_size': [max (params_batch_size)],
                'lr': [10e-03]
            })

        # Recurrent neuronal networks
        if 'bilstm' in architectures_to_evaluate:
            
            # Bidirectional Recurrent neuronal networks
            parameters_to_evaluate.append ({
                'type': ['rnn'],
                'n_iter': [5],
                'architecture': ['bilstm'],
                'num_layers': [1, 2],
                'shape': ['brick'],
                'pretrained_embeddings': params_pretrained_embeddings,
                'activation': ['relu'],
                'first_neuron': [4, 5],
                'batch_size': [max (params_batch_size)],
                'lr': [10e-03]
            })

        
        # the optimal size of the hidden layer is usually between the size of the input and size of the output layers
        # @link https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
        # @var min_neurons int
        min_neurons = self.dataset.get_num_labels ()
        
        
        # @var max_neurons int
        max_neurons = sum (inputs_size.values ())

        
        # @var param_first_neuron List
        # @link https://www.researchgate.net/post/How-to-decide-the-number-of-hidden-layers-and-nodes-in-a-hidden-layer
        param_first_neuron = [neurons for neurons in [4, 8, 16, 48, 64, 128, 256, 512, 1024] if neurons >= min_neurons and neurons <= max_neurons]
        param_first_neuron.append (min_neurons)
        param_first_neuron.append (10 + int (pow (round (min_neurons + max_neurons), .5)))
        param_first_neuron = list (set (param_first_neuron))
        
        
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
                
                x[subset]['input_' + key] = tensorflow.cast (features[features.index.isin (indexes[subset])].reindex (indexes[subset]), tensorflow.float32)
                

        
        # @var y_labels_columns List
        y_labels_columns = lb.classes_ if self.dataset.get_num_labels () > 2 else 'label'
        
        
        # @var y labels
        y = {
            'train': tensorflow.convert_to_tensor (train_df[y_labels_columns].values),
            'val': tensorflow.convert_to_tensor (val_df[y_labels_columns].values)
        }
        
        
        # @var param_list List
        param_list = []

        
        # Sample uniformly
        for parameters in parameters_to_evaluate:
        
            # @var n_iter int
            n_iter = parameters['n_iter'][0]
            
            
            # @var parameter_to_sample Dict
            parameter_to_sample = {key: item for key, item in parameters.items () if key not in ['n_iter']}
            
            
            # Attach
            param_list += list (ParameterSampler (parameter_to_sample, n_iter = n_iter))
        
        
        # Sort parameters from shallow networks to deep neuronal networks
        param_list = sorted (param_list, key = lambda combination: (
            combination['num_layers'], 
            combination['first_neuron']
        ))
        
        
        # @var results List
        results = []
        
        
        # @var best_model Model
        best_model = None
        
        
        # @var metric_of_the_best_model float
        metric_of_the_best_model = float ('-inf') if task_type in ['classification', 'multi_label'] else float ('inf')
        
        
        # Print header
        str_metrics  = ' '.join ([f'{metric: >{max (6, len (metric))}}' for metric in metrics_to_show])
        str_features  = ' '.join ([f'{feature: >{max (6, len (feature))}}' for feature in features_to_show])

        
        # @var train_resume Dict We track information of the training process
        # including the features, best model, metrics, etc.
        train_resume = {
            'dataset': self.dataset.dataset,
            'corpus': self.dataset.corpus,
            'task': self.dataset.task,
            'task_type': task_type,
            'initial_bias': initial_bias.tolist () if initial_bias else "",
            'main_metric': main_metric,
            'iterations': len (param_list),
            'best_model_index': None,
            'features': {key: self.features[key].cache_file for key in feature_combination}
        };
        
        
        if task_type in ['classification']:
            train_resume['train_labels'] = train_df['label'].value_counts ().to_json ()
            train_resume['val_labels'] = val_df['label'].value_counts ().to_json ()
            train_resume['class_weights'] = str (json.dumps (class_weights))

        
        # Attach classification stuff to the resume
        # @todo Change train labels for multi_label problems
        if task_type in ['classification', 'multi_label']:
            # @var f1_score_scheme String
            train_resume['f1_score_scheme'] = dataset_options['f1_score_scheme'] \
                if 'f1_score_scheme' in dataset_options else 'micro'
            
        
        # Start logging
        print ("dataset {dataset} corpus {corpus} task {task}".format (
            dataset = train_resume['dataset'], 
            corpus = train_resume['corpus'],
            task = train_resume['task']
        ))
        
        
        # Print resume
        print ('resume')
        print ('------')
        print (json.dumps (train_resume, indent = 4))
        
        
        # Print Table header
        print ("{:>3}".format (str ("#")) + ": " + str_metrics + " " + str_features)
        
        
        # @var best_model_weights String
        best_model_weights = self.dataset.get_working_dir (self.dataset.task, 'models', 'deep-learning', feature_key, 'checkpoint')
        
        
        # @var checkpoint_monitor String
        checkpoint_monitor = main_metric
        
        
        # @var checkpoint_mode String
        checkpoint_mode = 'max' if task_type == 'classification' else 'min'
        
        
        # Iterate over each model
        for counter, params in enumerate (param_list):
        
            # Attach extra stuff
            params['dataset'] =  self.dataset
            params['features'] = '-'.join (feature_combination)
            params['output_bias'] = initial_bias
            params['inputs_size'] = inputs_size
            
            
            # Attach word embeddings stuff
            if 'we' in feature_combination:
                params['tokenizer'] = tokenizer
                params['maxlen'] = maxlen
                params['trainable_embedding_layer'] = True
            
            
            # @var model Keras model based on the parameters
            model = kerasModel.create (**params)
            
            
            # @var early_stopping Early Stopping callback
            early_stopping = self.get_early_stopping_callback (
                patience = self.get_patience_per_achitecture (params['architecture'], params['features'])
            )
            
            
            # @var checkpoint_callback ModelCheckpoint
            checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint (
                filepath = best_model_weights,
                monitor = checkpoint_monitor, 
                save_best_only = True,
                save_weights_only = True,
                mode = checkpoint_mode,
                verbose = 0
            )    
            
            
            # @var lr_scheduler LearningRateScheduler
            lr_scheduler = self.get_learning_rate_scheduler_callback (
                lr = params['lr'],
                epochs = params['epochs']
            )
            
            
            # @var logs_path_model String
            logs_path_model = self.dataset.get_working_dir (self.dataset.task, 'models', 'deep-learning', feature_key, 'logs', 'model-' + str (counter))
            
            
            # @var tboard_callback String
            tboard_callback = tensorflow.keras.callbacks.TensorBoard (
                log_dir = logs_path_model, 
                histogram_freq = 1, 
                profile_batch = '500,520'
            ) if self.log else None
            
            
            # @var datasets Dict
            # @link https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle/48096625#48096625
            datasets = {split: tensorflow.data.Dataset.from_tensor_slices ((x[split], y[split])) \
                .cache () \
                .shuffle (len (y[split]), reshuffle_each_iteration = False) \
                .batch (params['batch_size']) \
                    for split in ['train', 'val']}
            
            
            # Reset graph
            tensorflow.compat.v1.reset_default_graph ()
            tensorflow.keras.backend.clear_session ()
            
            
            # Get all the features String
            str_features = ' '.join ([f'{str(params[feature])[:6]: >{max (6, len (feature))}}' for feature in features_to_show])
            
            
            # @var callbacks List
            callbacks = [checkpoint_callback, early_stopping, lr_scheduler, tboard_callback]
           
            
            # @var history
            try:
                history = model.fit (
                    x = datasets['train'], 
                    validation_data = datasets['val'],
                    epochs = params['epochs'],
                    callbacks = [callback for callback in callbacks if callback],
                    verbose = 1 if 'we' in params['features'] else 0,
                    class_weight = class_weights
                )
                
                
                # The model weights (that are considered the best) are loaded again into the model.
                model.load_weights (best_model_weights)
            
            
                # Save evaluated model
                model.save (self.dataset.get_working_dir (self.dataset.task, 'models', 'deep-learning', feature_key, 'model-' + str (len(results)) + '.h5'))
            
            
            # User exit
            except KeyboardInterrupt:
                print ("@user exit")
                break
            
            
            # Go for the next model
            except:
                traceback.print_exc()
                print ("@exception")
                continue
            
            # @var features Dict
            features = {feature: params[feature] for feature in features_to_show}
            
            
            # @var model_results Dict
            model_results = {
                'best': 0,
                'epochs': early_stopping.stopped_epoch or params['epochs']
            }
            
            
            # Get results for this model
            if task_type in ['classification', 'multi_label']:
            
                # @var validation_predictions 
                validation_predictions = self.get_y_pred (model, datasets['val'])
                
                
                # @var train_predictions 
                train_predictions = self.get_y_pred (model, datasets['train'])
                
                
                # @var validation_real 
                validation_real = self.get_y_real (datasets['val'])
                
                
                # @var train_real 
                train_real = self.get_y_real (datasets['train'])
                
                
                # @var f1_score_scheme String
                f1_score_scheme = dataset_options['f1_score_scheme'] \
                    if 'f1_score_scheme' in dataset_options else 'micro'

                
                # Store results
                if 'loss' in metrics_to_show:
                    model_results['loss'] = history.history['loss'][-1]
                
                if 'val_loss' in metrics_to_show:
                    model_results['val_loss'] = history.history['val_loss'][-1]
                
                if 'accuracy' in metrics_to_show:
                    model_results['accuracy'] = accuracy_score (y_true = train_real, y_pred = train_predictions)
                
                if 'val_accuracy' in metrics_to_show:
                    model_results['val_accuracy'] = accuracy_score (y_true = validation_real, y_pred = validation_predictions)
                
                if 'f1_score' in metrics_to_show:
                    model_results['f1_score'] = f1_score (y_true = train_real, y_pred = train_predictions, average = f1_score_scheme)
                    
                if 'val_f1_score' in metrics_to_show:
                    model_results['val_f1_score'] = f1_score (y_true = validation_real, y_pred = validation_predictions, average = f1_score_scheme)
                
            
            # For regression
            if 'regression' == task_type:
            
                # Training
                metrics_results = model.evaluate (datasets['train'], verbose = 0, batch_size = len (train_df)) 
                for index, metric in enumerate (model.metrics_names):
                    if metric in metrics_to_show:
                        model_results[metric] = metrics_results[index]
            
                
                # Validation
                metrics_results = model.evaluate (datasets['val'], verbose = 0, batch_size = len (val_df)) 
                for index, metric in enumerate (model.metrics_names):
                    if ('val_' + metric) in metrics_to_show:
                        model_results['val_' + metric] = metrics_results[index]
            
            
            # Get best model according to the best metric
            if (task_type in ['classification', 'multi_label'] and model_results[main_metric] > metric_of_the_best_model) or ('regression' == task_type and model_results[main_metric] < metric_of_the_best_model):
                metric_of_the_best_model = model_results[main_metric]
                best_model = model
                train_resume['best_model_index'] = counter
                model_results['best'] = 1
            

            # Append the result
            results.append ({
                **model_results, 
                **features
            })
            
            
            # @var str_metrics List
            str_metrics = {label: (round (metric, 4) if label not in ['epochs', 'best'] else metric) for label, metric in model_results.items ()}
            str_metrics  = ' '.join ([f'{value: >{max (6, len (metric))}}' for metric, value in str_metrics.items ()])
            
            
            # Start
            print ("{:>3}".format (str (counter + 1)) + ": " + str_metrics + " " + str_features)
            
            
            """
            # @for testing
            print ("useful data")
            print ("===========")
            t = int (len (history.history['val_loss']) / 6)


            print ()
            print ("history")
            print ("------")
            for metric in ['accuracy', 'loss', 'val_accuracy', 'val_loss']:
                print (f'{metric: >12}' + ": " + str ([round (np.average (history.history[metric][i:i + t]), 4) for i in range (0, len (history.history[metric]), t)]))
            
            
            print ()
            print ("report")
            print ("------")
            
            print (pd.DataFrame (classification_report (
                y_true = train_real, 
                y_pred = train_predictions, 
                digits = 5,
                output_dict = True
            )).T)

            
            print (pd.DataFrame (classification_report (
                y_true = validation_real, 
                y_pred = validation_predictions, 
                digits = 5,
                output_dict = True
            )).T)
            
            print ()
            print ("summary")
            print ("------")
            print (model.summary ())
            """
            
        
        
        # There are no results. Possible by a control+c soon
        if not results:
            sys.exit ()
        
        
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
        scan_object_data.to_csv (params_filename, index = True)
        
        
        # @var best_features Series
        best_features = scan_object_data.iloc[train_resume['best_model_index']]
        
        
        
        # Save the best model
        best_model.save (best_model_file)


        # @var best_model_summary String
        best_model_summary = self.dataset.get_working_dir (self.dataset.task, 'models', 'deep-learning', feature_key, 'model-summary.txt')
        
        
        # Save summary
        with open (best_model_summary, 'w') as f:
            with redirect_stdout (f):
                best_model.summary ()
        
        
        # @var _model_file String
        resume_file = self.dataset.get_working_dir (self.dataset.task, 'models', 'deep-learning', feature_key, 'training_resume.json')
        
        
        # Store
        with open (resume_file, 'w') as resume_file:
            json.dump (train_resume, resume_file, indent = 4, sort_keys = True)
        
        
        
        # Training with development set
        print ()
        print ("....................")
        print ("retraining the best model with the development set")
        print ("....................")
        
        
        # Retrain the model with the validation set
        # https://github.com/keras-team/keras/issues/4446
        # Yes, successive calls to fit will incrementally train the model.
        best_model.fit (
            x = datasets['val'],
            epochs = 10,
            verbose = 1,
            class_weight = class_weights
        )
        
        
        # @var best_model_with_val_file String
        best_model_with_val_file = self.dataset.get_working_dir (self.dataset.task, 'models', 'deep-learning', feature_key, 'model_with_val.h5')

        
        # Save the best model
        best_model.save (best_model_with_val_file)


    def get_best_model (self, feature_key, use_train_val = False):
        """
        @inherit
        """
        
        # @var model_index int|None Used to select the best model according to a custom criteria
        model_index = None
        
        
        # Get the best model based on some criteria
        if self.best_model_criteria:
            
            # @var hyperparameter_df DataFrame
            hyperparameter_df = pd.read_csv (self.dataset.get_working_dir (self.dataset.task, 'models', 'deep-learning', feature_key, 'hyperparameters.csv'))
            hyperparameter_df = hyperparameter_df.rename (columns = {'Unnamed: 0': 'index'})

            
            # @var filter String Clone the original criteria and attach filter to get the best
            filter = self.best_model_criteria
            
            
            # Select a subframe
            # @link https://stackoverflow.com/questions/34157811/filter-a-pandas-dataframe-using-values-from-a-dict
            hyperparameter_df = hyperparameter_df.loc[(hyperparameter_df[list (filter)] == pd.Series (filter)).all (axis = 1)]
            
            
            # @var main_metric get metric
            main_metric = self.get_main_metric ()
            
            
            # Update model index
            model_index = hyperparameter_df.sort_values (by = main_metric, ascending = True).tail (1)['val_f1_score'].index.item ()
            
            
        # @var model_name Model name according to criteria
        if use_train_val:
            model_name = 'model_with_val.h5'
            
        elif model_index is not None:
            model_name = 'model-' + str (model_index) + '.h5'
            
        else:
            model_name = 'model.h5'
            
        
        # @var model_filename String Retrieve the filepath of the best model
        model_filename = self.dataset.get_working_dir (self.dataset.task, 'models', 'deep-learning', feature_key, model_name)
        
        
        print (model_name)
        
        # Ensure the model exists. If not, return nothing
        if not os.path.exists (model_filename):
            return
    
    
        # @var best_model KerasModel
        try:
            return tensorflow.keras.models.load_model (model_filename, compile = False)
        
        except:
            print ("model could not be loaded. Skip...")
            return
            

    def predict (self, using_official_test = False, callback = None, use_train_val = False):
        """
        @inherit
        """
        
        # @link https://stackoverflow.com/questions/58814130/tensorflow-2-0-custom-keras-metric-caused-tf-function-retracing-warning/62298471#62298471
        tensorflow.compat.v1.disable_eager_execution ()
        
        
        # @var feature_combination Tuple
        feature_combination = self.get_feature_combinations ()
        
        
        # @var df DataFrame
        df = self.dataset.get ()
        
        
        # @var task_type string Determine if we are dealing with a regression or classification problem
        task_type = self.dataset.get_task_type ()
        
        
        # @var feature_key String
        feature_key = '-'.join (feature_combination)
        
        
        # @var best_model KerasModel
        best_model = self.get_best_model (feature_key, use_train_val = use_train_val)
        
        
        # @var x Dict of features for each subset
        x = {key: pd.DataFrame (self.features[key].transform (df)) for key in feature_combination}
        
        
        # If the supplied dataset contains information of the split, that means that we are dealing
        # only with a subset of the features and we retrieve it accordingly
        if using_official_test:
            x = {'input_' + key: item[item.index.isin (df.index)].reindex (df.index) for key, item in x.items ()}
        
        else:
            x = {'input_' + key: item for key, item in x.items ()}


        # @var raw_predictions Get the logits of the predictions
        raw_predictions = best_model.predict (x)

        
        # According to the number of labels we discern between binary and multiclass
        if 'classification' == task_type:
            if self.dataset.get_num_labels () <= 2:
                predictions = raw_predictions >= 0.5
                predictions = np.squeeze (predictions)
            
            # Multiclass
            else:
                predictions = np.argmax (raw_predictions, axis = 1)
        
        
            # @var true_labels
            true_labels = self.dataset.get_available_labels ()
            
            
            # @var predictions List
            y_pred = [true_labels[int (prediction)] for prediction in predictions]
        
        
        # Multi label tasks
        if 'multi_label' == task_type:
            
            # Round the predictions
            predictions = raw_predictions
            
            
            # @var y_pred List
            y_pred = predictions.round ()
        
        # Regression tasks
        if 'regression' == task_type:
            
            # @var predictions Are the raw predictions
            predictions = raw_predictions
        
            
            # @var y_pred List
            y_pred = predictions.squeeze ()
        
        
        # @var model_metadata Dict
        model_metadata = {
            'model': best_model,
            'probabilities': raw_predictions
        }
        
        
        # run callback
        if callback:
            callback (feature_key, y_pred, model_metadata)
    
    
    def get_y_pred (self, model, X):
    
        # @var task_type
        task_type = self.dataset.get_task_type ()
        
        
        # @var labels 
        labels = self.dataset.get_available_labels ();
    
        
        # @var y_pred Predict 
        y_pred = model.predict (X, batch_size = len (X))
        
        
        # Transform predictions into binary or multiclass
        if 'classification' == task_type:
            y_pred = y_pred > .5 if self.dataset.get_num_labels () <= 2 else np.argmax (y_pred, axis = 1)
            return [labels[int (item)] for item in y_pred]
        
        elif 'multi_label' == task_type:
            return y_pred.round ()
        
    
    def get_y_real (self, dataset):
        
        # @var task_type
        task_type = self.dataset.get_task_type ()
        
        
        # @var labels 
        labels = self.dataset.get_available_labels ();
        
        
        # Transform predictions into binary or multiclass
        if 'classification' == task_type:
            
            # @var y_real 
            y_real = np.concatenate ([y for x, y in dataset], axis = 0)
            y_real = y_real > .5 if self.dataset.get_num_labels () <= 2 else np.argmax (y_real, axis = 1)
            return [labels[int (item)] for item in y_real]
        
        
        if 'multi_label' == task_type:
            return np.concatenate ([y for x, y in dataset], axis = 0)

    
    def get_early_stopping_callback (self, patience):
        
        # @var early_stopping Early Stopping callback
        # @todo Note this: https://github.com/tensorflow/tensorflow/issues/35634
        return tensorflow.keras.callbacks.EarlyStopping (
            monitor = self.get_early_stopping_metric (), 
            patience = patience,
            mode = 'auto',
            restore_best_weights = False
        )
        
    def get_learning_rate_scheduler_callback (self, lr, epochs):
        """
        @link https://towardsdatascience.com/learning-rate-schedule-in-practice-an-example-with-keras-and-tensorflow-2-0-2f48b2888a0c
        """
        
        # @var decay long
        decay = lr / epochs
        def lr_time_based_decay (epoch, lr):
            return lr * 1 / (1 + decay * epoch)
            
        
        # @var lr_scheduler LearningRateScheduler
        return tensorflow.keras.callbacks.LearningRateScheduler (lr_time_based_decay)


    def get_early_stopping_metric (self):
        
        # @var task_type String
        task_type = self.dataset.get_task_type ()
        
        # Regression metrics
        if 'regression' == task_type:
            return 'val_rmse'
        
        # Regression metrics
        if 'classification' == task_type:
            return 'val_prc' if self.dataset.is_imabalanced () else 'val_loss'
        
        # Multi_label metrics
        if 'multi_label' == task_type:
            return 'val_loss'


    def get_patience_per_achitecture (self, architecture, features):
        
        # @var patience int For the early stoppping mechanism
        patience = 100
        
        
        if 'we' in features:
            patience = 10
        
        
        if architecture in ['lstm', 'gru', 'bilstm', 'bigru']:
            patience = 5
            
        if architecture in ['cnn']:
            patience = 10
            
        return patience
