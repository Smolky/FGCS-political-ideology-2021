import sys
import os.path
import io
import csv
import math
import bootstrap
import numpy as np
import pandas as pd
import sklearn
import joblib
import pickle
import time

from pathlib import Path
from pipelinehelper import PipelineHelper

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report

import dask_ml.model_selection as dcv
from dask.diagnostics import ProgressBar



from .BaseModel import BaseModel


class ClassicMachineLearningTechniques (BaseModel):
    """
    ClassicMachineLearningTechniques
    
    This class allows to train and use traditional machine-learning models 
    used with scikit-learn. The idea is that this model will train these models
    over different feature set and store the best model for later use and 
    analysis
    """
    
    # @var all_available_features List
    all_available_features = ['lf', 'se', 'be', 'ne', 'cf', 'pr', 'ng']
    
    
    def has_external_features (self):
        """
        @inherit
        """        
        return False
        
    
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
        
        
        # @var train_val_df DataFrame 
        # As we use RandomizedSearchCV, we need to merge train and validation splits
        # and use PredefinedSplit to indicate which elements of the list were used 
        # for validation and which ones for training. 
        # In this sense, we always use the same validation split
        train_val_df =  pd.concat ([train_df, val_df])
        
        
        # @var available_labels
        available_labels = self.dataset.get_available_labels ()
        
        
        # @var split_index List Train data indices are -1 and validation data indices are 0
        # @link https://stackoverflow.com/questions/31948879/using-explicit-predefined-validation-set-for-grid-search-with-sklearn
        split_index = [-1 if x in train_df.index else 0 for x in train_val_df.index]
    
    
        # @var split int Create a Predefined split based on my indexes
        split = sklearn.model_selection.PredefinedSplit (test_fold = split_index)
        
        
        # @var n_estimators int 
        # Default number of estimators, used with the baggingclassifier
        # to split it and use the multi-core capabilities
        # @link https://stackoverflow.com/questions/31681373/making-svm-run-faster-in-python
        n_estimators = 10
        
        
        # @var classifiers Dict Classifiers to evaluate and their default parameters
        classifiers = [
            ('lr', LogisticRegression (max_iter = 4000)),
            ('rf', RandomForestClassifier ()),
            ('svm', SVC (probability = True))
        ]
        
        
        # @var pipe Pipeline Create pipeline with the feature selection and the classifiers
        # @todo Bring combination of the parameters in the pipeline
        pipe = Pipeline ([
            ('classifier', PipelineHelper (classifiers))
        ])
    
    
        # Define some parameters space beforehand
        # @var rf__max_depth
        rf__max_depth = [int(x) for x in np.linspace (10, 110, num = 5)]
        rf__max_depth.append (None)
    
        
        # @var rf__n_estimadors List
        rf__n_estimadors = [int (x) for x in np.linspace (start = 200, stop = 2000, num = 5)]
        
        
        # @var hyperparameters Dict  
        hyperparameters = {
            'rf__n_estimators': rf__n_estimadors,
            'rf__max_features': ['auto', 'sqrt'],
            'rf__max_depth': rf__max_depth,
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4],
            'lr__solver': ['liblinear', 'lbfgs'],
            'lr__fit_intercept': [True, False],
            'svm__C': [1],
            'svm__kernel': ['rbf', 'poly', 'linear'],
        }
    
    
        # @var classifier_hyperparameters Filter only those parameters related to the classifiers we use
        classifier_hyperparameters = {key: hyperparameter for key, hyperparameter in hyperparameters.items () 
                                    if key.startswith (tuple ([(classifier_key[0] + "__") for classifier_key in classifiers]))}
    

        # @var parameters Dictionary
        parameters = {
            'classifier__selected_model': pipe.named_steps['classifier'].generate (classifier_hyperparameters)
        }
        
        
        # @var feature_combination Tuple
        feature_combination = self.get_feature_combinations ()
        
        
        # @var feature_key String
        feature_key = '-'.join (feature_combination)
        
        
        # @var best_model_file String
        best_model_file = self.dataset.get_working_dir (self.dataset.task, 'models', 'classic', feature_key, 'best_model.joblib')
        
        
        # If the file exists, then skip (unless we force to retrain)
        if os.path.isfile (best_model_file):
            if not force:
                return
            
            
        # @var features FeatureUnion
        features = pd.DataFrame (FeatureUnion ([(key, self.features[key]) for key in feature_combination]).fit_transform ([]))
        
        
        # Select the split for training and testing of the selected features
        train_val_features = features[features.index.isin (train_val_df.index)].reindex (train_val_df.index)
        
        
        # Moreover, we are going to generate data to refit the final model using only the 
        # train dataset
        train_features = features[features.index.isin (train_df.index)].reindex (train_df.index)
        
        
        # @var scoring_metric
        scoring_metric = self.dataset.get_scoring_metric ().replace ('val_', '')
        
        
        # Adjust F1-Score metric
        if 'scoring' in dataset_options:
            if dataset_options['scoring'] == 'micro':
                scoring_metric = 'f1_micro'
            
            elif dataset_options['scoring'] == 'macro':
                scoring_metric = 'f1_macro'

            elif dataset_options['scoring'] == 'weighted':
                scoring_metric = 'f1_weighted'
            
            else:
                scoring_metric = 'f1_weighted'
                
        elif scoring_metric == 'f1_score':
            scoring_metric = 'f1_weighted'
        
        
        # @var n_iter int Number of models to test
        n_iter = 100
        
        
        # @var search RandomizedSearchCV
        search = dcv.RandomizedSearchCV (pipe, parameters, 
            cv = split, 
            n_iter = n_iter, 
            scoring = scoring_metric, 
            random_state = bootstrap.seed,
            refit = True
        )
        
        
        # Fit
        with ProgressBar ():
            search.fit (train_val_features, train_val_df['label'].cat.codes)
        
        
        # @var df_summary DataFrame
        df_summary = pd.DataFrame (search.cv_results_) \
            .drop (columns = ['std_fit_time', 'std_score_time', 'param_classifier__selected_model', 'std_test_score', 'split0_test_score'], axis = 1) \
            .sort_values ('rank_test_score', ascending = True) \
            [['rank_test_score', 'mean_test_score', 'mean_fit_time', 'mean_score_time', 'params']]
        
        
        # Refit the estimator
        # @var link https://stackoverflow.com/questions/57059016/refit-attribute-of-grid-search-with-pre-defined-split
        search.best_estimator_.fit (train_features, train_df['label'].cat.codes)
        
        
        # Store the results for further anaylsis
        joblib.dump (search.best_estimator_, best_model_file)
        
        
        # @var summary_path String
        summary_path = self.dataset.get_working_dir (self.dataset.task, 'models', 'classic', feature_key, 'hyperparameters.csv')
        
        
        # Output summaries
        df_summary.to_csv (summary_path, index = False, quoting = csv.QUOTE_ALL)

    
    def predict (self, using_official_test = False, callback = None):
        """
        @inherit
        
        @todo using_official_test
        
        """

        # @var feature_combination
        feature_combination = self.get_feature_combinations ()

        
        # @var df DataFrame
        df = self.dataset.get ()
        
        
        # @var true_labels
        true_labels = self.dataset.get_available_labels ()

    
        # @var feature_key String
        feature_key = '-'.join (feature_combination)
        
        
        # @var model_filename String
        model_filename = self.dataset.get_working_dir (self.dataset.task, 'models', 'classic', feature_key, 'best_model.joblib')
        
        
        # Ensure them model exists
        if not os.path.exists (model_filename):
            return

        
        # @var best_model
        try:
            best_model = joblib.load (model_filename)
        except:
            print ("model could not be loaded. Skip...")
            return
        
        
        # @var features FeatureUnion
        features = FeatureUnion ([(key, self.features[key]) for key in feature_combination]).fit_transform ([])
        features = pd.DataFrame (features)
        

        # If the supplied dataset contains information of the split, that means that we are dealing
        # only with a subset of the features and we retrieve it accordingly
        if using_official_test:
            features = features[features.index.isin (df.index)].reindex (df.index)
        
        
        # @var predictions
        y_pred = best_model.predict (features)

        
        # @var y_pred
        y_pred = [true_labels[int (item)] for item in y_pred]
        

        # @var predictions
        # predictions = best_model.predict_proba (features)
        predictions = None


        # @var model_metadata Dict
        model_metadata = {
            'model': best_model,
            'created_at': time.ctime (os.path.getmtime (model_filename)),
            'probabilities': predictions
        }
        
        
        # Store the results
        if callback:
            callback (feature_key, y_pred, model_metadata)
