import pandas as pd
import sys
import numpy as np

from sklearn.metrics import f1_score
from sklearn.utils.extmath import weighted_mode

from pathlib import Path
from features.FeatureResolver import FeatureResolver
from .ModelResolver import ModelResolver
from .BaseModel import BaseModel
from sklearn.metrics import classification_report

class EnsembleModel (BaseModel):


    """
    Ensemble Model
    
    This model uses the predictions of the other models
    
    """
    
    def train (self, force = False, using_official_test = True):
        """
        @inherit
        """
        
        
    def predict (self, using_official_test = False, callback = None):
        """
        @inherit
        """
        
        # @var df DataFrame
        df = self.dataset.get ()
        
        
        # @var model_resolver ModelResolver
        model_resolver = ModelResolver ()
        
        
        # @var result_predictions Dict To store predictions of each model / feature set
        result_predictions = {}
        
        
        # @var result_probabilities Dict To store probabilities of each model / feature set
        result_probabilities = {}


        # @var models List @todo Parametrize
        models = ['deep-learning']
        
        
        # @var task_type String
        task_type = self.dataset.get_task_type ()
        
            
        # @var number_of_classes int
        number_of_classes = self.dataset.get_num_labels ()


        # @var labels List
        labels = self.dataset.get_available_labels ();

    
        # Determine if the task is binary or multi-class
        is_binary = number_of_classes <= 2
        
        
        # @var features_in_the_ensemble List @todo Parametrize?
        features_in_the_ensemble = ['lf', 'se', 'be']

        
        def callback_ensemble (feature_key, y_pred, model_metadata):
        
            """
            This callback is used to keep track of the result of each model
            separately
            
            @param feature_key String
            @param y_pred List
            @param model_metadata Dict
            """
            
            # Store predictions
            result_predictions[feature_key] = pd.DataFrame (
                {feature_key: y_pred} if task_type == 'classification' else y_pred,
                index = df.index
            )

            # @var columns List
            columns = [feature_key + "_" + _label for _label in self.dataset.get_available_labels ()] \
                if task_type == 'classification' else feature_key
            

            
            # For binary classification problems
            if self.dataset.get_num_labels () <= 2:
                columns = [columns[0]]

            
            # Store probabilities
            result_probabilities[feature_key] = pd.DataFrame (
                model_metadata['probabilities'], 
                columns = columns, 
                index = df.index
            )
        
        
        # Iterate over ensemble models
        for model_key in models:
        
            # @var model Model Configure each model of the ensemble
            model = model_resolver.get (model_key)
            model.set_dataset (self.dataset)
            model.is_merged (self.dataset.is_merged)
            
            
            # Evaluate models with external features, that is, features from transformers
            if model.has_external_features ():
            
                # @var feature_resolver FeatureResolver
                feature_resolver = FeatureResolver (self.dataset)
                
                
                # @var available_features List
                available_features = model.get_available_features ()
                
                
                print (available_features)
                
                
                # Iterate over all available features
                for feature_set in available_features:
                
                    # Skip those features that we don't want to bert part of the ensemble
                    if feature_set not in features_in_the_ensemble:
                        continue
                    
                    
                    # @var feature_file String
                    feature_file = feature_resolver.get_suggested_cache_file (feature_set, task_type)
                    
                    
                    # @var features_cache String The file where the features are stored
                    features_cache = self.dataset.get_working_dir (self.dataset.task, feature_file)
                    
                    
                    # Indicate what features are loaded
                    if not Path (features_cache).is_file ():
                        print ("skipping " + feature_file)
                        continue
                    
                    
                    # @var transformer Load the features
                    transformer = feature_resolver.get (feature_set, cache_file = features_cache)


                    # Set the features in the model
                    model.set_features (feature_set, transformer)


                    # Perform the prediction
                    print ("predict " + feature_set)
                    model.predict (using_official_test = using_official_test, callback = callback_ensemble)
                    
                    
                    # Clear session data for the model
                    model.clear_session ();
            
            # Models with no external features
            else:
            
                # Perform the prediction
                model.predict (using_official_test = using_official_test, callback = callback_ensemble)
                
                
        # @var concat_df Dataframe The ensemble composed by 
        ensemble_df = pd.concat (result_predictions.values (), axis = 'columns')
        
        
        
        # Classification ensembles
        if task_type == 'classification':
        
            # @var weights List|None
            weights = None

        
            if self.dataset.default_split == 'test':
                weights = pd.read_csv (self.dataset.get_working_dir (self.dataset.task, 'results', 'val', 'ensemble', 'ensemble-weighted-' + '-'.join (features_in_the_ensemble), 'weights.csv')).to_dict (orient='list')
                weights = {key: weight[0] for key, weight in weights.items ()}
            
            else:
                weights = {
                    feature: f1_score (
                        y_true = self.dataset.get ()['label'], 
                        y_pred = ensemble_df[feature], 
                        average = 'weighted'
                    ) for feature in ensemble_df.columns
                }
                
                
                # Normalize to 0 ... 1 scale
                weights = {key: (weight / sum (weights.values ())) for key, weight in weights.items ()}
                
                
            # @var weights Dict Filter only the weights of the features we are interested in
            weights = {key: weight for key, weight in weights.items () if key in features_in_the_ensemble}
            
            
            # @var y_pred_weighted Soft voting ensemble
            y_pred_weighted = ensemble_df[features_in_the_ensemble].apply (lambda row: weighted_mode (row, list (weights.values ()))[0][0], axis=1).to_list ()
            
            
            # @var y_pred_mean Mean probabilities
            y_pred_mean = pd.concat (result_probabilities, axis = 1)
            y_pred_mean = pd.concat (
                [y_pred_mean.iloc[:, t::len (labels)].mean (axis = 1) for t in range (len (labels))], 
                axis = 1,
            )
            y_pred_mean.columns = labels
            # y_pred_mean = y_pred_mean.mean (axis = 1)
            # y_pred_mean = y_pred_mean > .5 if is_binary else np.argmax (y_pred_mean, axis = 1)
            y_pred_mean = y_pred_mean > .5 if is_binary else np.argmax (y_pred_mean.values, axis = 1)
            y_pred_mean = [labels[int (item)] for item in y_pred_mean]

        
            # @var y_pred_mode Ensemble based on the mode (hard voting)
            y_pred_mode = ensemble_df[features_in_the_ensemble].mode (axis = 'columns')[0]
            
            
            # @var y_pred_hightest_chance is the mode
            y_pred_hightest_chance = pd.concat (result_probabilities, axis = 1)
            
            
            # Binary
            if is_binary:
                y_pred_hightest_chance = y_pred_hightest_chance.max (axis = 1)
                y_pred_hightest_chance = y_pred_hightest_chance > .5
                y_pred_hightest_chance = [labels[int (item)] for item in y_pred_hightest_chance]
            
            # Multiclass
            else:
                y_pred_hightest_chance = y_pred_hightest_chance.idxmax (axis = 1)
                y_pred_hightest_chance = [item[1].split ('_')[1] for item in y_pred_hightest_chance]

            
            
            # @var probabilities List
            probabilities = []
            
            
            # @var merged_probabilities DataFrame
            merged_probabilities = pd.concat (result_probabilities.values (), axis = 1)
            
            
            # Iterate...
            for idx, y_pred in enumerate (y_pred_weighted):
            
                # @var labels Series
                labels = ensemble_df[features_in_the_ensemble].iloc[idx]
            
                
                # @var feature_sets List
                feature_sets = [feature_set for feature_set, label in labels.iteritems () if label == y_pred]
                
                
                # @var temp Dict
                temp = {}
            
            
                #Iterate over each label
                for label in self.dataset.get_available_labels ():
                    
                    # @var cols List Retrieve the labels that match the matching class
                    cols = [col for col in merged_probabilities \
                        if col.startswith (tuple (feature_sets)) and col.endswith ('_' + label)]
                    
                    
                    # Calculate values
                    temp[label] = merged_probabilities.iloc[idx][cols].mean ()


                # Attach probabilities for each label
                probabilities.append (list (temp.values ()))
        
        
            # @var model_metadata Dict
            model_metadata = {
                'model': None,
                'created_at': '',
                'probabilities': probabilities,
                'weights': weights
            }
            
            
            # Run the callbacks
            if callback:
                callback (feature_key = 'ensemble-mean-' + '-'.join (features_in_the_ensemble), y_pred = y_pred_mean, model_metadata = model_metadata)
                callback (feature_key = 'ensemble-mode-' + '-'.join (features_in_the_ensemble), y_pred = y_pred_mode, model_metadata = model_metadata)
                callback (feature_key = 'ensemble-weighted-' + '-'.join (features_in_the_ensemble), y_pred = y_pred_weighted, model_metadata = model_metadata)
                callback (feature_key = 'ensemble-highest-' + '-'.join (features_in_the_ensemble), y_pred = y_pred_hightest_chance, model_metadata = model_metadata)
            
        

        # Regression tasks
        if task_type == 'regression':
        
            print (ensemble_df)
        
            # @var y_pred_mode is the mode
            y_pred_mode = ensemble_df[features_in_the_ensemble].mean (axis = 'columns')[0]        
        
            print (y_pred_mode)
            print ("@todo")
            sys.exit ()