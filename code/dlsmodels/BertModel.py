"""
    Fine tune a BERT Model for an specific task
    
    For this example we are using transformers Trainer
    
    @see config.py
    
    @link https://www.sbert.net/docs/quickstart.html
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import time
import json
import sys
import csv
import torch
import pandas as pd
import numpy as np
import config
import os
import transformers
import sklearn

import torch.nn.functional as F
import torch.utils.data as data_utils

from datasets import Dataset
from torch.utils.data import DataLoader
from torch import nn
from functools import partial



from .BaseModel import BaseModel


class BertModel (BaseModel):
    """
    Bert implementation
    """
    
    # @var all_available_features List
    all_available_features = []
    
    
    # @var total_predictions List
    total_predictions = []
    
    
    # @var total_labels List
    total_labels = []
    
    
    # @var total_probabilities List
    total_probabilities = []
    
    
    # @var field String
    field = 'tweet_clean'
    
    
    # @var batch_train_size int Batch size for training
    batch_train_size = 16
    
    
    # @var batch_val_size int Batch size for validating
    batch_val_size = 64
    
    
    # @var tokenizer Get the pretained tokenizer
    tokenizer = None
    
    
    def get_model_filename (self):
        """
        @return String
        """
        return self.dataset.get_working_dir (self.dataset.task, 'models', 'bert', 'bert-finetunning')
    
    
    def get_pretrained_model (self):
        """
        Get the most suitable BERT model according to the task type and language
        
        @return String
        """
        return 'dccuchile/bert-base-spanish-wwm-uncased' if self.dataset.get_dataset_language () == 'es' else 'bert-base-uncased'
        # return 'bert-base-multilingual-uncased' if self.dataset.get_dataset_language () == 'es' else 'bert-base-uncased'
        # return 'distilbert-base-multilingual-cased' if self.dataset.get_dataset_language () == 'es' else 'bert-base-uncased'
    
    def get_tokenizer_filename (self):
        """
        @return String
        """
        return 'dccuchile/bert-base-spanish-wwm-uncased' if self.dataset.get_dataset_language () == 'es' else 'bert-base-uncased'
        # return 'bert-base-multilingual-uncased' if self.dataset.get_dataset_language () == 'es' else 'bert-base-uncased'
        # return 'distilbert-base-multilingual-cased' if self.dataset.get_dataset_language () == 'es' else 'bert-base-uncased'

    
    def tokenize (self, batch):
        """
        @param batch
        """
        return self.tokenizer (batch[self.field], padding = True, truncation = True)


    def createDatasetFromPandas (self, df, task_type = 'classification'):
        """
        Encode datasets to work with transformers from a DataFrame and 
        "torch" the new columns. 
        
        @param df DataFrame
        @param task_type String
        
        @return Dataset
        """
        
        # Encode label as numbers instead of user names. This is necesary for PyTorch
        df['_temp'] = df['label'].astype (str).astype ('category').cat.codes if task_type == 'classification' else df['label']
        
        
        # Get the only labels we care (labels -if any- and the text)
        df = df[[self.field, '_temp']].rename (columns = {'_temp': 'label'})
        
        
        # @var dataset Dataset
        dataset = Dataset.from_pandas (df)
        
        
        # Get the input_ids and the attention mask
        dataset = dataset.map (partial (self.tokenize), batched = True, batch_size = len (dataset))
        
        
        # Torch (transform into tensors) the input ids, the attention mask, and the label
        dataset.set_format ('torch', columns = ['input_ids', 'attention_mask', 'label'], output_all_columns = False)
        
        
        # Return the result
        return dataset
        
        
    def compute_metrics (self, pred):
        """
        
        This function allows to calculate accuracy, precision, recall, and f1
        for the Trainer huggingface component
        
        @todo Adapt to another classifying problems such as regression or 
              binary
        
        return Dict
        """
        
        # @var task_type String
        task_type = self.dataset.get_task_type ()
        
        
        # @var labels
        labels = pred.label_ids
        
        
        # @var preds
        preds = pred.predictions.argmax (-1) if 'classification' == task_type else pred.predictions
        
        
        # Store metrics in the class
        self.total_predictions = preds
        self.total_labels = labels
        self.total_probabilities = pred.predictions

        
        # Classification metrics
        if 'classification' == task_type:
        
            # @var precision Float @var recall Float @var f1 Float @var _support array
            precision, recall, f1, _support = sklearn.metrics.precision_recall_fscore_support (labels, preds, average = 'macro')
            
            
            # @var acc Float Accuracy
            acc = sklearn.metrics.accuracy_score (labels, preds)
            
            return {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        # Regression
        else:
        
            return {
            }
    
    def train (self, num_train_epochs = 3.0, using_official_test = True, force = False):
        """
        @inherit
        
        @param num_train_epochs float
        @param using_official_test Boolean
        @param force Boolean
        
        """
    
        # @var df DataFrame
        df = self.dataset.get ()
        
        
        # @todo @aalmodobar
        df = df[df['label'].notna ()]
        df = df[df['__split'].notna()]
        
        
        # @var val_split String 
        # Determine which split do we get based on if we 
        # want to validate the val dataset of the train dataset
        val_split = 'val' if not using_official_test else 'test'
        
        
        # @var training_args TrainingArguments
        training_args = transformers.TrainingArguments (
            output_dir = './results',
            per_device_train_batch_size = self.batch_train_size,
            per_device_eval_batch_size = self.batch_val_size,
            warmup_steps = 500,
            weight_decay = 0.01,
            logging_dir = './logs',
        )
    
    
        # Load the tokenizer
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained (self.get_tokenizer_filename ())

        
        # @var train_df DataFrame Get training split
        train_df = self.dataset.get_split (df, 'train')
        
        
        # @var val_df DataFrame Get validation split
        val_df = self.dataset.get_split (df, val_split)
        
        
        # @var task_type String
        task_type = self.dataset.get_task_type ()

        
        # @var train_dataset Dataset Encode datasets to work with transformers
        train_dataset = self.createDatasetFromPandas (train_df, task_type)
        
        
        # @var val_dataset Dataset Encode datasets to work with transformers
        val_dataset = self.createDatasetFromPandas (val_df, task_type)
        
        
        # @var model BertForSequenceClassification Loads your fine-tunned model
        model = transformers.BertForSequenceClassification.from_pretrained (self.get_pretrained_model (), num_labels = self.dataset.get_num_labels ())
        
        
        # @var trainer
        trainer = transformers.Trainer (
            model = model, 
            args = training_args, 
            train_dataset = train_dataset, 
            eval_dataset = val_dataset,
            compute_metrics = self.compute_metrics 
        )
    
        
        # Train
        trainer.train ()
        
        
        # Finally, after training evaluate the val_dataset
        # @todo. check if this step is necessary
        model.eval ()
        
        
        # Evaluate
        print (json.dumps (trainer.evaluate (), indent = 2))
        
    
        # Ensure path exists
        os.makedirs (os.path.dirname (self.get_model_filename ()), exist_ok = True)


        # Save model
        model.save_pretrained (self.get_model_filename ())
        self.tokenizer.save_pretrained (self.get_model_filename ())
        
        
    def predict (self, using_official_test = False, callback = None):
        """
        @inherit
        """
        
        # @var df DataFrame
        df = self.dataset.get ()
        
        
        # @todo @aalmodobar
        """
        df = df[df['label'].notna ()]
        """
        
        
        # @var model BertForSequenceClassification
        model = transformers.BertForSequenceClassification.from_pretrained (self.get_model_filename (), num_labels = self.dataset.get_num_labels ())


        # @var task_type string
        task_type = self.dataset.get_task_type ()
        
        
        # @var tokenizer Get the pretained tokenizer
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained (self.get_tokenizer_filename ())
        
        
        # Set the model in evaluation form in order to get reproductible results
        model.eval ()
        
        
        # @var hugging_face_dataset Dataset Encode datasets to work with transformers
        hugging_face_dataset = self.createDatasetFromPandas (df)
        
        
        # @var training_args TrainingArguments
        training_args = transformers.TrainingArguments (
            output_dir = './results',
            num_train_epochs = 0,
            per_device_train_batch_size = 0,
            per_device_eval_batch_size = self.batch_val_size,
            warmup_steps = 500,
            weight_decay = 0.01,
            logging_dir = './logs',
        )
        
        
        # @var trainer Trainer Get the trainer, but only for evaluating purposes
        trainer = transformers.Trainer (
            model = model, 
            args = training_args, 
            eval_dataset = hugging_face_dataset, 
            compute_metrics = self.compute_metrics
        )
        
        
        # @var predictions @todo I think I should return this
        predictions = trainer.predict (hugging_face_dataset)
        
        
        # Transform predictions into labels
        if 'classification' == task_type:

            # @var true_labels Get true labels as string
            true_labels = self.dataset.get_true_labels ()
            
            
            # @var y_predicted_classes
            y_predicted_classes = [true_labels[int (prediction)] for prediction in self.total_predictions]
            
        
        
        # @var model_metadata Dict
        model_metadata = {
            'model': model,
            'created_at': time.ctime (os.path.getmtime (self.get_model_filename ())),
            'probabilities': self.total_probabilities
        }
        
        
        # run callback
        if callback:
            callback ('bert', y_predicted_classes, model_metadata)
        
        
