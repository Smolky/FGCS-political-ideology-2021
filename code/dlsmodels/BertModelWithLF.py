"""
    This model tries to combine the linguistic features with fine-tuned BERT model
    
    @see BertModel
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Ricardo Colomo-Palacios <ricardo.colomo-palacios@hiof.no>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from tqdm import tqdm

from datasets import Dataset
from torch.utils.data import DataLoader
from torch import nn
from functools import partial

from utils.PrettyPrintConfussionMatrix import PrettyPrintConfussionMatrix

from .BaseModel import BaseModel



class CustomBERTModel (nn.Module):
    """
    CustomBERTModel
    
    This model mixes the fine tunned BERT model with custom features based 
    on linguistic features
    """
    
    def __init__ (self, pretrained_model, input_size, num_classes):
        """
        @param input_size
        @param num_classes
        """
        super (CustomBERTModel, self).__init__()
        
        
        # Store num classes
        self.num_classes = num_classes
        
        
        # Init BERT model in a way that returns information in a dict mode
        self.bert = transformers.BertForSequenceClassification.from_pretrained (pretrained_model, 
            return_dict = True, 
            output_hidden_states = True,
            num_labels = num_classes
        )
        
        
        # Linguistic features layers
        self.fc1 = nn.Linear (input_size + (768 * 1), 48)
        self.fc2 = nn.Linear (48, 32)
        self.fc2_dropout = nn.Dropout (0.1)
        self.fc3 = nn.Linear (32, 16)
        self.fc4 = nn.Linear (16, num_classes)
    
    
    def forward (self, lf, input_ids, token_type_ids = None, attention_mask = None, position_ids = None, head_mask = None, epoch = 0):
        
        # Get BERT results
        with torch.no_grad ():
            sequence_output = self.bert (input_ids, attention_mask = attention_mask)
        
        
        # @var hidden_states Get BERT hidden_states
        hidden_states = sequence_output.hidden_states
        
        
        # @var cls_tokens The first token for each batch
        # @link https://stackoverflow.com/questions/61465103/how-to-get-intermediate-layers-output-of-pre-trained-bert-model-in-huggingface
        
        # This way works fine, getting the last layer
        cls_tokens = hidden_states[-1][:, 0]
        
        
        # @var combined_with_bert_features Combine BERT with LF
        combined_with_bert_features = torch.cat ((cls_tokens, lf), dim = 1)
        
        
        # Handle LF
        lf_x = F.relu (self.fc1 (combined_with_bert_features))
        lf_x = F.relu (self.fc2_dropout (self.fc2 (lf_x)))
        lf_x = F.relu (self.fc3 (lf_x))
        lf_x = self.fc4 (lf_x)
        
        
        # According to the task type, we need to apply a sigmoid function
        # or return the value as it is
        lf_x = torch.sigmoid (lf_x) if self.num_classes == 1 else lf_x
        
        
        return lf_x
        
        
class BertModelWithLF (BaseModel):

    # @var all_available_features List
    all_available_features = ['lf']
    
    
    # @var field String
    field = 'tweet_clean'


    # @var device Get device
    device = torch.device ('cuda') if torch.cuda.is_available () else torch.device ('cpu')


    # @var tokenizer Get the pretained tokenizer
    tokenizer = None
    
    
    # @var epochs int
    epochs = 5
    
    
    # @var training_batch_size int
    training_batch_size = 64
    
    
    # @var validation_batch_size int
    validation_batch_size = 128
    
    
    def get_model_filename (self):
        """
        @return String
        """
        return self.dataset.get_working_dir (self.dataset.task, 'models', 'bert', 'bert-with-lf.pt')
        

    def tokenize (self, batch):
        """
        @param batch
        
        @refactor
        """
        return self.tokenizer (batch[self.field], padding = True, truncation = True)
        

    def createDatasetFromPandas (self, df, lf):
        """
        Encode datasets to work with transformers from a DataFrame and 
        "torch" the new columns. 
        
        @param df
        @param lf
        
        @return Dataset
        """
        
        # Encode label as numbers instead of user names. This is necesary for PyTorch
        df['_temp'] = df['label'].cat.codes
        
        
        # Get the only labels we care (labels -if any- and the text)
        df = df[[self.field, '_temp']].rename (columns = {'_temp': 'label'})
        
        
        # Force labels to be encoded as float
        df['label'] = df['label'].astype (np.float32)
        
        
        # @var dataset Dataset
        dataset = Dataset.from_pandas (df)
        
        
        # Get the input_ids and the attention mask
        dataset = dataset.map (partial (self.tokenize), batched = True, batch_size = len (dataset))
        
        
        # Torch (transform into tensors) the input ids, the attention mask, and the label
        dataset.set_format ('torch', columns = ['input_ids', 'attention_mask', 'label'], output_all_columns = False)
        
        
        # Create a dataset with the linguistic features joined, the input id, the attention mask, and the labels
        dataset = data_utils.TensorDataset (
            torch.tensor (lf.values), 
            dataset['input_ids'],
            dataset['attention_mask'],
            dataset['label']
        )

        
        
        # Return the result
        return dataset
        
    def get_criterion (self, num_classes):
        """
        Return the loss function most suited for this task
        """
    
        # @var criterion Set the loss Criteria according to the task type
        criterion = torch.nn.BCELoss () if num_classes == 1 else torch.nn.CrossEntropyLoss ()
        criterion = criterion.to (self.device)

        return criterion
        

    def train (self, num_train_epochs = 3.0, using_official_test = True):
        """
        @inherit
        
        @param num_train_epochs float
        @param using_official_test
        """
        
        # @var df DataFrame
        df = self.dataset.get ()
        
        
        # @todo @aalmodobar
        df = df[df['label'].notna ()]
        

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
        
        
        # @var x Dict of features for the linguistic features
        x = {}
        for subset in ['train', 'val']:
            x[subset] = {}

            features = pd.DataFrame (self.features['lf'].transform (df))
            x[subset]['input_lf'] = features[features.index.isin (indexes[subset])].reindex (indexes[subset])
        

        # @var input_size int 
        input_size = x['train']['input_lf'].shape[1]
    
    
        # @var pretrained_model String
        pretrained_model = self.dataset.get_working_dir (self.dataset.task, 'models', 'bert', 'bert-finetunning')
        
        
        # @var tokenizer Load the tokenizer
        # @todo Get pretrained tokenizer
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained ('dccuchile/bert-base-spanish-wwm-uncased')
        
        
        # @var task_type string Determine if we are dealing with a regression or classification problem
        task_type = self.dataset.get_task_type ()
        
        
        # @var num_classes
        num_classes = self.dataset.get_num_labels ()
        
        
        # Adjust the number of classes of the model for binary classification tasks
        num_classes = num_classes if num_classes > 2 else 1
        
        
        # @var model CustomBERTModel model
        model = CustomBERTModel (
            pretrained_model = pretrained_model, 
            input_size = input_size, 
            num_classes = num_classes
        )
        model.to (self.device)
        
        
        # @var train_dataset Dataset Encode datasets to work with transformers
        train_dataset = self.createDatasetFromPandas (train_df, x['train']['input_lf'])
        
        
        # @var val_dataset Dataset Encode datasets to work with transformers
        val_dataset = self.createDatasetFromPandas (val_df, x['val']['input_lf'])
        
        
        # @var train_loader 
        train_loader = torch.utils.data.DataLoader (train_dataset, batch_size = self.training_batch_size, shuffle = False)
        
        
        # @var val_loader
        val_loader = torch.utils.data.DataLoader (val_dataset, batch_size = self.validation_batch_size, shuffle = False)
        
        
        # Set the AdamW optimizer
        optimizer = transformers.AdamW (model.parameters (), lr = 1e-2)
        
        
        # @var criterion Set the loss Criteria according to the task type
        criterion = self.get_criterion (num_classes)
        
        
        # @var true_labels List
        true_labels = sorted (df['label'].unique ())
        
        
        # Train and eval each epoch
        for epoch in range (1, self.epochs + 1):
            
            # @var pbar
            pbar = tqdm (enumerate (train_loader), desc = "training...", total = int (train_df.shape[0] / self.training_batch_size), position = epoch)
            
            
            # Train this epoch
            model.train ()
            
            
            # @var y_true List
            y_true = []
            
            
            # @var y_pred List
            y_pred = []
            
            
            # Get all batches in this epoch
            for i, (lf, input_ids, attention_mask, labels) in pbar:
            
                # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates 
                # the gradients on subsequent backward passes. This is convenient while training RNNs. 
                # So, the default action is to accumulate (i.e. sum) the gradients on every loss.backward() call.
                # @link https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
                optimizer.zero_grad ()
                
                
                # @var input_ids 
                input_ids = input_ids.to (self.device)
                
                
                # @var attention_mask
                attention_mask = attention_mask.to (self.device)
                
                
                # @var labels
                labels = labels.to (self.device)
                
                
                # @var lf
                lf = lf.float ().to (self.device)
        
        
                # @var predictions Forward model
                predictions = model (lf, input_ids, attention_mask = attention_mask)
                
                
                # @var loss Get loss
                loss = criterion (torch.squeeze (predictions), labels.long () if num_classes > 1 else labels)

                
                # Store loss
                pbar.set_description (str (round (loss.item (), 4)))
                
                
                # Do deep learning stuff
                loss.backward ()
                optimizer.step ()
                
                
                # Use max to get the correct class
                # _, preds = torch.max (predictions, dim = 1)
                y_pred.extend (torch.round (predictions))
                
                
                # Store labels
                y_true.extend (labels.tolist ())
            
            
            # @var y_pred
            if num_classes == 1:
                y_pred = [prediction.tolist () for prediction in y_pred]
                y_pred = [true_labels[int (prediction[0])] for prediction in y_pred]
            else:
                y_pred = [np.argmax (F.softmax (prediction).detach ().numpy ()).item () for prediction in y_pred]
                y_pred = [true_labels[prediction] for prediction in y_pred]
            
            # @var y_pred
            y_true = [true_labels[int (label)] for label in y_true]
            
            
            
            # Classification report
            report = classification_report (y_true = y_true, y_pred = y_pred, output_dict = True)
            
            
            print ("training")
            print (json.dumps (report, indent = 4))


            # Eval this epoch with the test
            model.eval ()
            
            
            # @var pbar
            pbar = tqdm (enumerate (val_loader), desc = 'validation...', total = int (val_df.shape[0] / self.validation_batch_size), position = epoch)
            
            
            # @var y_true List
            y_true = []
            
            
            # @var y_pred List
            y_pred = []
            
            
            # No gradient is needed
            with torch.no_grad ():
            
                for i, (lf, input_ids, attention_mask, labels) in pbar:
                
                    # Move features to device
                    input_ids = input_ids.to (self.device)
                    attention_mask = attention_mask.to (self.device)
                    labels = labels.to (self.device)
                    lf = lf.float ().to (self.device)
                    
                    
                    # Forward model
                    predictions = model (lf, input_ids, attention_mask = attention_mask)
                    
                    
                    # Use max to get the correct class
                    y_pred.extend (torch.round (predictions))
                    
                    
                    # Store labels
                    y_true.extend (labels.tolist ())
                    

            # @var y_pred
            if num_classes == 1:
                y_pred = [prediction.tolist () for prediction in y_pred]
                y_pred = [true_labels[int (prediction[0])] for prediction in y_pred]
            else:
                y_pred = [np.argmax (F.softmax (prediction).detach ().numpy ()).item () for prediction in y_pred]
                y_pred = [true_labels[prediction] for prediction in y_pred]
            
            
            # @var y_pred
            y_true = [true_labels[int (label)] for label in y_true]
            
            # Classification report
            report = classification_report (y_true = y_true, y_pred = y_pred, output_dict = True)


            print ("validation")
            print (json.dumps (report, indent = 4))
                    
                    
        # Save model
        torch.save (model.state_dict (), self.get_model_filename ())
        
        
    def predict (self, using_official_test = False, callback = None):
        """
        @inherit
        """
        
        # @var df DataFrame
        df = self.dataset.get ()
        
        
        # @todo @aalmodobar
        df = df[df['label'].notna ()]
        
        
        # @var tokenizer Load the tokenizer
        # @todo Get pretrained tokenizer
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained ('dccuchile/bert-base-spanish-wwm-uncased')
        
        
        # @var x Dict of features for the linguistic features
        x = {
            'lf': pd.DataFrame (self.features['lf'].transform (df))
        }
        
        
        # If the supplied dataset contains information of the split, that means that we are dealing
        # only with a subset of the features and we retrieve it accordingly
        if using_official_test:
            x = {'input_' + key: item[item.index.isin (df.index)].reindex (df.index) for key, item in x.items ()}


        # @var input_size int 
        input_size = x['input_lf'].shape[1]
        
        
        # @var pretrained_model String
        pretrained_model = self.dataset.get_working_dir (self.dataset.task, 'models', 'bert', 'bert-finetunning')
        
        
        # @var num_classes
        num_classes = self.dataset.get_num_labels ()
        
        
        # Adjust the number of classes of the model for binary classification tasks
        num_classes = num_classes if num_classes > 2 else 1
        
        
        # @var model CustomBERTModel model
        model = CustomBERTModel (
            pretrained_model = pretrained_model, 
            input_size = input_size, 
            num_classes = num_classes
        )
        model.to (self.device)
    
        
        # Load model pretrained weights
        model.load_state_dict (torch.load (self.get_model_filename ()))

    
    
        # @var dataset Dataset Encode datasets to work with transformers
        dataset = self.createDatasetFromPandas (df, x['input_lf'])
        
        
        # @var loader
        loader = torch.utils.data.DataLoader (dataset, batch_size = self.validation_batch_size, shuffle = False)
        
        
        # @var pbar
        pbar = tqdm (enumerate (loader), desc = 'testing...', total = int (df.shape[0] / self.validation_batch_size))
        
        
        # Evaluation mode
        model.eval ()
            
        
        # @var y_true List
        y_true = []
        
        
        # @var y_pred List
        y_pred = []
        
        
        # @var true_labels List
        true_labels = sorted (df['label'].unique ())
        
        
        # No gradient is needed
        with torch.no_grad ():
            
            for i, (lf, input_ids, attention_mask, labels) in pbar:
            
                # Move features to device
                input_ids = input_ids.to (self.device)
                attention_mask = attention_mask.to (self.device)
                labels = labels.to (self.device)
                lf = lf.float ().to (self.device)
                
                
                # Forward model
                predictions = model (lf, input_ids, attention_mask = attention_mask)
                
                
                # Use max to get the correct class
                # _, preds = torch.max (predictions, dim = 1)
                y_pred.extend (torch.round (predictions))
                
                
                # Store labels
                y_true.extend (labels.tolist ())
                    
                    
        # @var y_pred
        if num_classes == 1:
            y_pred = [prediction.tolist () for prediction in y_pred]
            y_pred = [true_labels[int (prediction[0])] for prediction in y_pred]
        else:
            y_pred = [np.argmax (F.softmax (prediction).detach ().numpy ()).item () for prediction in y_pred]
            y_pred = [true_labels[prediction] for prediction in y_pred]
        
        
        # @var y_pred
        y_true = [true_labels[int (label)] for label in y_true]
            
        
        
        # @testing. Grouping
        y_true = df.groupby (['user'])['label'].apply (pd.Series.mode).to_list ()
        
        _df = df.assign (y_pred = y_pred)
        y_pred = _df.groupby (['user'])['y_pred'].agg (lambda x: pd.Series.mode (x)[0]).to_list ()
        
        
        
        # Classification report
        report = classification_report (y_true = y_true, y_pred = y_pred, output_dict = True)


        print ("testing")
        print (json.dumps (report, indent = 4))