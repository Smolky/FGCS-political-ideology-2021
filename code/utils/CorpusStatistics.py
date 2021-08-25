"""
    Line length distribution
    
    @see config.py
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import pandas as pd

class CorpusStatistics ():

    def __init__ (self, dataset):
        """
        
        @param dataset Dataset
        """
        self.dataset = dataset

    def get_line_length_distribution (self, split = 'all', field = 'tweet'):
        """ 
        Obtains the distribution of the line-length 
        
        @param split string
        @param field string
        """
        # Get datasets splits for training, validation and testing
        df = self.dataset.get ()
        df = self.dataset.get_split (df, split)
        df = self.dataset.preprocess (df)
        return df[field].str.len ().describe ()
    
    
    def get_duplicated_labels_in_different_splits (self, label = 'user'):
        """
        Determines if the same label appears in different splits of the corpus. 
        
        This is useful to assert that splits does not contain repeated data
        in author profiling tasks
        
        @param label
        
        @return set
        """
        
        # @var df DataFrame
        df = self.dataset.get ()
        
        
        # No label
        if label not in df.columns:
            return {};
        
        
        # @var users List
        users = [set (self.dataset.get_split (df, split)[label].unique ()) for split in ['train', 'val', 'test']]

        return set.intersection (*users)
        
    
    def get_columns_distribution_in_different_splits (self):
        """
        get_columns_distribution_in_different_splits
        """
        
        # @var df DataFrame
        df = self.dataset.get ()
        
        
        # @var categorical_columns
        categorical_columns = list (set (df.columns) - set (df._get_numeric_data ().columns))
        
        
        # @var stats Dictionary
        stats = {field: {split: list (df[field].value_counts ()) for split in ['all', 'train', 'val', 'test']} for field in categorical_columns}
        
        
        # @var df_resume DataFrame
        return pd.DataFrame.from_dict (stats, orient = 'columns')
    
