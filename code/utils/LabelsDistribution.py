"""
    Gets the label distributionpython
    
    @see config.py
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import config
import sys


class LabelsDistribution ():

    def __init__ (self, df):
        self.df = df

    def get_labels (self, normalize = False):
        """Returns the labels"""
        return self.df['label'].value_counts (normalize = normalize)

    def get_latex (self, normalize = False):
        """Outpus the result in Latex format"""
        return self.df['label'].value_counts (normalize = normalize).rename_axis ('labels').reset_index (name = '# of tweets').to_latex (index = False)

