import argparse

class DefaultParser (argparse.ArgumentParser):
    """It is a argument parser with some default options """

    def __init__ (self, description = 'No description provided', defaults = {}):
        
        # Delegate on parent constructor
        super ().__init__ (description)
        
        
        # @var default_dataset String
        default_dataset = defaults['dataset'] if 'dataset' in defaults else None
        
        
        # @var default_corpus String
        default_corpus = defaults['corpus'] if 'corpus' in defaults else ''
        
        

        # Parser
        self.add_argument ('--dataset', dest = 'dataset', default = default_dataset)
        self.add_argument ('--corpus', dest = 'corpus', default = default_corpus, help = 'To filter by one corpus of the dataset. Default, all')
        self.add_argument ('--task', dest = 'task', default = '', help = 'Get the task, if any, for this corpus')
        self.add_argument ('--force', dest='force', default = False, help = 'If True, it forces to replace existing files')
    