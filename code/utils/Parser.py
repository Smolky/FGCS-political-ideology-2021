import argparse

class DefaultParser (argparse.ArgumentParser):
    """It is a argument parser with some default options """

    def __init__ (self, description = 'No description provided'):
        
        # Delegate on parent constructor
        super ().__init__ (description)
        

        # Parser
        self.add_argument ('--dataset', dest = 'dataset')
        self.add_argument ('--corpus', dest = 'corpus', default = '', help = 'To filter by one corpus of the dataset. Default, all')
        self.add_argument ('--task', dest = 'task', default = '', help = 'Get the task, if any, for this corpus')
        self.add_argument ('--force', dest='force', default = False, help = 'If True, it forces to replace existing files')
    