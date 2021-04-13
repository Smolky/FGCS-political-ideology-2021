import itertools


class BaseModel ():
    """
    BaseModel
    
    @todo Separate models and merge at the end to have better sampling of the classifiers
    @todo Create ensemble model from different classifiers
    @todo Allow to make predictions easly for the winning models
    """
    
    # @var all_available_features List
    all_available_features = []
    
    
    # @var features Here we store the transformers we will use in our work
    features = {}
    
    
    # @var is_merged boolean
    is_merged = False
    
    
    def is_merged (self, is_merged):
        self.is_merged = is_merged
        
    
    def get_available_features (self):
        """
        get_available_features
        
        @return List
        """
        return self.all_available_features
    
    
    def set_dataset (self, dataset):
        """
        set_dataset
        
        @param dataset Dataset
        """
        self.dataset = dataset
    
    
    def set_features (self, key, features):
        """
        set_features
        
        @param key string
        @features Transformer
        """
        self.features[key] = features;
        
    
    def get_feature_combinations (self):
        """
        List Get all the keys of the feature sets we are going to use
        Expand it to have features in isolation and combination (lf), (lf, se), ...
        
        @return List
        """
        
        # @var feature_combinations 
        feature_combinations = [key for key, _void in self.features.items ()]
        feature_combinations = [list (subset) \
                                    for L in range (1, len (feature_combinations) + 1) \
                                        for subset in itertools.combinations (feature_combinations, L)]

        return feature_combinations

    @property
    def train (self):
        raise NotImplementedError ("Subclasses should implement this!")    
        
        
    @property
    def predict (self):
        raise NotImplementedError ("Subclasses should implement this!")    