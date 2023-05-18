class Base_Oneshot_Model:
    def __init__(self, n_components, n_class, data):
        assert data in ['Omniglot', 'EMNIST'], 'data not found'
        self.data = data
        self.n_components = n_components
        self.n_class = n_class

    def train_background(self):
        raise NotImplementedError
    
    def train_oneshot(self):
        raise NotImplementedError
    
    def clear_oneshot(self):
        raise NotImplementedError
    
    def save_background(self, dir):
        raise NotImplementedError
    
    def load_background(self, dir):
        raise NotImplementedError
    
    def encode(self, x):
        raise NotImplementedError
    
    def distance(self, x1, x2):
        raise NotImplementedError
    
    def predict(self, x):
        raise NotImplementedError