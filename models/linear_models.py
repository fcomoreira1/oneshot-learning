import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from base_model import Base_Oneshot_Model


class PCA_Model(Base_Oneshot_Model):
    '''
    Wrapper class around a linear PCA decomposition from sklearn.
    '''
    def __init__(self, n_components, n_class, data='Omniglot'):
        super().__init__(n_components, n_class, data)
        self.pca = PCA(n_components=n_components)

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
        return self.pca.transform(x)
    
    def distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    def predict(self, x):
        raise NotImplementedError
    

class LDA_Model(Base_Oneshot_Model):
    '''
    Wrapper class around a linear LDA.
    The block learn the most important components with respect to
    the data and the labels.
    '''
    def __init__(self, n_components, n_class, data='Omniglot'):
        super().__init__(n_components, n_class, data)
        self.lda = LDA(n_components=n_components)

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
        return self.lda.transform(x)
    
    def distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    def predict(self, x):
        raise NotImplementedError
    

class QDA_Model(Base_Oneshot_Model):
    '''
    Wrapper class around a QDA (not really linear, but there you go).
    The block learn the most important components with respect to
    the data and the labels.
    '''
    def __init__(self, n_components, n_class, data='Omniglot'):
        super().__init__(n_components, n_class, data)
        self.qda = QDA(n_components=n_components)

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
        return self.qda.transform(x)
    
    def distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    def predict(self, x):
        raise NotImplementedError