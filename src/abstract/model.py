from abc import ABC, abstractmethod

class AbstractModel(ABC):
    
    @abstractmethod
    def fit(self, train_data, validation_data=None, epochs=1, callbacks=None):
        pass
    
    @abstractmethod
    def predict(self, test_data):
        pass