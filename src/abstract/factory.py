from abc import ABC, abstractmethod
from .preprocessor import AbstractPreprocessor
from .model import AbstractModel

class AbstractFactory(ABC):

    @abstractmethod
    def create_preprocessor(self) -> AbstractPreprocessor:
        pass
    
    @abstractmethod
    def create_model(self, num_classes: int) -> AbstractModel:
        pass