from abc import ABC, abstractmethod
from .preprocessor import AbstractPreprocessor
from .model import AbstractModel

class AbstractFactory(ABC):
    """
    Абстрактная фабрика, которая определяет методы для создания 
    препроцессора и модели
    """
    @abstractmethod
    def create_preprocessor(self) -> AbstractPreprocessor:
        """Создает препроцессор"""
        pass
    
    @abstractmethod
    def create_model(self, num_classes: int) -> AbstractModel:
        """Создает модель"""
        pass