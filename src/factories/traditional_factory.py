from ..abstract.factory import AbstractFactory
from ..models.traditional_model import TraditionalModel
from ..preprocessors.traditional_preprocessor import TraditionalPreprocessor

class TraditionalFactory(AbstractFactory):
    """
    Конкретная фабрика для создания традиционной модели и соответствующего препроцессора
    """
    
    def create_preprocessor(self,ngram_range, max_features):
        return TraditionalPreprocessor(ngram_range, max_features)
    
    def create_model(self):
        return TraditionalModel()