from abc import ABC, abstractmethod

class AbstractPreprocessor(ABC):
    """
    Абстрактный класс для предобработки данных
    """
    
    @abstractmethod
    def transform(self, texts):
        """Преобразование текстов в векторы или последовательности токенов"""
        pass
    
    @abstractmethod
    def fit_transform(self, texts):
        """Обучение препроцессора и преобразование текстов"""
        pass