from abc import ABC, abstractmethod

class AbstractModel(ABC):
    """
    Абстрактный класс для моделей машинного обучения
    """
    
    @abstractmethod
    def fit(self, train_data, validation_data=None, epochs=1, callbacks=None):
        """Обучение модели"""
        pass
    
    @abstractmethod
    def predict(self, test_data):
        """Предсказание на основе модели"""
        pass