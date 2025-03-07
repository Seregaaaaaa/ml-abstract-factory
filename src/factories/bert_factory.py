from ..abstract.factory import AbstractFactory
from ..models.bert_model import BERTModel
from ..preprocessors.bert_preprocessor import BertPreprocessor

class BertFactory(AbstractFactory):

    
    def create_preprocessor(self):
        return BertPreprocessor()
    
    def create_model(self, num_classes, optimizer, loss, metrics):
        return BERTModel(num_classes, optimizer, loss, metrics)