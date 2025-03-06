from transformers import TFBertForSequenceClassification
from src.abstract.model import AbstractModel
import numpy as np

class BERTModel(AbstractModel):
    def __init__(self, num_classes, optimizer, loss, metrics):
        self.model = TFBertForSequenceClassification.from_pretrained("DeepPavlov/rubert-base-cased", num_labels=num_classes, from_pt=True)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, train_dataset, validation_data=None, epochs=1, callbacks=None):
        return self.model.fit(train_dataset, validation_data=validation_data, epochs=epochs, callbacks=callbacks)

    def predict(self, test_dataset):
        predictions = self.model.predict(test_dataset)
        logits = predictions.logits if hasattr(predictions, 'logits') else predictions[0]
        return np.argmax(logits, axis=-1)