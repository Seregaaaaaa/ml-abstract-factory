from src.abstract.preprocessor import AbstractPreprocessor
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder

class BertPreprocessor(AbstractPreprocessor):
    def __init__(self, max_length=128, model_name="DeepPavlov/rubert-base-cased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.label_encoder = LabelEncoder()

    def fit(self, texts, labels):
        self.label_encoder.fit(labels)
        return self

    def transform(self, texts):
        encoded_inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="tf"
        )
        return encoded_inputs

    def fit_transform(self, texts, labels):
        self.fit(texts, labels)
        return self.transform(texts)

    def inverse_transform(self, encoded_labels):
        return self.label_encoder.inverse_transform(encoded_labels)

    def encode_labels(self, labels):
        return self.label_encoder.transform(labels)