from src.abstract.preprocessor import AbstractPreprocessor
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

class TraditionalPreprocessor(AbstractPreprocessor):
    def __init__(self,ngram_range, max_features):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        self.label_encoder = LabelEncoder()

    def fit(self, texts, labels):
        self.vectorizer.fit(texts)
        self.label_encoder.fit(labels)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts, labels):
        self.fit(texts, labels)
        return self.vectorizer.transform(texts)
'''
    def inverse_transform(self, encoded_labels):
        return self.label_encoder.inverse_transform(encoded_labels)'''