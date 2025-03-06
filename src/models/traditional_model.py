from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.abstract.model import AbstractModel


class TraditionalModel(AbstractModel):
    def __init__(self):
        self.model = LogisticRegression(random_state=42, multi_class='multinomial')
    
    def fit(self, train_data, validation_data=None, epochs=1, callbacks=None):
        X_train, y_train = train_data
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, test_data):
        return self.model.predict(test_data)