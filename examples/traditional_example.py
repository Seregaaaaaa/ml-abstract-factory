from src.factories.traditional_factory import TraditionalFactory
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

train_df = pd.read_csv('/Users/sergei.samoilov/Documents/учеба/Учеба_3.2/ООАП/train_texts.csv', index_col=0)
test_df = pd.read_csv('/Users/sergei.samoilov/Documents/учеба/Учеба_3.2/ООАП/test_texts.csv', index_col=0)
num_classes = len(train_df['author'].unique())

X_train, X_val, y_train, y_val = train_test_split(train_df['text'], train_df['author'], test_size=0.2, stratify=train_df['author'], random_state=42)

factory = TraditionalFactory()
preprocessor = factory.create_preprocessor(ngram_range=(1, 2), max_features=50000)
model = factory.create_model()

X_train_vec = preprocessor.fit_transform(X_train, y_train)

model.fit((X_train_vec, y_train))

X_val_vec = preprocessor.transform(X_val)
val_predictions = model.predict(X_val_vec)

print(classification_report(y_val,val_predictions))
print(y_val[:10],val_predictions[:10])