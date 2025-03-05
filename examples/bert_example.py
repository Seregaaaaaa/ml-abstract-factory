import sys
import os
from src.factories.bert_factory import BertFactory
import tensorflow as tf
import numpy as np

texts = [
    "Пример текста для классификации",
    "Еще один текст для обработки",
]

labels = np.array([0, 1])  
batch_size = 2

factory = BertFactory()

preprocessor = factory.create_preprocessor()
model = factory.create_model(num_classes=2,
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

input_data = preprocessor.transform(texts)

features_dataset = tf.data.Dataset.from_tensor_slices(dict(input_data))
labels_dataset = tf.data.Dataset.from_tensor_slices(labels)

dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
dataset = dataset.batch(batch_size)

model.fit(dataset, epochs=2)

predictions = model.predict(preprocessor.transform(["Новый текст для предсказания"]))
print(f"Предсказания: {predictions}")