from src.factories.bert_factory import BertFactory

def main():
    # Создание экземпляра фабрики BERT
    factory = BertFactory()

    # Создание препроцессора
    preprocessor = factory.create_preprocessor()
    print("Созданный препроцессор:", preprocessor)

    # Создание модели
    model = factory.create_model()
    print("Созданная модель:", model)

if __name__ == "__main__":
    main()