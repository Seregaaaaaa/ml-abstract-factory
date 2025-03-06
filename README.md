# Machine Learning Abstract Factory

ML-Abstract-Factory — это библиотека, реализующая паттерн проектирования "Абстрактная фабрика" для создания компонентов конвейера машинного обучения. Она позволяет гибко переключаться между различными подходами к обработке текста и классификации, сохраняя единый интерфейс взаимодействия.

## Project Structure
```
ml-abstract-factory/
├── src/                     # Исходный код проекта
│   ├── abstract/            # Абстрактные базовые классы
│   │   ├── factory.py       # Абстрактная фабрика
│   │   ├── model.py         # Абстрактная модель
│   │   └── preprocessor.py  # Абстрактный препроцессор
│   │
│   ├── factories/           # Конкретные реализации фабрик
│   │   ├── bert_factory.py  # Фабрика для BERT-моделей
│   │   └── traditional_factory.py  # Фабрика для классических ML-моделей
│   │
│   ├── models/              # Реализации моделей
│   │   ├── bert_model.py    # Модель на основе BERT
│   │   └── traditional_model.py  # Классические алгоритмы ML
│   │
│   └── preprocessors/       # Реализации препроцессоров
│       ├── bert_preprocessor.py  # Токенизатор для BERT
│       └── traditional_preprocessor.py  # Классический препроцессор (TF-IDF и LE)
│
├── examples/                # Примеры использования
│   ├── bert_example.py      # Пример с BERT
│   └── traditional_example.py  # Пример с LogReg и TF-IDF
│
└── README.md                # Основная документация
```
## Description

ML-Abstract-Factory реализует паттерн "Абстрактная фабрика" для создания масштабируемых и гибких конвейеров обработки текста и классификации. Этот проект решает проблему интеграции различных подходов к обработке естественного языка (от традиционных методов до современных трансформеров) в единый фреймворк.

### Основные преимущества подхода:

1. **Инкапсуляция**: Скрывает сложность создания и настройки компонентов NLP конвейера
2. **Взаимозаменяемость**: Позволяет легко переключаться между разными технологиями (например, TF-IDF + LogReg или BERT)
3. **Расширяемость**: Простое добавление новых фабрик для поддержки других моделей и методов предобработки
4. **Согласованность**: Гарантирует совместимость между моделями и их препроцессорами
5. **Стандартизированный интерфейс**: Унифицированный API для работы с разными типами моделей

### Применение:

- **Исследования**: Быстрое сравнение различных подходов к классификации текста
- **Продакшн**: Упрощение интеграции разных моделей в производственные системы
- **Обучение**: Демонстрация паттернов проектирования в контексте машинного обучения

Библиотека разработана с учетом современных практик объектно-ориентированного программирования и предоставляет элегантное решение для организации кода в проектах обработки естественного языка.

### Диаграмма классов
![Image](class_diagram.drawio.png)