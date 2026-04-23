## Описание
Проект по анализу и предсказанию стоимости жилья в Калифорнии.

## Структура
- data_analysis.py – EDA и визуализация
- model_training.py – обучение модели
- utils.py – вспомогательные функции

## Pipeline
1. Анализ данных
2. Feature engineering
3. Проверка мультиколлинеарности (VIF)
4. Обучение Linear Regression
5. Оценка модели (R², RMSE)
6. Log-transform сравнение

## Запуск
```bash
python data_analysis.py
python model_training.py