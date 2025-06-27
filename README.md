# Error Narrator

[![PyPI version](https://badge.fury.io/py/error-narrator.svg)](https://badge.fury.io/py/error-narrator)

Простая Python-библиотека для получения "человеческих" объяснений ошибок и исключений с помощью моделей ИИ от Hugging Face.

Вместо того чтобы просто видеть `traceback`, вы можете получить структурированный анализ, который поможет быстрее понять и исправить проблему.

## Основные возможности

-   Превращает непонятные `traceback`'и в понятные объяснения.
-   Асинхронная работа "из коробки" (`asyncio`).
-   Гибкая настройка: можно передать свой API-ключ Hugging Face и выбрать любую модель.
-   Безопасное управление API-ключом через переменные окружения.

## Установка

```bash
pip install error-narrator
```

## Как пользоваться

### 1. Настройка API-ключа

Библиотека требует API-ключ от Hugging Face. Есть два способа его задать:

**Способ А (рекомендуемый): Переменная окружения**

Установите переменную окружения `HUGGINGFACE_API_KEY`. Библиотека автоматически ее найдет.

```bash
# Для Linux/macOS
export HUGGINGFACE_API_KEY='hf_xxxxxxxxxxxxxxxxxxxx'

# Для Windows (PowerShell)
$env:HUGGINGFACE_API_KEY='hf_xxxxxxxxxxxxxxxxxxxx'
```

**Способ Б: Передача в конструктор**

Вы можете передать ключ напрямую при создании объекта `ErrorNarrator`.

```python
narrator = ErrorNarrator(api_key="hf_xxxxxxxxxxxxxxxxxxxx")
```

### 2. Пример кода

Вот простой пример, как перехватить исключение и получить для него объяснение.

```python
import asyncio
import traceback
from error_narrator import ErrorNarrator

# Инициализируем класс.
# Ключ будет взят из переменной окружения HUGGINGFACE_API_KEY.
narrator = ErrorNarrator()

# Или можно указать модель
# narrator = ErrorNarrator(model_id="mistralai/Mistral-7B-Instruct-v0.2")

async def main():
    try:
        # Симулируем ошибку
        result = 1 / 0
    except Exception:
        # Получаем полный traceback ошибки
        error_traceback = traceback.format_exc()
        
        # Получаем объяснение от AI
        explanation = await narrator.explain_error(error_traceback)
        
        print("--- ОБЫЧНАЯ ОШИБКА ---")
        print(error_traceback)
        print("\n--- ОБЪЯСНЕНИЕ ОТ AI ---")
        print(explanation)

if __name__ == "__main__":
    asyncio.run(main())
```

### Результат выполнения

```
--- ОБЫЧНАЯ ОШИБКА ---
Traceback (most recent call last):
  File "example.py", line 19, in main
    result = 1 / 0
ZeroDivisionError: division by zero

--- ОБЪЯСНЕНИЕ ОТ AI ---
**1. Причина ошибки:**
Произошла ошибка `ZeroDivisionError`. Это стандартное исключение в Python, которое возникает при попытке деления любого числа на ноль.

**2. Местоположение:**
Ошибка произошла в файле `example.py` в строке 19, внутри функции `main`: `result = 1 / 0`.

**3. Рекомендация по исправлению:**
Перед выполнением операции деления необходимо добавить проверку, чтобы убедиться, что делитель не равен нулю.

Пример исправления:
```python
divisor = 0
if divisor != 0:
    result = 1 / divisor
else:
    # Обработать случай, когда делитель равен нулю
    print("Ошибка: Деление на ноль невозможно.")
    result = None
```
Эта проверка предотвратит падение программы и позволит корректно обработать ситуацию.
```

## Лицензия

Этот проект распространяется под лицензией MIT. 