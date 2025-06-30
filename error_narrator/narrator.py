import os
import logging
import asyncio
from gradio_client import Client as GradioClient
from openai import OpenAI, AsyncOpenAI
from rich.console import Console
from rich.markdown import Markdown

# Настраиваем базовый логгер для библиотеки
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Понижаем уровень логирования для httpx и gradio_client, чтобы убрать лишний шум
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("gradio_client").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class NarratorException(Exception):
    """Базовое исключение для библиотеки ErrorNarrator."""
    pass

class ApiKeyNotFoundError(NarratorException):
    """Вызывается, когда API-ключ не найден."""
    pass

class ErrorNarrator:
    """
    Класс для получения объяснений ошибок с помощью AI.
    Поддерживает несколько провайдеров: 'gradio' (по умолчанию, бесплатно) и 'openai'.
    """
    DEFAULT_PROMPT_TEMPLATE = (
        "You are an expert Python developer's assistant. An internal application error occurred. "
        "Your task is to provide a comprehensive analysis of the traceback for the developer in Russian. "
        "Your response must be structured in Markdown and include these sections:\n\n"
        "### 🎯 Причина ошибки\n"
        "A clear, concise explanation of the error's root cause.\n\n"
        "### 📍 Место ошибки\n"
        "The exact file and line number, with a code snippet showing the context (the error line and a few lines around it).\n\n"
        "### 🛠️ Предлагаемое исправление\n"
        "A clear, actionable suggestion for fixing the issue. Provide a code snippet using a diff format (lines with `-` for removal, `+` for addition) to illustrate the change.\n\n"
        "### 🎓 Почему это происходит (Обучающий момент)\n"
        "A brief explanation of the underlying concept that caused the error, to help the developer avoid similar mistakes in the future.\n\n"
        "Here is the technical traceback:\n"
        "```\n{traceback}\n```\n\n"
        "Provide a structured analysis for the developer's logs. Do not address the user. Do not ask for more code or provide any disclaimers."
    )
    DEFAULT_API_KEY_ERROR = (
        "API-ключ не найден для выбранного провайдера.\n"
        "Для 'gradio': может потребоваться для доступа к приватным Space (переменная HUGGINGFACE_API_KEY).\n"
        "Для 'openai': ключ обязателен (переменная OPENAI_API_KEY)."
    )
    GRADIO_MODEL_ID = "hysts/mistral-7b"
    OPENAI_MODEL_ID = "gpt-3.5-turbo"

    def __init__(self, provider: str = 'gradio', api_key: str = None, model_id: str = None, prompt_template: str = None, **kwargs):
        """
        Инициализирует ErrorNarrator.

        :param provider: Провайдер для получения объяснений ('gradio' или 'openai').
        :param api_key: API-ключ. Если не указан, будет взят из переменных окружения
                        (HUGGINGFACE_API_KEY для 'gradio', OPENAI_API_KEY для 'openai').
        :param model_id: Идентификатор модели. Если не указан, используется значение по умолчанию для провайдера.
        :param prompt_template: Шаблон промпта для модели.
        :param kwargs: Дополнительные параметры для модели (например, temperature, max_new_tokens).
        """
        self.provider = provider
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.model_params = kwargs
        self.cache = {} # Инициализируем кеш

        if self.provider == 'gradio':
            self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
            self.model_id = model_id or self.GRADIO_MODEL_ID
            self.client = GradioClient(self.model_id, hf_token=self.api_key)
            # Устанавливаем параметры по умолчанию для Gradio, если они не переданы
            self.model_params.setdefault('temperature', 0.6)
            self.model_params.setdefault('max_new_tokens', 1024)
            self.model_params.setdefault('top_p', 0.9)
            self.model_params.setdefault('top_k', 50)
            self.model_params.setdefault('repetition_penalty', 1.2)

        elif self.provider == 'openai':
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ApiKeyNotFoundError(self.DEFAULT_API_KEY_ERROR)
            self.model_id = model_id or self.OPENAI_MODEL_ID
            self.client = OpenAI(api_key=self.api_key)
            self.async_client = AsyncOpenAI(api_key=self.api_key)
            # Устанавливаем параметры по умолчанию для OpenAI, если они не переданы
            self.model_params.setdefault('temperature', 0.7)
            self.model_params.setdefault('max_tokens', 1024) # OpenAI использует 'max_tokens'
        else:
            raise ValueError("Неизвестный провайдер. Доступные варианты: 'gradio', 'openai'")

    def _build_prompt(self, traceback: str) -> str:
        """Формирует промпт для модели."""
        return self.prompt_template.format(traceback=traceback)

    # --- Методы для провайдера Gradio ---

    def _predict_gradio(self, prompt: str) -> str:
        logger.info(f"Запрашиваю объяснение через Gradio (модель: {self.model_id})...")
        try:
            result = self.client.predict(
                prompt,
                self.model_params.get('max_new_tokens'),
                self.model_params.get('temperature'),
                self.model_params.get('top_p'),
                self.model_params.get('top_k'),
                self.model_params.get('repetition_penalty'),
                api_name="/chat"
            )
            return result.strip()
        except Exception as e:
            logger.error(f"Ошибка при запросе к Gradio: {e}")
            return f"К сожалению, не удалось получить объяснение от AI. (Ошибка Gradio: {e})"

    async def _predict_async_gradio(self, prompt: str) -> str:
        logger.info(f"Асинхронно запрашиваю объяснение через Gradio (модель: {self.model_id})...")
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, self._predict_gradio, prompt)
            return result
        except Exception as e:
            logger.error(f"Ошибка при асинхронном запросе к Gradio: {e}")
            return f"К сожалению, не удалось получить асинхронное объяснение от AI. (Ошибка Gradio: {e})"
            
    # --- Методы для провайдера OpenAI ---

    def _predict_openai(self, prompt: str) -> str:
        logger.info(f"Запрашиваю объяснение через OpenAI (модель: {self.model_id})...")
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                **self.model_params
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Ошибка при запросе к OpenAI: {e}")
            return f"К сожалению, не удалось получить объяснение от AI. (Ошибка OpenAI: {e})"

    async def _predict_async_openai(self, prompt: str) -> str:
        logger.info(f"Асинхронно запрашиваю объяснение через OpenAI (модель: {self.model_id})...")
        try:
            completion = await self.async_client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                **self.model_params
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Ошибка при асинхронном запросе к OpenAI: {e}")
            return f"К сожалению, не удалось получить асинхронное объяснение от AI. (Ошибка OpenAI: {e})"

    # --- Диспетчеры предсказаний ---

    def _predict(self, prompt: str) -> str:
        if self.provider == 'gradio':
            return self._predict_gradio(prompt)
        elif self.provider == 'openai':
            return self._predict_openai(prompt)
        # Этот return никогда не должен сработать из-за проверки в __init__
        return "Ошибка: неверный провайдер."

    async def _predict_async(self, prompt: str) -> str:
        if self.provider == 'gradio':
            return await self._predict_async_gradio(prompt)
        elif self.provider == 'openai':
            return await self._predict_async_openai(prompt)
        return "Ошибка: неверный провайдер."

    def explain_error(self, traceback: str) -> str:
        """
        Получает объяснение для ошибки (traceback) с помощью AI.
        Проверяет кеш перед отправкой запроса.
        """
        if traceback in self.cache:
            logger.info("Объяснение найдено в кеше.")
            return self.cache[traceback]

        prompt = self._build_prompt(traceback)
        explanation = self._predict(prompt)
        self.cache[traceback] = explanation # Сохраняем результат в кеш
        return explanation

    async def explain_error_async(self, traceback: str) -> str:
        """
        Асинхронно получает объяснение для ошибки (traceback) с помощью AI.
        Проверяет кеш перед отправкой запроса.
        """
        if traceback in self.cache:
            logger.info("Объяснение найдено в кеше.")
            return self.cache[traceback]

        prompt = self._build_prompt(traceback)
        explanation = await self._predict_async(prompt)
        self.cache[traceback] = explanation # Сохраняем результат в кеш
        return explanation

    def explain_and_print(self, traceback: str):
        """
        Получает объяснение, форматирует его с помощью rich и выводит в консоль.
        """
        console = Console()
        with console.status("[bold green]Анализирую ошибку с помощью AI...[/]", spinner="dots"):
            explanation_md = self.explain_error(traceback)
        
        console.print(Markdown(explanation_md, style="default"))

    async def explain_and_print_async(self, traceback: str):
        """
        Асинхронно получает объяснение, форматирует и выводит в консоль.
        """
        console = Console()
        with console.status("[bold green]Анализирую ошибку с помощью AI...[/]", spinner="dots"):
            explanation_md = await self.explain_error_async(traceback)
        
        console.print(Markdown(explanation_md, style="default"))