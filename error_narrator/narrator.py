import os
import logging
import asyncio
import random
import time
from .gradio_models import GRADIO_MODELS
from gradio_client import Client
from openai import OpenAI, AsyncOpenAI
from rich.console import Console
from rich.markdown import Markdown

# Configure a basic logger for the library
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Lower the logging level for httpx and gradio_client to reduce noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("gradio_client").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_PROMPT_TEMPLATES = {
    "ru": (
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
    ),
    "en": (
        "You are an expert Python developer's assistant. An internal application error occurred. "
        "Your task is to provide a comprehensive analysis of the traceback for the developer in English. "
        "Your response must be structured in Markdown and include these sections:\n\n"
        "### 🎯 Root Cause\n"
        "A clear, concise explanation of the error's root cause.\n\n"
        "### 📍 Error Location\n"
        "The exact file and line number, with a code snippet showing the context (the error line and a few lines around it).\n\n"
        "### 🛠️ Suggested Fix\n"
        "A clear, actionable suggestion for fixing the issue. Provide a code snippet using a diff format (lines with `-` for removal, `+` for addition) to illustrate the change.\n\n"
        "### 🎓 Why This Happens (A Learning Moment)\n"
        "A brief explanation of the underlying concept that caused the error, to help the developer avoid similar mistakes in the future.\n\n"
        "Here is the technical traceback:\n"
        "```\n{traceback}\n```\n\n"
        "Provide a structured analysis for the developer's logs. Do not address the user. Do not ask for more code or provide any disclaimers."
    )
}

_TRANSLATIONS = {
    "ru": {
        "api_key_error": (
            "API-ключ не найден для выбранного провайдера.\n"
            "Для 'gradio': может потребоваться для доступа к приватным Space (переменная HUGGINGFACE_API_KEY).\n"
            "Для 'openai': ключ обязателен (переменная OPENAI_API_KEY)."
        ),
        "unknown_provider": "Неизвестный провайдер. Доступные варианты: 'gradio', 'openai'",
        "unsupported_language": "Неподдерживаемый язык. Доступные варианты: 'en', 'ru'",
        "gradio_request": "Запрашиваю объяснение через Gradio (модель: {model_id})...",
        "gradio_request_async": "Асинхронно запрашиваю объяснение через Gradio (модель: {model_id})...",
        "gradio_error": "К сожалению, не удалось получить объяснение от AI. (Ошибка Gradio: {e})",
        "gradio_error_async": "К сожалению, не удалось получить асинхронное объяснение от AI. (Ошибка Gradio: {e})",
        "openai_request": "Запрашиваю объяснение через OpenAI (модель: {model_id})...",
        "openai_request_async": "Асинхронно запрашиваю объяснение через OpenAI (модель: {model_id})...",
        "openai_error": "К сожалению, не удалось получить объяснение от AI. (Ошибка OpenAI: {e})",
        "openai_error_async": "К сожалению, не удалось получить асинхронное объяснение от AI. (Ошибка OpenAI: {e})",
        "invalid_provider": "Ошибка: неверный провайдер.",
        "cache_hit": "Объяснение найдено в кеше.",
        "status_analysis": "[bold green]Анализирую ошибку с помощью AI...[/]",
        # Rich status messages
        "cache_hit_rich": "💾 Найдено в кеше. Загружаю...",
        "gradio_request_rich": "⚡️ Пробую модель Gradio: [bold cyan]{model_id}[/bold cyan]...",
        "gradio_fail_rich": "❌ Модель [bold red]{model_id}[/bold red] не ответила. Пробую следующую...",
        "openai_request_rich": "⚡️ Обращаюсь к OpenAI: [bold green]{model_id}[/bold green]..."
    },
    "en": {
        "api_key_error": (
            "API key not found for the selected provider.\n"
            "For 'gradio': may be required for private Space access (HUGGINGFACE_API_KEY environment variable).\n"
            "For 'openai': key is mandatory (OPENAI_API_KEY environment variable)."
        ),
        "unknown_provider": "Unknown provider. Available options: 'gradio', 'openai'",
        "unsupported_language": "Unsupported language. Available options: 'en', 'ru'",
        "gradio_request": "Requesting explanation via Gradio (model: {model_id})...",
        "gradio_request_async": "Asynchronously requesting explanation via Gradio (model: {model_id})...",
        "gradio_error": "Unfortunately, failed to get an explanation from the AI. (Gradio Error: {e})",
        "gradio_error_async": "Unfortunately, failed to get an asynchronous explanation from the AI. (Gradio Error: {e})",
        "openai_request": "Requesting explanation via OpenAI (model: {model_id})...",
        "openai_request_async": "Asynchronously requesting explanation via OpenAI (model: {model_id})...",
        "openai_error": "Unfortunately, failed to get an explanation from the AI. (OpenAI Error: {e})",
        "openai_error_async": "Unfortunately, failed to get an asynchronous explanation from the AI. (OpenAI Error: {e})",
        "invalid_provider": "Error: invalid provider.",
        "cache_hit": "Explanation found in cache.",
        "status_analysis": "[bold green]Analyzing the error with AI...[/]",
        # Rich status messages
        "cache_hit_rich": "💾 Found in cache. Loading...",
        "gradio_request_rich": "⚡️ Trying Gradio model: [bold cyan]{model_id}[/bold cyan]...",
        "gradio_fail_rich": "❌ Model [bold red]{model_id}[/bold red] failed. Trying next...",
        "openai_request_rich": "⚡️ Contacting OpenAI: [bold green]{model_id}[/bold green]..."
    }
}

class NarratorException(Exception):
    """Base exception for the ErrorNarrator library."""
    pass

class ApiKeyNotFoundError(NarratorException):
    """Raised when the API key is not found."""
    pass

class ErrorNarrator:
    """
    A class to get AI-powered explanations for errors.
    Supports multiple providers: 'gradio' (default, free) and 'openai'.
    """
    OPENAI_MODEL_ID = "gpt-3.5-turbo"

    def __init__(self, provider: str = 'gradio', language: str = 'en', api_key: str = None, model_id: str = None, models: list = None, prompt_template: str = None, **kwargs):
        """
        Initializes the ErrorNarrator.

        :param provider: The provider to use for explanations ('gradio' or 'openai').
        :param language: The language for the AI response ('en' or 'ru').
        :param api_key: The API key. If not provided, it will be sourced from environment variables
                        (HUGGINGFACE_API_KEY for 'gradio', OPENAI_API_KEY for 'openai').
        :param model_id: A convenience parameter to use a single, specific Gradio model, disabling rotation.
        :param models: A list of Gradio model identifiers to use for rotation. Overrides the default list.
        :param prompt_template: A custom prompt template for the model.
        :param kwargs: Additional parameters for the model (e.g., temperature, max_new_tokens).
        """
        if language not in _TRANSLATIONS:
            raise ValueError(_TRANSLATIONS['en']['unsupported_language'])
        self.language = language
        self.T = _TRANSLATIONS[self.language]

        self.provider = provider
        self.prompt_template = prompt_template or _PROMPT_TEMPLATES[self.language]
        self.model_params = kwargs
        self.cache = {} # Initialize cache

        if self.provider == 'gradio':
            self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
            
            if models:
                self.models = models
            elif model_id:
                self.models = [model_id]
            else:
                self.models = GRADIO_MODELS
            
            if not self.models:
                raise ValueError("Gradio provider requires at least one model. Provide a list via the 'models' parameter or ensure 'gradio_models.py' is not empty.")

            # Set default parameters for Gradio, as they are required for the submit() call.
            self.model_params.setdefault('temperature', 0.6)
            self.model_params.setdefault('max_new_tokens', 1024)
            self.model_params.setdefault('top_p', 0.9)
            self.model_params.setdefault('top_k', 50)
            self.model_params.setdefault('repetition_penalty', 1.2)

        elif self.provider == 'openai':
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ApiKeyNotFoundError(self.T["api_key_error"])
            self.model_id = model_id or self.OPENAI_MODEL_ID
            self.client = OpenAI(api_key=self.api_key)
            self.async_client = AsyncOpenAI(api_key=self.api_key)
            # Set default parameters for OpenAI if not provided
            self.model_params.setdefault('temperature', 0.7)
            self.model_params.setdefault('max_tokens', 1024) # OpenAI uses 'max_tokens'
        else:
            raise ValueError(self.T["unknown_provider"])

    def _extract_last_string_from_result(self, result: any) -> str | None:
        """
        Recursively extracts the last string from a nested list structure.
        This is a robust way to handle varied API responses from Gradio models.
        """
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, list) and len(result) > 0:
            # Recursively call on the last element
            return self._extract_last_string_from_result(result[-1])
        return None

    def _build_prompt(self, traceback: str) -> str:
        """Builds the prompt for the model."""
        return self.prompt_template.format(traceback=traceback)

    # --- Methods for Gradio provider ---

    def _predict_gradio(self, prompt: str, status=None) -> str:
        shuffled_models = random.sample(self.models, len(self.models))
        last_error = None

        for model_id in shuffled_models:
            logger.info(self.T["gradio_request"].format(model_id=model_id))
            if status:
                status.update(self.T["gradio_request_rich"].format(model_id=model_id))
            try:
                client = Client(model_id, hf_token=self.api_key)
                job = client.submit(
                    prompt,
                    self.model_params.get('max_new_tokens'),
                    self.model_params.get('temperature'),
                    self.model_params.get('top_p'),
                    self.model_params.get('top_k'),
                    self.model_params.get('repetition_penalty'),
                    api_name="/chat"
                )
                
                # Wait for the full response by checking job status
                while not job.done():
                    time.sleep(0.5) # Prevent high CPU usage

                result = job.outputs()
                
                # The result can be in various nested list/string formats.
                # We need to robustly find the last string in the structure.
                explanation = self._extract_last_string_from_result(result)

                if explanation:
                    return explanation

                # If we reach here, the response format is unexpected.
                logger.warning(f"Model '{model_id}' returned an unexpected response format: {result}")
                last_error = f"Model '{model_id}' returned an unexpected response format."
                if status:
                    status.update(self.T["gradio_fail_rich"].format(model_id=model_id))
                    time.sleep(1) # Show the message for a moment
                continue

            except Exception as e:
                logger.error(f"Error during Gradio request with model {model_id}: {e}")
                last_error = e
                if status:
                    status.update(self.T["gradio_fail_rich"].format(model_id=model_id))
                    time.sleep(1) # Show the message for a moment
                continue # Try next model

        logger.error("All Gradio models failed.")
        return self.T["gradio_error"].format(e=f"All models failed. Last error: {last_error}")

    async def _predict_async_gradio(self, prompt: str, status=None) -> str:
        shuffled_models = random.sample(self.models, len(self.models))
        last_error = None
        loop = asyncio.get_running_loop()

        for model_id in shuffled_models:
            logger.info(self.T["gradio_request_async"].format(model_id=model_id))
            if status:
                status.update(self.T["gradio_request_rich"].format(model_id=model_id))
            try:
                # Use run_in_executor to avoid blocking the event loop with sync code
                client = await loop.run_in_executor(None, lambda: Client(model_id, hf_token=self.api_key))
                
                job = await loop.run_in_executor(None, lambda: client.submit(
                    prompt,
                    self.model_params.get('max_new_tokens'),
                    self.model_params.get('temperature'),
                    self.model_params.get('top_p'),
                    self.model_params.get('top_k'),
                    self.model_params.get('repetition_penalty'),
                    api_name="/chat"
                ))
                
                # Poll for completion without blocking
                while not await loop.run_in_executor(None, job.done):
                    await asyncio.sleep(0.5)

                result = await loop.run_in_executor(None, job.outputs)

                # The result can be in various nested list/string formats.
                explanation = self._extract_last_string_from_result(result)

                if explanation:
                    return explanation
                
                logger.warning(f"Model '{model_id}' returned an unexpected response format: {result}")
                last_error = f"Model '{model_id}' returned an unexpected response format."
                if status:
                    status.update(self.T["gradio_fail_rich"].format(model_id=model_id))
                    await asyncio.sleep(1) # Show the message for a moment
                continue

            except Exception as e:
                logger.error(f"Error during async Gradio request with model {model_id}: {e}")
                last_error = e
                if status:
                    status.update(self.T["gradio_fail_rich"].format(model_id=model_id))
                    await asyncio.sleep(1) # Show the message for a moment
                continue # Try next model

        logger.error("All Gradio models failed.")
        return self.T["gradio_error_async"].format(e=f"All models failed. Last error: {last_error}")
            
    # --- Methods for OpenAI provider ---

    def _predict_openai(self, prompt: str, status=None) -> str:
        logger.info(self.T["openai_request"].format(model_id=self.model_id))
        if status:
            status.update(self.T["openai_request_rich"].format(model_id=self.model_id))
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                **self.model_params
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error during OpenAI request: {e}")
            return self.T["openai_error"].format(e=e)

    async def _predict_async_openai(self, prompt: str, status=None) -> str:
        logger.info(self.T["openai_request_async"].format(model_id=self.model_id))
        if status:
            status.update(self.T["openai_request_rich"].format(model_id=self.model_id))
        try:
            completion = await self.async_client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                **self.model_params
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error during asynchronous OpenAI request: {e}")
            return self.T["openai_error_async"].format(e=e)

    # --- Prediction dispatchers ---

    def _predict(self, prompt: str, status=None) -> str:
        if self.provider == 'gradio':
            return self._predict_gradio(prompt, status=status)
        elif self.provider == 'openai':
            return self._predict_openai(prompt, status=status)
        # This return should never be reached due to the check in __init__
        return self.T["invalid_provider"]

    async def _predict_async(self, prompt: str, status=None) -> str:
        if self.provider == 'gradio':
            return await self._predict_async_gradio(prompt, status=status)
        elif self.provider == 'openai':
            return await self._predict_async_openai(prompt, status=status)
        return self.T["invalid_provider"]

    def explain_error(self, traceback: str, status=None) -> str:
        """
        Gets an explanation for a traceback using the AI.
        Checks the cache before sending a request.
        """
        if traceback in self.cache:
            logger.info(self.T["cache_hit"])
            if status:
                status.update(self.T["cache_hit_rich"])
                time.sleep(0.5) # Show the message for a moment
            return self.cache[traceback]

        prompt = self._build_prompt(traceback)
        explanation = self._predict(prompt, status=status)
        self.cache[traceback] = explanation # Save result to cache
        return explanation

    async def explain_error_async(self, traceback: str, status=None) -> str:
        """
        Asynchronously gets an explanation for a traceback using the AI.
        Checks the cache before sending a request.
        """
        if traceback in self.cache:
            logger.info(self.T["cache_hit"])
            if status:
                status.update(self.T["cache_hit_rich"])
                await asyncio.sleep(0.5)
            return self.cache[traceback]

        prompt = self._build_prompt(traceback)
        explanation = await self._predict_async(prompt, status=status)
        self.cache[traceback] = explanation # Save result to cache
        return explanation

    def explain_and_print(self, traceback: str):
        """
        Gets an explanation, formats it with rich, and prints it to the console.
        """
        console = Console()
        with console.status(self.T["status_analysis"], spinner="dots") as status:
            explanation_md = self.explain_error(traceback, status=status)
        
        console.print(Markdown(explanation_md, style="default"))

    async def explain_and_print_async(self, traceback: str):
        """
        Asynchronously gets an explanation, formats it, and prints it to the console.
        """
        console = Console()
        with console.status(self.T["status_analysis"], spinner="dots") as status:
            explanation_md = await self.explain_error_async(traceback, status=status)
        
        console.print(Markdown(explanation_md, style="default"))