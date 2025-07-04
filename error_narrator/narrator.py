import os
import logging
import asyncio
from gradio_client import Client as GradioClient
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
        "status_analysis": "[bold green]Анализирую ошибку с помощью AI...[/]"
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
        "status_analysis": "[bold green]Analyzing the error with AI...[/]"
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
    GRADIO_MODEL_ID = "hysts/mistral-7b"
    OPENAI_MODEL_ID = "gpt-3.5-turbo"

    def __init__(self, provider: str = 'gradio', language: str = 'en', api_key: str = None, model_id: str = None, prompt_template: str = None, **kwargs):
        """
        Initializes the ErrorNarrator.

        :param provider: The provider to use for explanations ('gradio' or 'openai').
        :param language: The language for the AI response ('en' or 'ru').
        :param api_key: The API key. If not provided, it will be sourced from environment variables
                        (HUGGINGFACE_API_KEY for 'gradio', OPENAI_API_KEY for 'openai').
        :param model_id: The model identifier. If not provided, a default is used for the provider.
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
            self.model_id = model_id or self.GRADIO_MODEL_ID
            self.client = GradioClient(self.model_id, hf_token=self.api_key)
            # Set default parameters for Gradio if not provided
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

    def _build_prompt(self, traceback: str) -> str:
        """Builds the prompt for the model."""
        return self.prompt_template.format(traceback=traceback)

    # --- Methods for Gradio provider ---

    def _predict_gradio(self, prompt: str) -> str:
        logger.info(self.T["gradio_request"].format(model_id=self.model_id))
        try:
            job = self.client.submit(
                prompt,
                self.model_params.get('max_new_tokens'),
                self.model_params.get('temperature'),
                self.model_params.get('top_p'),
                self.model_params.get('top_k'),
                self.model_params.get('repetition_penalty'),
                api_name="/chat"
            )
            # Wait for the full response
            while not job.done():
                pass
            result = job.outputs()[-1] # The last result is the final one
            return result.strip()
        except Exception as e:
            logger.error(f"Error during Gradio request: {e}")
            return self.T["gradio_error"].format(e=e)

    async def _predict_async_gradio(self, prompt: str) -> str:
        logger.info(self.T["gradio_request_async"].format(model_id=self.model_id))
        loop = asyncio.get_running_loop()
        try:
            # Use run_in_executor for the blocking call
            job = await loop.run_in_executor(None, lambda: self.client.submit(
                prompt,
                self.model_params.get('max_new_tokens'),
                self.model_params.get('temperature'),
                self.model_params.get('top_p'),
                self.model_params.get('top_k'),
                self.model_params.get('repetition_penalty'),
                api_name="/chat"
            ))
            
            # Asynchronously wait for completion
            while not job.done():
                await asyncio.sleep(0.1)

            result = job.outputs()[-1]
            return result.strip()
        except Exception as e:
            logger.error(f"Error during asynchronous Gradio request: {e}")
            return self.T["gradio_error_async"].format(e=e)
            
    # --- Methods for OpenAI provider ---

    def _predict_openai(self, prompt: str) -> str:
        logger.info(self.T["openai_request"].format(model_id=self.model_id))
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

    async def _predict_async_openai(self, prompt: str) -> str:
        logger.info(self.T["openai_request_async"].format(model_id=self.model_id))
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

    def _predict(self, prompt: str) -> str:
        if self.provider == 'gradio':
            return self._predict_gradio(prompt)
        elif self.provider == 'openai':
            return self._predict_openai(prompt)
        # This return should never be reached due to the check in __init__
        return self.T["invalid_provider"]

    async def _predict_async(self, prompt: str) -> str:
        if self.provider == 'gradio':
            return await self._predict_async_gradio(prompt)
        elif self.provider == 'openai':
            return await self._predict_async_openai(prompt)
        return self.T["invalid_provider"]

    def explain_error(self, traceback: str) -> str:
        """
        Gets an explanation for a traceback using the AI.
        Checks the cache before sending a request.
        """
        if traceback in self.cache:
            logger.info(self.T["cache_hit"])
            return self.cache[traceback]

        prompt = self._build_prompt(traceback)
        explanation = self._predict(prompt)
        self.cache[traceback] = explanation # Save result to cache
        return explanation

    async def explain_error_async(self, traceback: str) -> str:
        """
        Asynchronously gets an explanation for a traceback using the AI.
        Checks the cache before sending a request.
        """
        if traceback in self.cache:
            logger.info(self.T["cache_hit"])
            return self.cache[traceback]

        prompt = self._build_prompt(traceback)
        explanation = await self._predict_async(prompt)
        self.cache[traceback] = explanation # Save result to cache
        return explanation

    def explain_and_print(self, traceback: str):
        """
        Gets an explanation, formats it with rich, and prints it to the console.
        """
        console = Console()
        with console.status(self.T["status_analysis"], spinner="dots"):
            explanation_md = self.explain_error(traceback)
        
        console.print(Markdown(explanation_md, style="default"))

    async def explain_and_print_async(self, traceback: str):
        """
        Asynchronously gets an explanation, formats it, and prints it to the console.
        """
        console = Console()
        with console.status(self.T["status_analysis"], spinner="dots"):
            explanation_md = await self.explain_error_async(traceback)
        
        console.print(Markdown(explanation_md, style="default"))