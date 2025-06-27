import os
import httpx
import logging

# Настраиваем базовый логгер для библиотеки
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ErrorNarrator:
    """
    Класс для получения объяснений ошибок с помощью моделей Hugging Face.
    """
    DEFAULT_MODEL_ID = "MiniMaxAI/MiniMax-M1"
    API_URL_TEMPLATE = "https://huggingface.co/spaces/MiniMaxAI/MiniMax-M1"
    DEFAULT_PROMPT_TEMPLATE = (
        "You are an expert Python developer's assistant. An internal error occurred in an application. "
        "Your task is to analyze the traceback and provide a concise analysis for the developer in Russian. "
        "Your analysis should be structured and include:\n"
        "1. **Root Cause:** A clear explanation of the error's root cause.\n"
        "2. **Location:** The exact file and line number where the error occurred.\n"
        "3. **Suggested Fix:** A code snippet or a clear, actionable suggestion for debugging and fixing the issue.\n\n"
        "Here is the technical traceback:\n"
        "```\n{traceback}\n```\n\n"
        "Provide a structured analysis for the developer's logs. Do not address the user."
    )

    def __init__(self, api_key: str = None, model_id: str = None, prompt_template: str = None):
        """
        Инициализирует ErrorNarrator.

        :param api_key: API-ключ от Hugging Face. Если не указан, будет предпринята
                        попытка получить его из переменной окружения HUGGINGFACE_API_KEY.
        :param model_id: Идентификатор модели на Hugging Face. Если не указан,
                         будет использована модель по умолчанию.
        :param prompt_template: Шаблон промпта для модели. Должен содержать плейсхолдер {traceback}.
        """
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("API-ключ не найден. Передайте его в конструктор или установите переменную окружения HUGGINGFACE_API_KEY.")

        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.api_url = self.API_URL_TEMPLATE.format(model_id=self.model_id)
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def _build_prompt(self, traceback: str) -> str:
        """Формирует промпт для модели."""
        return self.prompt_template.format(traceback=traceback)

    async def explain_error(self, traceback: str) -> str:
        """
        Получает объяснение для ошибки (traceback) с помощью AI.

        :param traceback: Строка с полным traceback'ом ошибки.
        :return: Строка с объяснением от AI или сообщение об ошибке.
        """
        default_explanation = "К сожалению, не удалось получить объяснение от AI."
        prompt = self._build_prompt(traceback)
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 350,
                "return_full_text": False,
            }
        }

        logger.info(f"Запрашиваю объяснение у AI (модель: {self.model_id})...")

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(self.api_url, headers=self.headers, json=payload)
                response.raise_for_status()
            
            result = response.json()
            explanation = result[0]['generated_text'].strip()

            if not explanation:
                raise ValueError("AI returned an empty explanation.")
            
            logger.info("Объяснение от AI успешно получено.")
            return explanation

        except httpx.HTTPStatusError as e:
            logger.error(f"Ошибка API Hugging Face: {e.response.status_code} - {e.response.text}")
            return f"{default_explanation} (Ошибка API: {e.response.status_code})"
        except Exception as e:
            logger.error(f"Непредвиденная ошибка при запросе к AI: {e}")
            return f"{default_explanation} (Внутренняя ошибка: {e})" 