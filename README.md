[Read this README in Russian](README.ru.md)

# 🚀 Error Narrator

[![PyPI version](https://badge.fury.io/py/error-narrator.svg)](https://badge.fury.io/py/error-narrator)

**Error Narrator** is a Python library that uses AI to provide clear, human-readable explanations for Python exceptions and tracebacks. Instead of just getting a stack trace, you get a structured, educational breakdown of what went wrong, right in your console.

The library is multilingual, currently supporting English (default) and Russian.

## 📦 Features

-   **🤖 AI-Powered Explanations**: Uses language models from Gradio or OpenAI to explain errors.
-   **📝 Structured Output**: Provides a clear, markdown-formatted explanation with:
    -   🎯 **Root Cause**: What caused the error.
    -   📍 **Error Location**: Pinpoints the exact file and line.
    -   🛠️ **Suggested Fix**: Offers a code diff for a potential solution.
    -   🎓 **A Learning Moment**: Explains the underlying concepts to prevent future mistakes.
-   **🎨 Rich Console Output**: Uses the `rich` library to print beautiful, colorized output in the terminal.
-   **⚡ Async Support**: Provides asynchronous methods (`*_async`) for non-blocking operations.
-   **💾 Caching**: Caches explanations for identical tracebacks to speed up repeated runs and reduce API calls.
-   **🌐 Multilingual**: Supports explanations in English (`en`) and Russian (`ru`).

## 💾 Installation

```bash
pip install error-narrator
```

## 📝 How to Use

### 1. 🔑 Get an API Key

The library requires an API key for the chosen provider.

-   **Gradio (Default)**: You will need a Hugging Face User Access Token. You can get one from your [Hugging Face account settings](https://huggingface.co/settings/tokens).
-   **OpenAI**: You will need an API key from your [OpenAI dashboard](https://platform.openai.com/api-keys).

💡 **Tip:** It is recommended to set your API key as an environment variable:
-   `HUGGINGFACE_API_KEY` for Gradio.
-   `OPENAI_API_KEY` for OpenAI.

### 2. ⚙️ Basic Usage

Here is a simple example of how to use `ErrorNarrator`. The library will automatically catch exceptions within a `try...except` block and explain them.

By default, the explanation will be in **English**.

```python
import traceback
from error_narrator import ErrorNarrator

# The narrator will automatically look for the HUGGINGFACE_API_KEY environment variable
# if no api_key is provided.
narrator = ErrorNarrator() 

try:
    # Some code that might raise an error
    result = 1 / 0
except Exception:
    # Get the traceback as a string
    traceback_str = traceback.format_exc()
    # Get the explanation and print it to the console
    narrator.explain_and_print(traceback_str)

```

### 3. 🌍 Getting Explanations in Russian

To get explanations in a different language, use the `language` parameter during initialization.

```python
# ...
# Initialize with Russian language support
narrator = ErrorNarrator(language="ru")
# ...
```

### 4. 🏷️ Using a specific provider (e.g., OpenAI)

You can also specify a provider and pass the API key directly.

```python
narrator = ErrorNarrator(
    provider="openai",
    api_key="your-openai-api-key"
)
# ...
```

## 🤝 Development

Contributions are welcome! Please feel free to submit a pull request or open an issue.