[Read this README in Russian](README.ru.md)

# 🚀 Error Narrator

[![PyPI version](https://badge.fury.io/py/error-narrator.svg)](https://badge.fury.io/py/error-narrator)

**Error Narrator** is a Python library that uses AI to provide clear, human-readable explanations for Python exceptions and tracebacks. Instead of just getting a stack trace, you get a structured, educational breakdown of what went wrong, right in your console.

![Error Narrator Screenshot](https://i.postimg.cc/BbydKLcV/2025-07-05-135631.png)

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

### 1. 🔑 API Key Configuration

The library supports two providers: `gradio` (default) and `openai`.

-   **Gradio (Default & Free)**: This provider uses public, community-hosted models on Hugging Face Spaces. An API key (Hugging Face User Access Token) is **optional** for most public models but may be required for private models or to increase rate limits. You can get a token from your [Hugging Face account settings](https://huggingface.co/settings/tokens).
-   **OpenAI (Higher Quality & Paid)**: This provider uses official OpenAI models. An API key from your [OpenAI dashboard](https://platform.openai.com/api-keys) is **mandatory**.

💡 **Tip:** The library will automatically detect API keys set as environment variables:
-   `HUGGINGFACE_API_KEY` for Gradio (optional).
-   `OPENAI_API_KEY` for OpenAI (required).

#### How to Set Environment Variables

You can set these variables for your current terminal session or add them to your shell's profile file (e.g., `.bashrc`, `.zshrc`, or your system's environment variable settings) for them to be permanent.

**On macOS/Linux:**
```bash
export HUGGINGFACE_API_KEY="your-key-here"
```

**On Windows (Command Prompt):**
```cmd
set HUGGINGFACE_API_KEY="your-key-here"
```

**On Windows (PowerShell):**
```powershell
$env:HUGGINGFACE_API_KEY="your-key-here"
```

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

### 5. 🔄 Gradio Provider: Model List and Rotation

By default, the `gradio` provider uses a predefined list of public models and tries them in a random order for each request. This rotation and retry system increases the chances of getting a successful response.

#### How to Provide Your Own Model List (Recommended)

The best way to customize the model list is to pass your own list directly during initialization using the `models` parameter. This gives you full control without editing any package files.

```python
# Provide a custom list of models for rotation
my_favorite_models = [
    "hysts/mistral-7b",
    "Tblue/gemma-7b-it",
    "your-username/your-own-model"
]

narrator = ErrorNarrator(
    provider="gradio",
    models=my_favorite_models
)
```

#### How to Use a Single Specific Model

If you want to disable rotation and use only one model, you can use the `model_id` parameter as a convenient shortcut.

```python
# This will ONLY use the specified model
narrator = ErrorNarrator(provider="gradio", model_id="hysts/mistral-7b")
```

#### How to Edit the Default Model List

If you want to change the default behavior for all your projects, you can edit the `gradio_models.py` file directly in your Python `site-packages` directory. This is generally not recommended unless you know what you are doing.

## 🤝 Development

Contributions are welcome! Please feel free to submit a pull request or open an issue.