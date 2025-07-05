# A list of public Gradio Spaces that can be used as models for narration.
# Add your own compatible Gradio Space here to expand the model pool.
# This helps distribute the load and provides fallback options.
# Make sure the Gradio Space is public and supports a text generation interface
# compatible with the standard chat API (`/chat`).

GRADIO_MODELS = [
    "hysts/mistral-7b",
    "huggingface-projects/gemma-3n-E4B-it",
    # This is a very popular and generally reliable alternative.
] 