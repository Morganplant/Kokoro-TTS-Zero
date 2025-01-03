---
title: Kokoro TTS Zero
emoji: 🎴
colorFrom: gray
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: true
short_description: A100 GPU Accelerated Inference on Kokoro-82M Text-to-Speech
models:
- hexgrad/Kokoro-82M
---

# Kokoro TTS Demo Space

A Zero GPU-optimized Hugging Face Space for the Kokoro TTS model.
## Overview

This Space provides a Gradio interface for the Kokoro TTS model, allowing users to:
- Convert text to speech using multiple voices
- Adjust speech speed
## Project Structure

```
.
├── app.py              # Main Gradio interface
├── tts_model.py        # GPU-accelerated TTS model manager
├── lib/                # Utility modules
│   ├── __init__.py    # Package exports
│   ├── text_utils.py  # Text processing utilities
│   ├── file_utils.py  # File operations
│   └── audio_utils.py # Audio processing
└── requirements.txt    # Project dependencies
```

## Dependencies

Main dependencies:
- PyTorch 2.2.2
- Gradio 5.9.1
- Transformers 4.47.1
- HuggingFace Hub ≥0.25.1

For a complete list, see requirements.txt.