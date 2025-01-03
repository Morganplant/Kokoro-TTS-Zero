---
title: Kokoro TTS Zero
emoji: 📊
colorFrom: gray
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: true
license: apache-2.0
short_description: A100 GPU Accelerated Inference applied to Kokoro-82M TTS
models:
- hexgrad/Kokoro-82M
---

# Kokoro TTS Demo Space

A Zero GPU-optimized Hugging Face Space for the Kokoro TTS model.

## Overview

This Space provides a Gradio interface for the Kokoro TTS model, allowing users to:
- Convert text to speech using multiple voices
- Adjust speech speed
- Get instant audio playback

## Technical Details

- Zero GPU for efficient GPU resource management
- Dynamically loads required modules from hexgrad/Kokoro-82M repository

## Dependencies
- hexgrad/Kokoro-82M: Original model repository (core TTS functionality)

All dependencies are automatically handled:
- Core modules (kokoro.py, models.py, etc.) are downloaded from hexgrad/Kokoro-82M
- Model weights and voice files are cached in /data/.huggingface
- System dependencies (espeak-ng) are installed via packages.txt

## Environment

- Python 3.10.13
- PyTorch 2.2.2
- Gradio 5.9.1
- A100 Zero GPU Enabled

## Available Voices

af: Default
af_sky: Classic
af_bella: Warm
af_nicole: Soothing
af_sarah: Polished
bf_emma: Contemplative
bf_isabella: Poised

am_adam: Resonant
am_michael: Sincere
bm_george: Distinguished
bm_lewis: Gravelly

## Notes
- Model Warm-Up takes some time, it shines at longer lengths. 
