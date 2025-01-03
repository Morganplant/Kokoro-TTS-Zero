---
title: Kokoro TTS Zero
emoji: 📊
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: A100 GPU Accelerated Inference applied to Kokoro-82M TTS
---

# Kokoro TTS Demo Space

A Zero GPU-optimized Hugging Face Space for the Kokoro TTS model.

## Overview

This Space provides a Gradio interface for the Kokoro TTS model, allowing users to:
- Convert text to speech using multiple voices
- Adjust speech speed
- Get instant audio playback

## Technical Details

- Uses Zero GPU for efficient GPU resource management
- Dynamically loads required modules from hexgrad/Kokoro-82M repository
- Automatically downloads model and voice files from Hugging Face Hub
- Implements proper GPU memory handling
- Includes caching in /data/.huggingface for faster restarts

## Dependencies

The Space uses modules from two repositories:
- remsky/Kokoro-FastAPI: This repository (UI and Zero GPU implementation)
- hexgrad/Kokoro-82M: Original model repository (core TTS functionality)

All dependencies are automatically handled:
- Core modules (kokoro.py, models.py, etc.) are downloaded from hexgrad/Kokoro-82M
- Model weights and voice files are cached in /data/.huggingface
- System dependencies (espeak-ng) are installed via packages.txt

## Environment

- Python 3.10.13
- PyTorch 2.2.2
- Gradio 5.9.1
- Zero GPU compatible

## Available Voices

Adult Female voices:
- af: Confident, Friendly
- af_sky: You know and Love her
- af_bella: Warm and Self-Assured
- af_nicole: Whispered, ASMR
- af_sarah: Bright and Professional
- bf_emma: Pensive and Confident, British
- bf_isabella: Young Professional, British

Adult Male voices:
- am_adam: Deep Narrative Voice
- am_michael: Trustworthy and Thoughtful
- bm_george: Distinguished older voice, British
- bm_lewis: Assured and Raspy, British

## Notes

- First generation may take longer due to model initialization
- GPU is allocated only during speech generation
- Model and voices are cached in /data/.huggingface for faster subsequent runs
