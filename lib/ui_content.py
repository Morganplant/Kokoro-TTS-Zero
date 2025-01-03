# HTML content for the header section
header_html = """
<div>
    <!-- Top badges bar -->
    <div style="display: flex; justify-content: flex-end; padding: 4px; gap: 8px; height: 32px; align-items: center;">
        <div style="height: 28px; display: flex; align-items: center; margin-top: 3px;">
            <a class="github-button" href="https://github.com/remsky/Kokoro-FastAPI" data-color-scheme="no-preference: dark; light: dark; dark: dark;" data-size="large" data-show-count="true" aria-label="Star remsky/Kokoro-FastAPI on GitHub">Kokoro-FastAPI Repo</a>
        </div>
        <a href="https://huggingface.co/hexgrad/Kokoro-82M" target="_blank" style="height: 28px; display: flex; align-items: center;">
            <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-lg-dark.svg" alt="Model on HF" style="height: 100%;">
        </a>
    </div>

    <div style="text-align: center; margin-bottom: 1rem;">
        <h1 style="font-size: 1.75rem; font-weight: bold; color: #ffffff; margin-bottom: 0.5rem;">Kokoro TTS Demo</h1>
        <p style="color: #d1d5db;">Rapidly convert text to natural speech using various and blended voices.</p>
    </div>
        
    <div style="display: flex; gap: 1rem;">
        <div style="flex: 1; background: rgba(30, 58, 138, 0.3); border: 1px solid rgba(59, 130, 246, 0.3); padding: 0.5rem 1rem; border-radius: 6px; display: flex; align-items: center; justify-content: center;">
            <span style="font-weight: 500; color: #60a5fa; text-align: center;">⏱️ Small requests/Initial chunks can be slower due to warm-up</span>
        </div>
        
        <div style="flex: 1; background: rgba(147, 51, 234, 0.3); border: 1px solid rgba(168, 85, 247, 0.3); padding: 0.5rem 1rem; border-radius: 6px; display: flex; align-items: center; justify-content: center;">
            <span style="font-weight: 500; color: #e879f9; text-align: center;">⚠️ 120-second timeout per request (~1+ hour of output audio)</span>
        </div>
    </div>
</div>
<script async defer src="https://buttons.github.io/buttons.js"></script>
"""

# Markdown content for demo text info
demo_text_info = """
### Demo Text Info
The demo text is loaded from H.G. Wells' "The Time Machine". This classic text demonstrates the system's ability to handle long-form content through chunking.
"""
