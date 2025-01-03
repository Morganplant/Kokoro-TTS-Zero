import os
import gradio as gr
import spaces
import time
from tts_model import TTSModel
from lib import format_audio_output

# Set HF_HOME for faster restarts with cached models/voices
os.environ["HF_HOME"] = "/data/.huggingface"

# Create TTS model instance
model = TTSModel()

@spaces.GPU(duration=10)  # Quick initialization
def initialize_model():
    """Initialize model and get voices"""
    if model.model is None:
        if not model.initialize():
            raise gr.Error("Failed to initialize model")
    return model.list_voices()

# Get initial voice list
voice_list = initialize_model()

@spaces.GPU(duration=120)  # Allow 5 minutes for processing
def generate_speech_from_ui(text, voice_name, speed, progress=gr.Progress(track_tqdm=False)):
    """Handle text-to-speech generation from the Gradio UI"""
    try:
        start_time = time.time()
        gpu_timeout = 120  # seconds
        
        # Create progress state
        progress_state = {
            "progress": 0.0,
            "tokens_per_sec": 0.0,
            "gpu_time_left": gpu_timeout
        }
        
        def update_progress(chunk_num, total_chunks, tokens_per_sec, rtf):
            progress_state["progress"] = chunk_num / total_chunks
            progress_state["tokens_per_sec"] = tokens_per_sec
            
            # Update GPU time remaining
            elapsed = time.time() - start_time
            gpu_time_left = max(0, gpu_timeout - elapsed)
            progress_state["gpu_time_left"] = gpu_time_left
            
            # Only update progress display during processing
            progress(progress_state["progress"], desc=f"Processing chunk {chunk_num}/{total_chunks} | GPU Time Left: {int(gpu_time_left)}s")
        
        # Generate speech with progress tracking
        audio_array, duration = model.generate_speech(
            text, 
            voice_name, 
            speed,
            progress_callback=update_progress
        )
        
        # Format output for Gradio
        audio_output, duration_text = format_audio_output(audio_array)
        
        # Calculate final metrics
        total_time = time.time() - start_time
        total_duration = len(audio_array) / 24000  # audio duration in seconds
        final_rtf = total_time / total_duration if total_duration > 0 else 0
        
        # Prepare final metrics display
        metrics_text = (
            f"Tokens/sec: {progress_state['tokens_per_sec']:.1f}\n" +
            f"Real-time factor: {final_rtf:.2f}x (Processing Time / Audio Duration)\n" +
            f"GPU Time Used: {int(total_time)}s of {gpu_timeout}s"
        )
        
        return (
            audio_output,
            metrics_text,
            duration_text
        )
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")

# Create Gradio interface
with gr.Blocks(title="Kokoro TTS Demo") as demo:
    gr.HTML(
        """
        <div style="display: flex; justify-content: flex-end; padding: 10px; gap: 10px;">
            <a href="https://huggingface.co/hexgrad/Kokoro-82M" target="_blank">
                <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md-dark.svg" alt="Model on HF">
            </a>
            <a class="github-button" href="https://github.com/remsky/Kokoro-FastAPI" data-color-scheme="no-preference: light; light: light; dark: dark;" data-size="large" data-show-count="true" aria-label="Star remsky/Kokoro-FastAPI on GitHub">Repo for Local Use</a>
        </div>
        <div style="text-align: center; max-width: 800px; margin: 0 auto;">
            <h1>Kokoro TTS Demo</h1>
            <p>Convert text to natural-sounding speech using various voices.</p>
        </div>
        <script async defer src="https://buttons.github.io/buttons.js"></script>
        """
    )
    
    with gr.Row():
        # Column 1: Text Input
        with gr.Column():
            text_input = gr.TextArea(
                label="Text to speak",
                placeholder="Enter text here or upload a .txt file",
                lines=10,
                value=open("the_time_machine_hgwells.txt").read()[:1000]
            )
        
        # Column 2: Controls
        with gr.Column():
            file_input = gr.File(
                label="Upload .txt file",
                file_types=[".txt"],
                type="binary"
            )
            
            def load_text_from_file(file_bytes):
                if file_bytes is None:
                    return None
                try:
                    return file_bytes.decode('utf-8')
                except Exception as e:
                    raise gr.Error(f"Failed to read file: {str(e)}")

            file_input.change(
                fn=load_text_from_file,
                inputs=[file_input],
                outputs=[text_input]
            )
            
            with gr.Group():
                voice_dropdown = gr.Dropdown(
                    label="Voice",
                    choices=voice_list,
                    value=voice_list[0] if voice_list else None,
                    allow_custom_value=True
                )
                speed_slider = gr.Slider(
                    label="Speed",
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1
                )
                submit_btn = gr.Button("Generate Speech", variant="primary")
        
        # Column 3: Output
        with gr.Column():
            audio_output = gr.Audio(
                label="Generated Speech",
                type="numpy",
                format="wav",
                autoplay=False
            )
            progress_bar = gr.Progress(track_tqdm=False)
            metrics_text = gr.Textbox(
                label="Processing Metrics",
                interactive=False,
                lines=3
            )
            duration_text = gr.Textbox(
                label="Processing Info",
                interactive=False,
                lines=2
            )
    
    # Set up event handler
    submit_btn.click(
        fn=generate_speech_from_ui,
        inputs=[text_input, voice_dropdown, speed_slider],
        outputs=[audio_output, metrics_text, duration_text],
        show_progress=True
    )
    
    # Add text analysis info
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ### Demo Text Info
            The demo text is loaded from H.G. Wells' "The Time Machine". This classic text demonstrates the system's ability to handle long-form content through chunking.
            """)
        

# Launch the app
if __name__ == "__main__":
    demo.launch()
