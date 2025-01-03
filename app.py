import os
import gradio as gr
import spaces
from tts_model import TTSModel
import numpy as np

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
def generate_speech_from_ui(text, voice_name, speed):
    """Handle text-to-speech generation from the Gradio UI"""
    try:
        audio_array, duration = model.generate_speech(text, voice_name, speed)
        # Convert float array to int16 range (-32768 to 32767)
        audio_array = np.array(audio_array, dtype=np.float32)
        audio_array = (audio_array * 32767).astype(np.int16)
        return (24000, audio_array), f"Audio Duration: {duration:.2f} seconds\nProcessing complete - check console for detailed metrics"
    except Exception as e:
        raise gr.Error(str(e))

# Create Gradio interface
with gr.Blocks(title="Kokoro TTS Demo") as demo:
    gr.HTML(
        """
        <div style="text-align: center; max-width: 800px; margin: 0 auto;">
            <h1>Kokoro TTS Demo</h1>
            <p>Convert text to natural-sounding speech using various voices.</p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            # Input components
            text_input = gr.TextArea(
                label="Text to speak",
                placeholder="Enter text here...",
                lines=3,
                value=open("the_time_machine_hgwells.txt").read()[:1000]
            )
            voice_dropdown = gr.Dropdown(
                label="Voice",
                choices=voice_list,
                value=voice_list[0] if voice_list else None,
                allow_custom_value=True  # Allow custom values to avoid warnings
            )
            speed_slider = gr.Slider(
                label="Speed",
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1
            )
            submit_btn = gr.Button("Generate Speech")
        
        with gr.Column(scale=2):
            # Output components
            audio_output = gr.Audio(
                label="Generated Speech",
                type="numpy",
                format="wav",
                autoplay=False
            )
            duration_text = gr.Textbox(
                label="Processing Info",
                interactive=False,
                lines=4
            )
    
    # Set up event handler
    submit_btn.click(
        fn=generate_speech_from_ui,
        inputs=[text_input, voice_dropdown, speed_slider],
        outputs=[audio_output, duration_text]
    )
    
    # Add voice descriptions
    gr.Markdown("""
    ### Available Voices
    - Adult Female (af): Base female voice
        - Bella (af_bella): Warm and friendly
        - Nicole (af_nicole): Warm and Whispered
        - Sarah (af_sarah): Soft and gentle
        - Sky (af_sky): You know her, you love her
    - Adult Male (am):  Base male voice
        - Adam (am_adam): Clear and Friendly
        - Michael (am_michael): Smooth and natural
    - Young Female (bf):
        - Emma (bf_emma): Sweet and cheerful
        - Isabella (bf_isabella): Lively and expressive
    - Young Male (bm):
        - George (bm_george): Young and energetic
        - Lewis (bm_lewis): Deep and confident
    """)
    
    # Add text analysis info
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ### Demo Text Info
            The demo text is loaded from H.G. Wells' "The Time Machine". This classic text demonstrates the system's ability to handle long-form content through chunking.
            """)
            
            text_stats = gr.Textbox(
                label="Text Statistics",
                interactive=False,
                value=f"Characters: {len(open('the_time_machine_hgwells.txt').read())}\nEstimated chunks: {len(open('the_time_machine_hgwells.txt').read()) // 300 + 1}"
            )

# Launch the app
if __name__ == "__main__":
    demo.launch()
