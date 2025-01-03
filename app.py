import os
import gradio as gr
import spaces
import time
import matplotlib.pyplot as plt
import numpy as np
from tts_model import TTSModel
from lib import format_audio_output
from lib.ui_content import header_html, demo_text_info

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
            "tokens_per_sec": [],
            "rtf": [],
            "chunk_times": [],
            "gpu_time_left": gpu_timeout,
            "total_chunks": 0
        }
        
        def update_progress(chunk_num, total_chunks, tokens_per_sec, rtf):
            progress_state["progress"] = chunk_num / total_chunks
            progress_state["tokens_per_sec"].append(tokens_per_sec)
            progress_state["rtf"].append(rtf)
            
            # Update GPU time remaining
            elapsed = time.time() - start_time
            gpu_time_left = max(0, gpu_timeout - elapsed)
            progress_state["gpu_time_left"] = gpu_time_left
            progress_state["total_chunks"] = total_chunks
            
            # Track individual chunk processing time
            chunk_time = elapsed - (sum(progress_state["chunk_times"]) if progress_state["chunk_times"] else 0)
            progress_state["chunk_times"].append(chunk_time)
            
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
        rtf = total_time / total_duration if total_duration > 0 else 0
        mean_tokens_per_sec = np.mean(progress_state["tokens_per_sec"])
        
        # Create plot of tokens per second with median line
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        chunk_nums = list(range(1, len(progress_state["tokens_per_sec"]) + 1))
        
        # Plot bars for tokens per second
        ax.bar(chunk_nums, progress_state["tokens_per_sec"], color='#ff2a6d', alpha=0.8)
        
        # Add median line
        median_tps = np.median(progress_state["tokens_per_sec"])
        ax.axhline(y=median_tps, color='#05d9e8', linestyle='--', label=f'Median: {median_tps:.1f} tokens/sec')
        
        # Style improvements
        ax.set_xlabel('Chunk Number', fontsize=24, labelpad=20)
        ax.set_ylabel('Tokens per Second', fontsize=24, labelpad=20)
        ax.set_title('Processing Speed by Chunk', fontsize=28, pad=30)
        
        # Increase tick label size
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        # Remove gridlines
        ax.grid(False)
        
        # Style legend and position it in bottom left
        ax.legend(fontsize=20, facecolor='black', edgecolor='#05d9e8', loc='lower left')
        
        plt.tight_layout()
        
        # Prepare final metrics display including audio duration and real-time speed
        metrics_text = (
            f"Median Processing Speed: {np.median(progress_state['tokens_per_sec']):.1f} tokens/sec\n" +
            f"Real-time Factor: {rtf:.3f}\n" +
            f"Real Time Generation Speed: {int(1/rtf)}x \n" +
            f"Processing Time: {int(total_time)}s\n" +
            f"Output Audio Duration: {total_duration:.2f}s"
        )
        
        return (
            audio_output,
            fig,
            metrics_text
        )
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")

# Create Gradio interface
with gr.Blocks(title="Kokoro TTS Demo", css="""
    .equal-height {
        min-height: 400px;
        display: flex;
        flex-direction: column;
    }
""") as demo:
    gr.HTML(header_html)
    
    with gr.Row():
        # Column 1: Text Input
        with gr.Column(elem_classes="equal-height"):
            text_input = gr.TextArea(
                label="Text to speak",
                placeholder="Enter text here or upload a .txt file",
                lines=10,
                value=open("the_time_machine_hgwells.txt").read()[:1000]
            )
        
        # Column 2: Controls
        with gr.Column(elem_classes="equal-height"):
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
                default_voice = 'af_sky' if 'af_sky' in voice_list \
                    else voice_list[0] \
                        if voice_list else \
                            None
                        
                voice_dropdown = gr.Dropdown(
                    label="Voice",
                    choices=voice_list,
                    value=default_voice,
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
        with gr.Column(elem_classes="equal-height"):
            audio_output = gr.Audio(
                label="Generated Speech",
                type="numpy",
                format="wav",
                autoplay=False
            )
            progress_bar = gr.Progress(track_tqdm=False)
            metrics_text = gr.Textbox(
                label="Performance Summary",
                interactive=False,
                lines=4
            )
            metrics_plot = gr.Plot(
                label="Processing Metrics",
                show_label=True,
                format="png"  # Explicitly set format to PNG which is supported by matplotlib
            )
    
    # Set up event handler
    submit_btn.click(
        fn=generate_speech_from_ui,
        inputs=[text_input, voice_dropdown, speed_slider],
        outputs=[audio_output, metrics_plot, metrics_text],
        show_progress=True
    )
    
    # Add text analysis info
    with gr.Row():
        with gr.Column():
            gr.Markdown(demo_text_info)

# Launch the app
if __name__ == "__main__":
    demo.launch()
