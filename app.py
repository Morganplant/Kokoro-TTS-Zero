import os
import gradio as gr
import time
import math
import logging
import matplotlib.pyplot as plt
import numpy as np
# from lib.mock_tts import MockTTSModel
from lib import format_audio_output
from lib.ui_content import header_html, demo_text_info
from lib.book_utils import get_available_books, get_book_info, get_chapter_text
from lib.text_utils import count_tokens
from tts_model import TTSModel

# Set HF_HOME for faster restarts with cached models/voices
os.environ["HF_HOME"] = "/data/.huggingface"

# Create TTS model instance
model = TTSModel()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
# Suppress matplotlib debug messages
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.debug("Starting app initialization...")


model = TTSModel()

def initialize_model():
    """Initialize model and get voices"""
    if model.model is None:
        if not model.initialize():
            raise gr.Error("Failed to initialize model")
    
    voices = model.list_voices()
    if not voices:
        raise gr.Error("No voices found. Please check the voices directory.")
        
    default_voice = 'af_sky' if 'af_sky' in voices else voices[0] if voices else None
    
    return gr.update(choices=voices, value=default_voice)

def update_progress(chunk_num, total_chunks, tokens_per_sec, rtf, progress_state, start_time, gpu_timeout, progress):
    # Calculate time metrics
    elapsed = time.time() - start_time
    gpu_time_left = max(0, gpu_timeout - elapsed)
    
    # Calculate chunk time more accurately
    prev_total_time = sum(progress_state["chunk_times"]) if progress_state["chunk_times"] else 0
    chunk_time = elapsed - prev_total_time
    
    # Validate metrics before adding to state
    if chunk_time > 0 and tokens_per_sec >= 0:
        # Update progress state with validated metrics
        progress_state["progress"] = chunk_num / total_chunks
        progress_state["total_chunks"] = total_chunks
        progress_state["gpu_time_left"] = gpu_time_left
        progress_state["tokens_per_sec"].append(float(tokens_per_sec))
        progress_state["rtf"].append(float(rtf))
        progress_state["chunk_times"].append(chunk_time)
    
    # Only update progress display during processing
    progress(progress_state["progress"], desc=f"Processing chunk {chunk_num}/{total_chunks} | GPU Time Left: {int(gpu_time_left)}s")

def generate_speech_from_ui(text, voice_names, speed, gpu_timeout, progress=gr.Progress(track_tqdm=False)):
    """Handle text-to-speech generation from the Gradio UI"""
    try:
        if not text or not voice_names:
            raise gr.Error("Please enter text and select at least one voice")
            
        start_time = time.time()
        
        # Create progress state with explicit type initialization
        progress_state = {
            "progress": 0.0,
            "tokens_per_sec": [],  # Initialize as empty list
            "rtf": [],  # Initialize as empty list
            "chunk_times": [],  # Initialize as empty list
            "gpu_time_left": float(gpu_timeout),  # Ensure float
            "total_chunks": 0
        }
        
        # Handle single or multiple voices
        if isinstance(voice_names, str):
            voice_names = [voice_names]
        
        # Generate speech with progress tracking using combined voice
        audio_array, duration, metrics = model.generate_speech(
            text,
            voice_names,
            speed,
            gpu_timeout=gpu_timeout,
            progress_callback=update_progress,
            progress_state=progress_state,
            progress=progress
        )
    
        # Format output for Gradio
        audio_output, duration_text = format_audio_output(audio_array)
        
        # Create plot and metrics text outside GPU context
        fig, metrics_text = create_performance_plot(metrics, voice_names)
        
        return (
            audio_output,
            fig,
            metrics_text
        )
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")

def create_performance_plot(metrics, voice_names):
    """Create performance plot and metrics text from generation metrics"""
    # Clean and process the data
    tokens_per_sec = np.array(metrics["tokens_per_sec"])
    rtf_values = np.array(metrics["rtf"])
    
    # Calculate statistics using cleaned data
    median_tps = float(np.median(tokens_per_sec))
    mean_tps = float(np.mean(tokens_per_sec))
    std_tps = float(np.std(tokens_per_sec))
    
    # Set y-axis limits based on data range
    y_min = max(0, np.min(tokens_per_sec) * 0.9)
    y_max = np.max(tokens_per_sec) * 1.1
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Plot data points
    chunk_nums = list(range(1, len(tokens_per_sec) + 1))
    
    # Plot data points
    ax.bar(chunk_nums, tokens_per_sec, color='#ff2a6d', alpha=0.6)
    
    # Set y-axis limits with padding
    padding = 0.1 * (y_max - y_min)
    ax.set_ylim(max(0, y_min - padding), y_max + padding)
    
    # Add median line
    ax.axhline(y=median_tps, color='#05d9e8', linestyle='--', 
              label=f'Median: {median_tps:.1f} tokens/sec')
    
    # Style improvements
    ax.set_xlabel('Chunk Number', fontsize=24, labelpad=20, color='white')
    ax.set_ylabel('Tokens per Second', fontsize=24, labelpad=20, color='white')
    ax.set_title('Processing Speed by Chunk', fontsize=28, pad=30, color='white')
    ax.tick_params(axis='both', which='major', labelsize=20, colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.grid(False)
    ax.legend(fontsize=20, facecolor='black', edgecolor='#05d9e8', loc='lower left', 
             labelcolor='white')
    
    plt.tight_layout()
    
    # Calculate average RTF from individual chunk RTFs
    rtf = np.mean(rtf_values)
    
    # Prepare metrics text
    metrics_text = (
        f"Median Speed: {median_tps:.1f} tokens/sec (o200k_base)\n" +
        f"Real-time Factor: {rtf:.3f}\n" +
        f"Real Time Speed: {int(1/rtf)}x\n" +
        f"Processing Time: {int(metrics['total_time'])}s\n" +
        f"Total Tokens: {metrics['total_tokens']} (o200k_base)\n" +
        f"Voices: {', '.join(voice_names)}"
    )
    
    return fig, metrics_text


# Create Gradio interface
with gr.Blocks(title="Kokoro TTS Demo", css="""
    .equal-height {
        min-height: 400px;
        display: flex;
        flex-direction: column;
    }
    .token-label {
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    .token-count {
        color: #4169e1;
    }
""") as demo:
    gr.HTML(header_html)
    
    with gr.Row():
        # Column 1: Text Input and Book Selection
        with gr.Column(elem_classes="equal-height"):
            # Book selection
            books = get_available_books()
            book_dropdown = gr.Dropdown(
                label="Select Book",
                choices=[book['label'] for book in books],
                value=books[0]['label'] if books else None,
                type="value",
                allow_custom_value=True
            )
            
            # Initialize chapters for first book
            initial_book = books[0]['value'] if books else None
            initial_chapters = []
            if initial_book:
                book_path = os.path.join("texts/processed", initial_book)
                _, chapters = get_book_info(book_path)
                initial_chapters = [ch['title'] for ch in chapters]
            
            # Chapter selection with initial chapters
            chapter_dropdown = gr.Dropdown(
                label="Select Chapter",
                choices=initial_chapters,
                value=initial_chapters[0] if initial_chapters else None,
                type="value",
                allow_custom_value=True
            )
            lab_tps = 175
            lab_rts = 50
            # Text input area with initial chapter text
            initial_text = ""
            if initial_chapters and initial_book:
                book_path = os.path.join("texts/processed", initial_book)
                _, chapters = get_book_info(book_path)
                if chapters:
                    initial_text = get_chapter_text(book_path, chapters[0]['id'])
                    tokens = count_tokens(initial_text)
                    time_estimate = math.ceil(tokens / lab_tps)
                    output_estimate = (time_estimate * lab_rts)//60
                    initial_label = f'<div class="token-label"><span class="token-count">Estimated {output_estimate} minutes in ~{time_estimate}s</span></div>'
                else:
                    initial_label = '<div class="token-label"></div>'
            else:
                initial_label = '<div class="token-label"></div>'
            
            def update_text_label(text):
                if not text:
                    return '<div class="token-label"></div>'
                tokens = count_tokens(text)
                time_estimate = math.ceil(tokens / lab_tps)
                output_estimate = (time_estimate * lab_rts)//60 
                return  f'<div class="token-label"><span class="token-count">Estimated {output_estimate} minutes in ~{time_estimate}s</span></div>'

            
            text_input = gr.TextArea(
                label=None,
                placeholder="Enter text here, select a chapter, or upload a .txt file",
                value=initial_text,
                lines=8,
                show_label=False,
                show_copy_button=True  # Add copy button for convenience
            )
            
            clear_btn = gr.Button("Clear Text", variant="secondary")
            label_html = gr.HTML(initial_label)
            
            def clear_text():
                return "", '<div class="token-label"></div>'
            
            clear_btn.click(
                fn=clear_text,
                outputs=[text_input, label_html]
            )
            
            # Update label whenever text changes
            text_input.change(
                fn=update_text_label,
                inputs=[text_input],
                outputs=[label_html],
                trigger_mode="always_last"
            )
            
            def update_chapters(book_name):
                if not book_name:
                    return gr.update(choices=[], value=None), "", '<div class="token-label"></div>'
                # Find the corresponding book file
                book_file = next((book['value'] for book in books if book['label'] == book_name), None)
                if not book_file:
                    return gr.update(choices=[], value=None), "", '<div class="token-label"></div>'
                book_path = os.path.join("texts/processed", book_file)
                book_title, chapters = get_book_info(book_path)
                # Create simple choices list of chapter titles
                chapter_choices = [ch['title'] for ch in chapters]
                # Set initial chapter text when book is selected
                initial_text = get_chapter_text(book_path, chapters[0]['id']) if chapters else ""
                if initial_text:
                    tokens = count_tokens(initial_text)
                    time_estimate = math.ceil(tokens / 150 / 10) * 10
                    label = f'<div class="token-label"><span class="token-count">({tokens} tokens, ~{time_estimate}s generation time)</span></div>'
                else:
                    label = '<div class="token-label"></div>'
                return gr.update(choices=chapter_choices, value=chapter_choices[0] if chapter_choices else None), initial_text, label
            
            def load_chapter_text(book_name, chapter_title):
                if not book_name or not chapter_title:
                    return "", '<div class="token-label"></div>'
                # Find the corresponding book file
                book_file = next((book['value'] for book in books if book['label'] == book_name), None)
                if not book_file:
                    return "", '<div class="token-label"></div>'
                book_path = os.path.join("texts/processed", book_file)
                # Get all chapters and find the one matching the title
                _, chapters = get_book_info(book_path)
                for ch in chapters:
                    if ch['title'] == chapter_title:
                        text = get_chapter_text(book_path, ch['id'])
                        tokens = count_tokens(text)
                        time_estimate = math.ceil(tokens / 150 / 10) * 10
                        return text, f'<div class="token-label"> <span class="token-count">({tokens} tokens, ~{time_estimate}s generation time)</span></div>'
                return "", '<div class="token-label"></div>'
            
            # Set up event handlers for book/chapter selection
            book_dropdown.change(
                fn=update_chapters,
                inputs=[book_dropdown],
                outputs=[chapter_dropdown, text_input, label_html]
            )
            
            chapter_dropdown.change(
                fn=load_chapter_text,
                inputs=[book_dropdown, chapter_dropdown],
                outputs=[text_input, label_html]
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
                    return None, '<div class="token-label"></div>'
                try:
                    text = file_bytes.decode('utf-8')
                    tokens = count_tokens(text)
                    time_estimate = math.ceil(tokens / 150 / 10) * 10  # Round up to nearest 10 seconds
                    return text, f'<div class="token-label"><span class="token-count">({tokens} tokens, ~{time_estimate}s generation time)</span></div>'
                except Exception as e:
                    raise gr.Error(f"Failed to read file: {str(e)}")

            file_input.change(
                fn=load_text_from_file,
                inputs=[file_input],
                outputs=[text_input, label_html]
            )
            
            with gr.Group():
                voice_dropdown = gr.Dropdown(
                    label="Voice(s)",
                    choices=[],  # Start empty, will be populated after initialization
                    value=None,
                    allow_custom_value=True,
                    multiselect=True
                )
                
                # Add refresh button to manually update voice list
                refresh_btn = gr.Button("🔄 Refresh Voices", size="sm")
                
                speed_slider = gr.Slider(
                    label="Speed",
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1
                )
                gpu_timeout_slider = gr.Slider(
                    label="GPU Timeout (seconds)",
                    minimum=15,
                    maximum=120,
                    value=90,
                    step=1,
                    info="Maximum time allowed for GPU processing"
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
                lines=5
            )
            metrics_plot = gr.Plot(
                label="Processing Metrics",
                show_label=True,
                format="png"  # Explicitly set format to PNG which is supported by matplotlib
            )
    
    # Set up event handlers
    refresh_btn.click(
        fn=initialize_model,
        outputs=[voice_dropdown]
    )
    
    submit_btn.click(
        fn=generate_speech_from_ui,
        inputs=[text_input, voice_dropdown, speed_slider, gpu_timeout_slider],
        outputs=[audio_output, metrics_plot, metrics_text],
        show_progress=True
    )
    
    # Add text analysis info
    with gr.Row():
        with gr.Column():
            gr.Markdown(demo_text_info)
    
    # Initialize voices on load
    demo.load(
        fn=initialize_model,
        outputs=[voice_dropdown]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
