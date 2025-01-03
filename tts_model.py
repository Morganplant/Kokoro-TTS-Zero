import os
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Tuple, List
from statistics import mean, median, stdev
from lib import (
    normalize_text,
    chunk_text,
    count_tokens,
    load_module_from_file,
    download_model_files,
    list_voice_files,
    download_voice_files,
    ensure_dir,
    concatenate_audio_chunks
)

class TTSModel:
    """GPU-accelerated TTS model manager"""
    
    def __init__(self):
        self.model = None
        self.voices_dir = "voices"
        self.model_repo = "hexgrad/Kokoro-82M"
        ensure_dir(self.voices_dir)
        
        # Load required modules
        py_modules = ["istftnet", "plbert", "models", "kokoro"]
        module_files = download_model_files(self.model_repo, [f"{m}.py" for m in py_modules])
        
        for module_name, file_path in zip(py_modules, module_files):
            load_module_from_file(module_name, file_path)
        
        # Import required functions from kokoro module
        kokoro = __import__("kokoro")
        self.generate = kokoro.generate
        self.build_model = __import__("models").build_model
        
    def initialize(self) -> bool:
        """Initialize model and download voices"""
        try:
            print("Initializing model...")
            
            # Download model files
            model_files = download_model_files(
                self.model_repo,
                ["kokoro-v0_19.pth", "config.json"]
            )
            model_path = model_files[0]  # kokoro-v0_19.pth
            
            # Build model directly on GPU
            with torch.cuda.device(0):
                torch.cuda.set_device(0)
                self.model = self.build_model(model_path, 'cuda')
                self._model_on_gpu = True
            
            print("Model initialization complete")
            return True
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            return False
    
    def ensure_voice_downloaded(self, voice_name: str) -> bool:
        """Ensure specific voice is downloaded"""
        try:
            voice_path = os.path.join(self.voices_dir, f"{voice_name}.pt")
            if not os.path.exists(voice_path):
                print(f"Downloading voice {voice_name}.pt...")
                download_voice_files(self.model_repo, [f"{voice_name}.pt"], self.voices_dir)
            return True
        except Exception as e:
            print(f"Error downloading voice {voice_name}: {str(e)}")
            return False

    def list_voices(self) -> List[str]:
        """List available voices"""
        return [
            "af_bella", "af_nicole", "af_sarah", "af_sky", "af",
            "am_adam", "am_michael", "bf_emma", "bf_isabella",
            "bm_george", "bm_lewis"
        ]
    
    def _ensure_model_on_gpu(self) -> None:
        """Ensure model is on GPU and stays there"""
        if not hasattr(self, '_model_on_gpu') or not self._model_on_gpu:
            print("Moving model to GPU...")
            with torch.cuda.device(0):
                torch.cuda.set_device(0)
                if hasattr(self.model, 'to'):
                    self.model.to('cuda')
                else:
                    for name in self.model:
                        if isinstance(self.model[name], torch.Tensor):
                            self.model[name] = self.model[name].cuda()
                self._model_on_gpu = True
    
    def _generate_audio(self, text: str, voicepack: torch.Tensor, lang: str, speed: float) -> np.ndarray:
        """GPU-accelerated audio generation"""
        try:
            with torch.cuda.device(0):
                torch.cuda.set_device(0)
                
                # Move everything to GPU in a single context
                if not hasattr(self, '_model_on_gpu') or not self._model_on_gpu:
                    print("Moving model to GPU...")
                    if hasattr(self.model, 'to'):
                        self.model.to('cuda')
                    else:
                        for name in self.model:
                            if isinstance(self.model[name], torch.Tensor):
                                self.model[name] = self.model[name].cuda()
                    self._model_on_gpu = True
                
                # Move voicepack to GPU
                voicepack = voicepack.cuda()
                
                # Run generation with everything on GPU
                audio, _ = self.generate(
                    self.model,
                    text,
                    voicepack,
                    lang=lang,
                    speed=speed
                )
                
                return audio
            
        except Exception as e:
            print(f"Error in audio generation: {str(e)}")
            raise e

    def generate_speech(self, text: str, voice_name: str, speed: float = 1.0, progress_callback=None) -> Tuple[np.ndarray, float]:
        """Generate speech from text. Returns (audio_array, duration)
        
        Args:
            text: Input text to convert to speech
            voice_name: Name of voice to use
            speed: Speech speed multiplier
            progress_callback: Optional callback function(chunk_num, total_chunks, tokens_per_sec, rtf)
        """
        try:
            if not text or not voice_name:
                raise ValueError("Text and voice name are required")
            
            start_time = time.time()
            
            # Count tokens and normalize text
            total_tokens = count_tokens(text)
            text = normalize_text(text)
            if not text:
                raise ValueError("Text is empty after normalization")
            
            # Load voice and process within GPU context
            with torch.cuda.device(0):
                torch.cuda.set_device(0)
                
                voice_path = os.path.join(self.voices_dir, f"{voice_name}.pt")
                
                # Ensure voice is downloaded and load directly to GPU
                if not self.ensure_voice_downloaded(voice_name):
                    raise ValueError(f"Failed to download voice: {voice_name}")
                voicepack = torch.load(voice_path, map_location='cuda', weights_only=True)
                
                # Break text into chunks for better memory management
                chunks = chunk_text(text)
                print(f"Processing {len(chunks)} chunks...")
                
                # Ensure model is initialized and on GPU
                if self.model is None:
                    print("Model not initialized, reinitializing...")
                    if not self.initialize():
                        raise ValueError("Failed to initialize model")
                
                # Move model to GPU if needed
                if not hasattr(self, '_model_on_gpu') or not self._model_on_gpu:
                    print("Moving model to GPU...")
                    if hasattr(self.model, 'to'):
                        self.model.to('cuda')
                    else:
                        for name in self.model:
                            if isinstance(self.model[name], torch.Tensor):
                                self.model[name] = self.model[name].cuda()
                    self._model_on_gpu = True
                
                # Process all chunks within same GPU context
                audio_chunks = []
                chunk_times = []
                chunk_sizes = []  # Store chunk lengths
                total_processed_tokens = 0
                total_processed_time = 0
                
                for i, chunk in enumerate(chunks):
                    chunk_start = time.time()
                    chunk_audio = self._generate_audio(
                        text=chunk,
                        voicepack=voicepack,
                        lang=voice_name[0],
                        speed=speed
                    )
                    chunk_time = time.time() - chunk_start
                    
                    # Update metrics
                    chunk_tokens = count_tokens(chunk)
                    total_processed_tokens += chunk_tokens
                    total_processed_time += chunk_time
                    current_tokens_per_sec = total_processed_tokens / total_processed_time
                    
                    # Calculate processing speed metrics
                    chunk_duration = len(chunk_audio) / 24000  # audio duration in seconds
                    rtf = chunk_time / chunk_duration
                    times_faster = 1 / rtf
                    
                    chunk_times.append(chunk_time)
                    chunk_sizes.append(len(chunk))
                    print(f"Chunk {i+1}/{len(chunks)} processed in {chunk_time:.2f}s")
                    print(f"Current tokens/sec: {current_tokens_per_sec:.2f}")
                    print(f"Real-time factor: {rtf:.2f}x")
                    print(f"{times_faster:.1f}x faster than real-time")
                    
                    audio_chunks.append(chunk_audio)
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(i + 1, len(chunks), current_tokens_per_sec, rtf)
            
            # Concatenate audio chunks
            audio = concatenate_audio_chunks(audio_chunks)
            
            def setup_plot(fig, ax, title):
                """Configure plot styling"""
                # Improve grid
                ax.grid(True, linestyle="--", alpha=0.3, color="#ffffff")
                
                # Set title and labels with better fonts and more padding
                ax.set_title(title, pad=40, fontsize=16, fontweight="bold", color="#ffffff")
                ax.set_xlabel(ax.get_xlabel(), fontsize=14, fontweight="medium", color="#ffffff")
                ax.set_ylabel(ax.get_ylabel(), fontsize=14, fontweight="medium", color="#ffffff")
                
                # Improve tick labels
                ax.tick_params(labelsize=12, colors="#ffffff")
                
                # Style spines
                for spine in ax.spines.values():
                    spine.set_color("#ffffff")
                    spine.set_alpha(0.3)
                    spine.set_linewidth(0.5)
                
                # Set background colors
                ax.set_facecolor("#1a1a2e")
                fig.patch.set_facecolor("#1a1a2e")
                
                return fig, ax

            # Set dark style
            plt.style.use("dark_background")
            
            # Create figure with subplots
            fig = plt.figure(figsize=(18, 16))
            fig.patch.set_facecolor("#1a1a2e")
            
            # Create subplot grid
            gs = plt.GridSpec(2, 1, left=0.15, right=0.85, top=0.9, bottom=0.15, hspace=0.4)
            
            # Processing times plot
            ax1 = plt.subplot(gs[0])
            chunks_x = list(range(1, len(chunks) + 1))
            bars = ax1.bar(chunks_x, chunk_times, color='#ff2a6d', alpha=0.8)
            
            # Add statistics lines
            mean_time = mean(chunk_times)
            median_time = median(chunk_times)
            std_time = stdev(chunk_times) if len(chunk_times) > 1 else 0
            
            ax1.axhline(y=mean_time, color='#05d9e8', linestyle='--', 
                       label=f'Mean: {mean_time:.2f}s')
            ax1.axhline(y=median_time, color='#d1f7ff', linestyle=':', 
                       label=f'Median: {median_time:.2f}s')
            
            # Add ±1 std dev range
            if len(chunk_times) > 1:
                ax1.axhspan(mean_time - std_time, mean_time + std_time, 
                          color='#8c1eff', alpha=0.2, label='±1 Std Dev')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f'{height:.2f}s',
                        ha='center',
                        va='bottom',
                        color='white',
                        fontsize=10)
            
            ax1.set_xlabel('Chunk Number')
            ax1.set_ylabel('Processing Time (seconds)')
            setup_plot(fig, ax1, 'Chunk Processing Times')
            ax1.legend(facecolor="#1a1a2e", edgecolor="#ffffff")
            
            # Chunk sizes plot
            ax2 = plt.subplot(gs[1])
            ax2.plot(chunks_x, chunk_sizes, color='#ff9e00', marker='o', linewidth=2)
            ax2.set_xlabel('Chunk Number')
            ax2.set_ylabel('Chunk Size (chars)')
            setup_plot(fig, ax2, 'Chunk Sizes')
            
            # Save plot
            plt.savefig('chunk_times.png', format='png')
            plt.close()
            
            # Calculate metrics
            total_time = time.time() - start_time
            tokens_per_second = total_tokens / total_time
            
            print(f"\nProcessing Metrics:")
            print(f"Total tokens: {total_tokens}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Tokens per second: {tokens_per_second:.2f}")
            print(f"Mean chunk time: {mean_time:.2f}s")
            print(f"Median chunk time: {median_time:.2f}s")
            if len(chunk_times) > 1:
                print(f"Std dev: {std_time:.2f}s")
            print(f"\nChunk time plot saved as 'chunk_times.png'")
            
            return audio, len(audio) / 24000  # Return audio array and duration
            
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            raise
