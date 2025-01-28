import os
import torch
import numpy as np
import time
from typing import Tuple, List
import soundfile as sf
from kokoro import KPipeline
import spaces
from lib.file_utils import download_voice_files, ensure_dir

class TTSModelV1:
    """KPipeline-based TTS model for v1.0.0"""
    
    def __init__(self):
        self.pipeline = None
        self.model_repo = "hexgrad/Kokoro-82M"
        # Use v1 voices from Kokoro-82M repo
        self.voices_dir = os.path.join(os.path.dirname(__file__), "voices")
        
    def initialize(self) -> bool:
        """Initialize KPipeline and verify voices"""
        try:
            print("Initializing v1.0.0 model...")
            
            self.pipeline = None # cannot be initialized outside of GPU decorator
            
            # Download v1 voices if needed
            ensure_dir(self.voices_dir)
            if not os.path.exists(os.path.join(self.voices_dir, "voices")):
                print("Downloading v1 voices...")
                download_voice_files(self.model_repo, "voices", self.voices_dir)

            # Verify voices were downloaded successfully
            available_voices = self.list_voices()
            if not available_voices:
                print("Warning: No voices found after initialization")
            else:
                print(f"Found {len(available_voices)} voices")
            
            print("Model initialization complete")
            return True
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            return False
    
    def list_voices(self) -> List[str]:
        """List available voices"""
        voices = []
        voices_dir = os.path.join(self.voices_dir, "voices")
        if os.path.exists(voices_dir):
            for file in os.listdir(voices_dir):
                if file.endswith(".pt"):
                    voice_name = file[:-3]
                    voices.append(voice_name)
        return voices
        
    @spaces.GPU(duration=None)  # Duration will be set by the UI
    def generate_speech(self, text: str, voice_names: list[str], speed: float = 1.0, gpu_timeout: int = 60, progress_callback=None, progress_state=None, progress=None) -> Tuple[np.ndarray, float]:
        """Generate speech from text using KPipeline
        
        Args:
            text: Input text to convert to speech
            voice_names: List of voice names to use (will be mixed if multiple)
            speed: Speech speed multiplier
            progress_callback: Optional callback function
            progress_state: Dictionary tracking generation progress metrics
            progress: Progress callback from Gradio
        """
        try:
            start_time = time.time()
            if self.pipeline is None:
                lang_code = voice_names[0][0] if voice_names else 'a'
                self.pipeline = KPipeline(lang_code=lang_code)
                
            if not text or not voice_names:
                raise ValueError("Text and voice name are required")
            
            # Handle voice mixing
            if isinstance(voice_names, list) and len(voice_names) > 1:
                t_voices = []
                for voice in voice_names:
                    try:
                        voice_path = os.path.join(self.voices_dir, "voices", f"{voice}.pt")
                        try:
                            voicepack = torch.load(voice_path, weights_only=True)
                        except Exception as e:
                            print(f"Warning: weights_only load failed, attempting full load: {str(e)}")
                            voicepack = torch.load(voice_path, weights_only=False)
                        t_voices.append(voicepack)
                    except Exception as e:
                        print(f"Warning: Failed to load voice {voice}: {str(e)}")
                
                # Combine voices by taking mean
                voicepack = torch.mean(torch.stack(t_voices), dim=0)
                voice_name = "_".join(voice_names)
                # Save mixed voice temporarily
                mixed_voice_path = os.path.join(self.voices_dir, "voices", f"{voice_name}.pt")
                torch.save(voicepack, mixed_voice_path)
            else:
                voice_name = voice_names[0]
                voice_path = os.path.join(self.voices_dir, "voices", f"{voice_name}.pt")
                try:
                    voicepack = torch.load(voice_path, weights_only=True)
                except Exception as e:
                    print(f"Warning: weights_only load failed, attempting full load: {str(e)}")
                    voicepack = torch.load(voice_path, weights_only=False)
            
            # Initialize tracking
            audio_chunks = []
            chunk_times = []
            chunk_sizes = []
            total_tokens = 0
            
            # Get generator from pipeline
            generator = self.pipeline(
                text,
                voice=voice_name,
                speed=speed,
                split_pattern=r'\n+'  # Default chunking pattern
            )
            
            # Process chunks
            for i, (gs, ps, audio) in enumerate(generator):
                chunk_start = time.time()
                audio_chunks.append(audio)
                
                # Calculate metrics
                chunk_time = time.time() - chunk_start
                chunk_tokens = len(gs)
                total_tokens += chunk_tokens
                
                # Calculate speed metrics
                chunk_duration = len(audio) / 24000
                rtf = chunk_time / chunk_duration
                chunk_tokens_per_sec = chunk_tokens / chunk_time
                
                chunk_times.append(chunk_time)
                chunk_sizes.append(len(gs))
                
                print(f"Chunk {i+1} processed in {chunk_time:.2f}s")
                print(f"Current tokens/sec: {chunk_tokens_per_sec:.2f}")
                print(f"Real-time factor: {rtf:.2f}x")
                print(f"{(1/rtf):.1f}x faster than real-time")
                
                # Update progress
                if progress_callback and progress_state:
                    # Initialize lists if needed
                    if "tokens_per_sec" not in progress_state:
                        progress_state["tokens_per_sec"] = []
                    if "rtf" not in progress_state:
                        progress_state["rtf"] = []
                    if "chunk_times" not in progress_state:
                        progress_state["chunk_times"] = []
                    
                    # Update progress state
                    progress_state["tokens_per_sec"].append(chunk_tokens_per_sec)
                    progress_state["rtf"].append(rtf)
                    progress_state["chunk_times"].append(chunk_time)
                    
                    progress_callback(
                        i + 1,
                        -1,  # Let UI handle total chunks
                        chunk_tokens_per_sec,
                        rtf,
                        progress_state,
                        start_time,
                        gpu_timeout,
                        progress
                    )
            
            # Concatenate audio chunks
            audio = np.concatenate(audio_chunks)
            
            # Cleanup temporary mixed voice if created
            if len(voice_names) > 1:
                try:
                    os.remove(mixed_voice_path)
                except:
                    pass
            
            # Return audio and metrics
            return (
                audio,
                len(audio) / 24000,
                {
                    "chunk_times": chunk_times,
                    "chunk_sizes": chunk_sizes,
                    "tokens_per_sec": [float(x) for x in progress_state["tokens_per_sec"]] if progress_state else [],
                    "rtf": [float(x) for x in progress_state["rtf"]] if progress_state else [],
                    "total_tokens": total_tokens,
                    "total_time": time.time() - start_time
                }
            )
            
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            raise
