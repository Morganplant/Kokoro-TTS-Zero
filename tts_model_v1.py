import os
import torch
import numpy as np
import time
from typing import Tuple, List
import soundfile as sf
from kokoro import KPipeline
import spaces

class TTSModelV1:
    """KPipeline-based TTS model for v1.0.0"""
    
    def __init__(self):
        self.pipeline = None
        self.voices_dir = "voices"
        self.model_repo = "hexgrad/Kokoro-82M"
        
    def initialize(self) -> bool:
        """Initialize KPipeline and verify voices"""
        try:
            print("Initializing v1.0.0 model...")
            
            # Initialize KPipeline with American English
            self.pipeline = None
            
            # Verify local voice files are available
            voices_dir = os.path.join(self.voices_dir, "voices")
            if not os.path.exists(voices_dir):
                raise ValueError("Voice files not found")

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
        voices_subdir = os.path.join(self.voices_dir, "voices")
        if os.path.exists(voices_subdir):
            for file in os.listdir(voices_subdir):
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
                self.pipeline = KPipeline(lang_code='a')
                
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
            
            # Generate speech using KPipeline
            generator = self.pipeline(
                text,
                voice=voice_name,
                speed=speed,
                split_pattern=r'\n+'  # Default chunking pattern
            )
            
            # Process chunks and collect metrics
            audio_chunks = []
            chunk_times = []
            chunk_sizes = []
            total_tokens = 0
            
            for i, (gs, ps, audio) in enumerate(generator):
                chunk_start = time.time()
                
                # Store chunk audio
                audio_chunks.append(audio)
                
                # Calculate metrics
                chunk_time = time.time() - chunk_start
                chunk_times.append(chunk_time)
                chunk_sizes.append(len(gs))  # Use grapheme length as chunk size
                
                # Update progress if callback provided
                if progress_callback:
                    chunk_duration = len(audio) / 24000
                    rtf = chunk_time / chunk_duration
                    progress_callback(
                        i + 1,
                        -1,  # Total chunks unknown with generator
                        len(gs) / chunk_time,  # tokens/sec
                        rtf,
                        progress_state,
                        start_time,
                        gpu_timeout,
                        progress
                    )
                
                print(f"Chunk {i+1} processed in {chunk_time:.2f}s")
                print(f"Graphemes: {gs}")
                print(f"Phonemes: {ps}")
            
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
