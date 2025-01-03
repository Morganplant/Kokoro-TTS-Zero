import os
import io
import spaces
import torch
import numpy as np
import time
import tiktoken
import scipy.io.wavfile as wavfile
from huggingface_hub import hf_hub_download
import importlib.util
import sys

def load_module_from_file(module_name, file_path):
    """Load a Python module from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Download and load required Python modules
py_modules = ["istftnet", "plbert", "models"]
for py_module in py_modules:
    path = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename=f"{py_module}.py")
    load_module_from_file(py_module, path)

# Load the kokoro module
kokoro_path = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="kokoro.py")
kokoro = load_module_from_file("kokoro", kokoro_path)

# Import required functions
generate = kokoro.generate
normalize_text = kokoro.normalize_text
models = sys.modules['models']
build_model = models.build_model

# Set HF_HOME for faster restarts
os.environ["HF_HOME"] = "/data/.huggingface"

class TTSModel:
    """Self-contained TTS model manager for Hugging Face Spaces"""
    
    def __init__(self):
        self.model = None
        self.voices_dir = "voices"
        self.model_repo = "hexgrad/Kokoro-82M"
        os.makedirs(self.voices_dir, exist_ok=True)
        
    def initialize(self):
        """Initialize model and download voices"""
        try:
            print("Initializing model...")
            
            # Download model and config
            model_path = hf_hub_download(
                repo_id=self.model_repo,
                filename="kokoro-v0_19.pth"
            )
            config_path = hf_hub_download(
                repo_id=self.model_repo,
                filename="config.json"
            )
            
            # Build model directly on GPU if available
            with torch.cuda.device(0):
                torch.cuda.set_device(0)
                self.model = build_model(model_path, 'cuda')
                self._model_on_gpu = True
            
            # Download all available voices
            voices = [
                "af_bella.pt", "af_nicole.pt", "af_sarah.pt", "af_sky.pt", "af.pt",
                "am_adam.pt", "am_michael.pt",
                "bf_emma.pt", "bf_isabella.pt",
                "bm_george.pt", "bm_lewis.pt"
            ]
            for voice in voices:
                try:
                    # Download voice file
                    # Create full destination path
                    voice_path = os.path.join(self.voices_dir, voice)
                    print(f"Attempting to download voice {voice} to {voice_path}")
                    
                    # Ensure directory exists
                    os.makedirs(self.voices_dir, exist_ok=True)
                    
                    # Download with explicit destination
                    try:
                        downloaded_path = hf_hub_download(
                            repo_id=self.model_repo,
                            filename=f"voices/{voice}",
                            local_dir=self.voices_dir,
                            local_dir_use_symlinks=False,
                            force_filename=voice
                        )
                        print(f"Download completed to: {downloaded_path}")
                        
                        # Verify file exists
                        if not os.path.exists(voice_path):
                            print(f"Warning: File not found at expected path {voice_path}")
                            print(f"Checking download location: {downloaded_path}")
                            if os.path.exists(downloaded_path):
                                print(f"Moving file from {downloaded_path} to {voice_path}")
                                os.rename(downloaded_path, voice_path)
                        else:
                            print(f"Verified voice file exists: {voice_path}")
                            
                    except Exception as e:
                        print(f"Error downloading voice {voice}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        
                except Exception as e:
                    print(f"Error downloading voice {voice}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            print("Model initialization complete")
            return True
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            return False
    
    def list_voices(self):
        """List available voices"""
        voices = []
        try:
            # Verify voices directory exists
            if not os.path.exists(self.voices_dir):
                print(f"Voices directory does not exist: {self.voices_dir}")
                return voices
                
            # Get list of files
            files = os.listdir(self.voices_dir)
            print(f"Found {len(files)} files in voices directory")
            
            # Filter for .pt files
            for file in files:
                if file.endswith(".pt"):
                    voices.append(file[:-3])  # Remove .pt extension
                    print(f"Found voice: {file[:-3]}")
                    
            if not voices:
                print("No voice files found in voices directory")
                
        except Exception as e:
            print(f"Error listing voices: {str(e)}")
            import traceback
            traceback.print_exc()
            
        return sorted(voices)
    
    def _ensure_model_on_gpu(self):
        """Ensure model is on GPU and stays there"""
        if not hasattr(self, '_model_on_gpu') or not self._model_on_gpu:
            print("Moving model to GPU...")
            with torch.cuda.device(0):
                torch.cuda.set_device(0)
                # Move model to GPU using torch.nn.Module method
                if hasattr(self.model, 'to'):
                    self.model.to('cuda')
                else:
                    # Fallback for Munch object - move parameters individually
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
                audio, _ = generate(
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
    
    def chunk_text(self, text: str, max_chars: int = 300) -> list[str]:
        """Break text into chunks at natural boundaries"""
        chunks = []
        current_chunk = ""
        
        # Split on sentence boundaries first
        sentences = text.replace(".", ".|").replace("!", "!|").replace("?", "?|").replace(";", ";|").split("|")
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # If sentence is already too long, break on commas
            if len(sentence) > max_chars:
                parts = sentence.split(",")
                for part in parts:
                    if len(current_chunk) + len(part) <= max_chars:
                        current_chunk += part + ","
                    else:
                        # If part is still too long, break on whitespace
                        if len(part) > max_chars:
                            words = part.split()
                            for word in words:
                                if len(current_chunk) + len(word) > max_chars:
                                    chunks.append(current_chunk.strip())
                                    current_chunk = word + " "
                                else:
                                    current_chunk += word + " "
                        else:
                            chunks.append(current_chunk.strip())
                            current_chunk = part + ","
            else:
                if len(current_chunk) + len(sentence) <= max_chars:
                    current_chunk += sentence
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def generate_speech(self, text: str, voice_name: str, speed: float = 1.0) -> tuple[np.ndarray, float]:
        """Generate speech from text. Returns (audio_array, duration)"""
        try:
            if not text or not voice_name:
                raise ValueError("Text and voice name are required")
            
            start_time = time.time()
            
            # Initialize tokenizer
            enc = tiktoken.get_encoding("cl100k_base")
            total_tokens = len(enc.encode(text))
            
            # Normalize text
            text = normalize_text(text)
            if not text:
                raise ValueError("Text is empty after normalization")
            
            # Load voice and process within GPU context
            with torch.cuda.device(0):
                torch.cuda.set_device(0)
                
                voice_path = os.path.join(self.voices_dir, f"{voice_name}.pt")
                if not os.path.exists(voice_path):
                    raise ValueError(f"Voice not found: {voice_name}")
                
                # Load voice directly to GPU
                voicepack = torch.load(voice_path, map_location='cuda', weights_only=True)
                
                # Break text into chunks for better memory management
                chunks = self.chunk_text(text)
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
            for i, chunk in enumerate(chunks):
                chunk_start = time.time()
                chunk_audio = self._generate_audio(
                    text=chunk,
                    voicepack=voicepack,
                    lang=voice_name[0],
                    speed=speed
                )
                chunk_time = time.time() - chunk_start
                print(f"Chunk {i+1}/{len(chunks)} processed in {chunk_time:.2f}s")
                audio_chunks.append(chunk_audio)
            
            # Concatenate audio chunks
            audio = np.concatenate(audio_chunks)
            
            # Calculate metrics
            total_time = time.time() - start_time
            tokens_per_second = total_tokens / total_time
            
            print(f"\nProcessing Metrics:")
            print(f"Total tokens: {total_tokens}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Tokens per second: {tokens_per_second:.2f}")
            
            return audio, len(audio) / 24000  # Return audio array and duration
            
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            raise
