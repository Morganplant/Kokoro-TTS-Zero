import os
import importlib.util
import sys
from huggingface_hub import hf_hub_download
from typing import List, Optional

def load_module_from_file(module_name: str, file_path: str):
    """Load a Python module from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def download_model_files(repo_id: str, filenames: List[str], local_dir: Optional[str] = None) -> List[str]:
    """Download multiple files from Hugging Face Hub"""
    paths = []
    for filename in filenames:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            paths.append(path)
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
            raise
    return paths

def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't"""
    os.makedirs(path, exist_ok=True)

def list_voice_files(voices_dir: str) -> List[str]:
    """List available voice files in directory"""
    voices = []
    try:
        if not os.path.exists(voices_dir):
            print(f"Voices directory does not exist: {voices_dir}")
            return voices
            
        files = os.listdir(voices_dir)
        print(f"Found {len(files)} files in voices directory")
        
        for file in files:
            if file.endswith(".pt"):
                voice_name = file[:-3]  # Remove .pt extension
                print(f"Found voice: {voice_name}")
                voices.append(voice_name)
                
        if not voices:
            print("No voice files found in voices directory")
            
    except Exception as e:
        print(f"Error listing voices: {str(e)}")
        import traceback
        traceback.print_exc()
        
    return sorted(voices)

def download_voice_files(repo_id: str, voices: List[str], voices_dir: str) -> None:
    """Download voice files from Hugging Face Hub"""
    ensure_dir(voices_dir)
    
    for voice in voices:
        try:
            voice_path = os.path.join(voices_dir, voice)
            print(f"Attempting to download voice {voice} to {voice_path}")
            
            try:
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"voices/{voice}",
                    local_dir=voices_dir,
                    local_dir_use_symlinks=False,
                    force_filename=voice
                )
                print(f"Download completed to: {downloaded_path}")
                
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
