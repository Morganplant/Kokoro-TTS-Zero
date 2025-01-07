# """Mock TTS implementation for local development"""
# import numpy as np

# class MockTTSModel:
#     def __init__(self):
#         self.model = None
        
#     def initialize(self):
#         """Mock initialization"""
#         self.model = "mock_model"
#         return True
        
#     def list_voices(self):
#         """Return mock list of voices"""
#         return ["mock_voice_1", "mock_voice_2"]
        
#     def generate_speech(self, text, voice_names, speed, gpu_timeout=90, progress_callback=None, progress_state=None, progress=None):
#         """Generate mock audio data"""
#         # Create mock audio data (1 second of silence)
#         sample_rate = 22050
#         duration = 1.0
#         t = np.linspace(0, duration, int(sample_rate * duration))
#         audio_array = np.zeros_like(t)
        
#         # Mock metrics
#         metrics = {
#             "tokens_per_sec": [10.5, 11.2, 10.8],
#             "rtf": [0.5, 0.48, 0.52],
#             "total_time": 3,
#             "total_tokens": 100
#         }
        
#         # Simulate progress updates
#         if progress_callback and progress_state and progress:
#             for i in range(3):
#                 progress_callback(i+1, 3, metrics["tokens_per_sec"][i], 
#                                 metrics["rtf"][i], progress_state, 
#                                 progress_state.get("start_time", 0),
#                                 gpu_timeout, progress)
        
#         return audio_array, duration, metrics
