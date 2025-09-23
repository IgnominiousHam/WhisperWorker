#!/usr/bin/env python3
import logging
import os
import queue
import signal
import sys
import tempfile
import threading
import time
import traceback
import nanoid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import uuid
import requests
from faster_whisper import WhisperModel
import re
import numpy as np


# Try to import speaker processing dependencies
try:
    import torch
    import torchaudio
    from pyannote.audio import Pipeline, Model, Inference
    from sklearn.preprocessing import normalize
    HAS_SPEAKER_DEPS = True
except ImportError:
    HAS_SPEAKER_DEPS = False

class WorkerClient:
    def __init__(self, server_url: str, worker_name: str, max_concurrent: int = 4, username: str = None, password: str = None, mode: str = "transcription", hf_token: Optional[str] = None):
        self.server_url = server_url.rstrip('/')
        self.worker_name = worker_name
        self.worker_id = nanoid.generate(size=10)
        self.max_concurrent = max_concurrent  # Maximum concurrent downloads/processing
        self.mode = mode  # "transcription" or "embedding"
        self.running = False
        self.whisper_model = None
        self.speaker_models = {}  # For speaker embedding models
        self._model_lock = threading.Lock()  # Protect model access
        self._task_queue = queue.Queue(maxsize=self.max_concurrent * 2)  # Buffer for tasks
        self._active_workers = 0
        self._workers_lock = threading.Lock()
        self.username = username
        self.password = password
        # Prefer explicit token, else environment
        self.hf_token = hf_token

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                # Enable file logging if needed
                # logging.FileHandler(f'worker_{self.worker_id}.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
    def initialize_whisper(self):
        """Initialize the Whisper model"""
        try:
            self.logger.info("Initializing Whisper model...")
            # Try to use GPU if available
            device = "cuda" if self._check_cuda() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            self.logger.info(f"Using device: {device}, compute_type: {compute_type}")
            self.whisper_model = WhisperModel(
                "small",  # You can make this configurable
                device=device,
                compute_type=compute_type
            )
            self.logger.info("Whisper model initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Whisper model: {e}")
            return False

    def initialize_speaker_models(self):
        """Initialize speaker embedding models"""
        if not HAS_SPEAKER_DEPS:
            self.logger.error("Speaker processing dependencies not available")
            return False
            
        try:
            self.logger.info("Initializing speaker embedding models...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {device}")
            if not self.hf_token:
                self.logger.warning("HUGGINGFACE_TOKEN not set. pyannote models require an access token. Provide via .env or --hf-token.")
            
            # Initialize diarization pipeline
            try:
                # Try latest stable first
                self.speaker_models['diarization'] = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization",
                    use_auth_token=self.hf_token
                )
                self.logger.info("Diarization pipeline initialized (pyannote/speaker-diarization)")
            except Exception as e:
                self.logger.warning(f"Failed to initialize diarization pipeline 'pyannote/speaker-diarization': {e}")
                # Fallback to a specific version
                try:
                    self.speaker_models['diarization'] = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=self.hf_token
                    )
                    self.logger.info("Diarization pipeline initialized (pyannote/speaker-diarization-3.1)")
                except Exception as e2:
                    self.logger.warning(f"Failed fallback diarization pipeline 'pyannote/speaker-diarization-3.1': {e2}")
                    self.speaker_models['diarization'] = None
            
            # Initialize embedding model
            try:
                embedding_model = Model.from_pretrained(
                    "pyannote/embedding", 
                    use_auth_token=self.hf_token
                )
                self.speaker_models['embedding'] = Inference(embedding_model, window="whole")
                self.logger.info("Embedding model initialized (pyannote/embedding)")
            except Exception as e:
                self.logger.warning(f"Failed to initialize embedding model: {e}")
                self.speaker_models['embedding'] = None
            
            if not any(self.speaker_models.values()):
                self.logger.error("No speaker models could be initialized. Ensure pyannote.audio deps are installed and HUGGINGFACE_TOKEN is valid.")
                return False
                
            self.logger.info("Speaker models initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize speaker models: {e}")
            return False
            
    def preprocess_audio_for_speaker(self, audio_path, target_sample_rate=16000):
        """Load and preprocess audio for speaker processing"""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to target sample rate if needed
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=target_sample_rate
                )
                waveform = resampler(waveform)
            
            return waveform, target_sample_rate
        except Exception as e:
            self.logger.error(f"Error preprocessing audio: {e}")
            return None, None

    def process_speaker_embeddings(self, file_path: str, filename: str) -> Optional[Dict]:
        """Process audio file for speaker embeddings"""
        try:
            self.logger.info(f"Processing speaker embeddings for: {filename}")
            
            if not self.speaker_models.get('diarization') or not self.speaker_models.get('embedding'):
                self.logger.error("Speaker models not properly initialized")
                if not self.speaker_models.get('diarization'):
                    self.logger.error("Diarization pipeline unavailable. Requires Hugging Face token and pyannote.audio.")
                if not self.speaker_models.get('embedding'):
                    self.logger.error("Embedding model unavailable. Requires Hugging Face token and pyannote.audio.")
                return None
            
            # Preprocess audio
            waveform, sample_rate = self.preprocess_audio_for_speaker(file_path)
            if waveform is None:
                return None
            
            # Create audio dict for pyannote
            audio_dict = {"waveform": waveform, "sample_rate": sample_rate}
            
            with self._model_lock:
                # Get diarization results
                diarization = self.speaker_models['diarization'](audio_dict)
                
                # Collect embeddings per speaker
                speaker_embeddings = {}
                timeline_data = []
                
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    duration = turn.end - turn.start
                    if duration < 1.0:  # Skip segments shorter than 1 second
                        continue
                    
                    # Store timeline information
                    timeline_data.append({
                        'start': round(turn.start, 2),
                        'end': round(turn.end, 2),
                        'duration': round(duration, 2),
                        'local_speaker': speaker
                    })
                    
                    try:
                        # Extract segment for embedding
                        start_sample = int(turn.start * sample_rate)
                        end_sample = int(turn.end * sample_rate)
                        segment_waveform = waveform[:, start_sample:end_sample]
                        
                        segment_audio = {"waveform": segment_waveform, "sample_rate": sample_rate}
                        segment_embedding = self.speaker_models['embedding'](segment_audio)
                        
                        if segment_embedding is not None and segment_embedding.size > 0:
                            if speaker not in speaker_embeddings:
                                speaker_embeddings[speaker] = []
                            speaker_embeddings[speaker].append(segment_embedding)
                            
                    except Exception as e:
                        self.logger.warning(f"Error processing segment for speaker {speaker}: {e}")
                        continue
            
            # Average embeddings per speaker
            file_embeddings = []
            local_speakers = []
            
            for local_speaker, embeddings in speaker_embeddings.items():
                if len(embeddings) > 0:
                    try:
                        mean_embedding = np.mean(np.vstack(embeddings), axis=0)
                        file_embeddings.append(mean_embedding.tolist())  # Convert to list for JSON
                        local_speakers.append(local_speaker)
                    except Exception as e:
                        self.logger.warning(f"Error averaging embeddings for speaker {local_speaker}: {e}")
                        continue
            
            if not file_embeddings:
                self.logger.warning(f"No valid embeddings extracted from {filename}")
                return None
            
            # Create result
            result = {
                "filename": filename,
                "embeddings": file_embeddings,
                "local_speakers": local_speakers,
                "timeline": timeline_data,
                "embedding_count": len(file_embeddings)
            }
            
            self.logger.info(f"Successfully extracted {len(file_embeddings)} embeddings from {filename}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing speaker embeddings: {e}")
            return None
            
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
            
    def register_worker(self) -> bool:
        """Register this worker with the server"""
        try:
            response = requests.post(
                f"{self.server_url}/api/workers/register",
                json={
                    "worker_id": self.worker_id,
                    "name": self.worker_name,
                    "mode": self.mode
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    self.logger.info(f"Successfully registered worker {self.worker_id} in {self.mode} mode")
                    return True
                else:
                    self.logger.error(f"Registration failed: {result.get('message')}")
                    return False
            else:
                self.logger.error(f"Registration failed with status {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error registering worker: {e}")
            return False
            
    def send_heartbeat(self) -> bool:
        """Send heartbeat to server"""
        try:
            response = requests.post(
                f"{self.server_url}/api/workers/heartbeat",
                json={"worker_id": self.worker_id},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("success", False)
            else:
                self.logger.warning(f"Heartbeat failed with status {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.warning(f"Error sending heartbeat: {e}")
            return False
            
    def task_producer(self):
        """Continuously fetch tasks from server and put them in the queue"""
        while self.running:
            try:
                # Only fetch new tasks if we have room in the queue
                if not self._task_queue.full():
                    task = self.get_task()
                    if task:
                        self._task_queue.put(task, timeout=1)
                        self.logger.debug(f"Added task {task['id']} to queue")
                    else:
                        # No tasks available, wait a bit before trying again
                        time.sleep(2)
                else:
                    # Queue is full, wait a bit
                    time.sleep(1)
            except queue.Full:
                # Queue is full, wait a bit
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in task producer: {e}")
                time.sleep(5)
    
    def process_task_worker(self):
        """Worker thread that processes individual tasks"""
        while self.running:
            try:
                # Get a task from the queue
                task = self._task_queue.get(timeout=5)
                
                with self._workers_lock:
                    self._active_workers += 1
                
                try:
                    self.logger.info(f"Worker processing task {task['id']}: {task['filename']}")
                    
                    # Download the file
                    task_id = task["id"]
                    file_path = task["file_path"]
                    filename = task["filename"]
                    
                    # Create temporary file for download
                    temp_file = tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False)
                    temp_path = temp_file.name
                    temp_file.close()
                    
                    try:
                        # Download the file
                        if self.download_file(file_path, temp_path):
                            # Process based on task type
                            task_type = task.get("task_type", "transcription")
                            
                            if task_type == "speaker_embedding":
                                # Process for speaker embeddings
                                result = self.process_speaker_embeddings(temp_path, filename)
                                if result:
                                    self.submit_embedding_result(task_id, True, result)
                                else:
                                    self.submit_embedding_result(task_id, False, error_message="No embeds detected")
                            else:
                                # Process for transcription (original behavior)
                                result = self.process_audio_file(temp_path, filename)
                                if result and self.is_coherent_transcript(result.get("transcript", "")):
                                    self.submit_result(task_id, True, result)
                                else:
                                    self.submit_result(task_id, True, None)
                        else:
                            # Download failed
                            task_type = task.get("task_type", "transcription")
                            if task_type == "speaker_embedding":
                                self.submit_embedding_result(task_id, False, error_message="Failed to download file")
                            else:
                                self.submit_result(task_id, False, error_message="Failed to download file")
                    
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                
                finally:
                    with self._workers_lock:
                        self._active_workers -= 1
                    self._task_queue.task_done()
                    
            except queue.Empty:
                # No tasks available, continue loop
                continue
            except Exception as e:
                self.logger.error(f"Error in worker thread: {e}")
                self.logger.error(traceback.format_exc())
                with self._workers_lock:
                    self._active_workers -= 1
                    
    def get_task(self) -> Optional[Dict]:
        """Get a task from the server"""
        try:
            # Request tasks based on worker mode
            endpoint = "/api/workers/get-task"
            if self.mode == "embedding":
                endpoint = "/api/tasks/embedding/get"
                
            response = requests.post(
                f"{self.server_url}{endpoint}",
                json={"worker_id": self.worker_id},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success") and result.get("task"):
                    return result["task"]
            else:
                self.logger.debug(f"No tasks available (status: {response.status_code})")
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting task: {e}")
            return None
            
    def download_file(self, file_path: str, local_path: str) -> bool:
        """Download file from server"""
        try:
            response = requests.get(
                f"{self.server_url}/download/{file_path}",
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            else:
                self.logger.error(f"Failed to download file: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error downloading file: {e}")
            return False
            
    def extract_file_info(self, filename):
        """Extract frequency, modulation, and box info from filename"""
        try:
            parts = filename.split('_')
            if len(parts) >= 4:
                frequency = parts[2]
                modulation = parts[3]
                box = os.path.splitext(parts[-1])[0]
                return frequency, modulation, box
        except Exception as e:
            self.logger.warning(f"Error parsing filename {filename}: {e}")
        return None, None, None
    
    def format_timestamp(self, seconds):
        """Format timestamp from seconds to MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    

    def is_coherent_transcript(self, text):
        """Return True if transcript is coherent (not blank, not repetitive, not gibberish, not just Unicode blocks, not sound effects)."""
        if not text or not text.strip():
            return False
        cleaned = text.strip()
        # Remove common timestamp prefixes like "[00:01-00:05] " so patterns match lines without timestamps
        # Operate on a timestamp-free copy for pattern matching
        plain = re.sub(r"\[\d{1,2}:\d{2}-\d{1,2}:\d{2}\]\s*", "", cleaned)
        # Exclude if a single phrase is repeated more than twice (e.g. 'I don't know what I'm talking about' repeated)
        phrase_counts = {}
        for line in cleaned.splitlines():
            phrase = line.strip()
            if phrase:
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        if phrase_counts and max(phrase_counts.values()) > 3:
            self.logger.debug("Excluding: repeated phrase >3")
            return False
        # Exclude repeated syllables or short words separated by punctuation (e.g. 'Bă, bă, bă, bă, bă, bă!')
        if re.fullmatch(r'(\b\w{1,4}[,!?\s]*){3,}', plain, re.IGNORECASE):
            self.logger.debug("Excluding: repeated short-word pattern")
            return False
        # Exclude if a single short word (<=4 chars) is repeated in multiple lines (e.g. 'You' repeated)
        short_word_counts = {}
        for line in cleaned.splitlines():
            word = line.strip()
            if word and len(word) <= 4:
                short_word_counts[word] = short_word_counts.get(word, 0) + 1
        if short_word_counts and max(short_word_counts.values()) > 2:
            self.logger.debug("Excluding: repeated short word across lines")
            return False
        # Exclude if more than 80% of characters are not letters or spaces (gibberish, Unicode blocks)
        non_letter_ratio = sum(1 for c in cleaned if not (c.isalpha() or c.isspace())) / max(1, len(cleaned))
        if non_letter_ratio > 0.8:
            self.logger.debug("Excluding: high non-letter ratio (gibberish)")
            return False
        if len(cleaned) < 5:
            self.logger.debug("Excluding: too short")
            return False
        # Exclude repeated single word/character (e.g. 'You You You', 'ლლლლლლლლ')
        if re.fullmatch(r'(\w+)( \1){2,}', plain):
            self.logger.debug("Excluding: repeated single word")
            return False
        if re.fullmatch(r'([\W_])\1{4,}', plain):
            self.logger.debug("Excluding: repeated symbol sequence")
            return False
        # Exclude mostly non-ASCII or gibberish
        ascii_ratio = sum(1 for c in cleaned if ord(c) < 128) / max(1, len(cleaned))
        if ascii_ratio < 0.3:
            self.logger.debug("Excluding: low ASCII ratio")
            return False
        # Exclude mostly punctuation or dots
        if re.fullmatch(r'[.\s]+', plain):
            self.logger.debug("Excluding: only dots/whitespace")
            return False
        # Exclude repeated lines (e.g. 'Thank you for watching!' 10x)
        lines = [l.strip() for l in plain.splitlines() if l.strip()]
        if lines and len(set(lines)) == 1 and len(lines) > 3:
            self.logger.debug("Excluding: repeated identical line")
            return False
        # Exclude onomatopoeia and sound effects (e.g. 'BOOM!', 'BEEP!', 'BUUUU...')
        if re.fullmatch(r'(BEEP!|BOOM!|BU+H*!|BU+U+U+)', plain, re.IGNORECASE):
            self.logger.debug("Excluding: onomatopoeia/sound effect")
            return False
        # Exclude repetitive syllables (e.g. 'Buh-Buh-Buh-Buh-Buh-Buh-Buh-Buh-Buh!')
        if re.fullmatch(r'((\w+-){3,}\w+!?)', plain):
            self.logger.debug("Excluding: repetitive syllable pattern")
            return False
        # Exclude single-word exclamations or sound effects
        if re.fullmatch(r'[A-Z]{2,}!?', plain):
            self.logger.debug("Excluding: all-caps exclamation")
            return False
        # Exclude 'Thank you for watching' and similar phrases
        thank_you_patterns = [
            r'^thank you for watching!?$',
            r'^thanks for watching!?$',
            r'^thank you very much for watching( and i\'ll see you in the next video\.)?$',
            r'^thank you\.?$',
            r'^thanks$',
            r'^thank you very much\.?$',
            r'^thank you for watching this video\.?$',
            r'^i hope you enjoyed it\.?$',
            r'^i\'ll see you in the next video\.?$',
            r'^bye!?$',
            r'^Thank you very much for watching, please subscribe and hit that like button!?$',
            r'^You\.?$'
        ]
        # Check thank-you / bye phrases on a per-line basis (stripped of timestamps)
        for pat in thank_you_patterns:
            for line in lines:
                if re.fullmatch(pat, line, re.IGNORECASE):
                    self.logger.debug(f"Excluding by thank-you pattern: {pat} matched line: {line}")
                    return False
            # Also check the whole plain text as a fallback
            if re.fullmatch(pat, plain, re.IGNORECASE):
                self.logger.debug(f"Excluding by thank-you pattern on whole text: {pat}")
                return False
        return True
    
            
    def process_audio_file(self, file_path: str, filename: str) -> Optional[Dict]:
        """Process audio file and return transcription"""
        try:
            self.logger.info(f"Processing audio file: {file_path}")
            
            if not self.whisper_model:
                self.logger.error("Whisper model not initialized")
                return None
            
            # Use lock to ensure thread-safe access to Whisper model
            with self._model_lock:
                # Transcribe the audio file
                segments, info = self.whisper_model.transcribe(
                    file_path,
                    beam_size=5,
                    language=None  # Auto-detect language
                )
                
                # Convert segments to list to avoid iterator issues
                segment_list = list(segments)
            
            # Build timestamped transcript (outside the lock)
            timestamped_transcript = ""
            total_duration = 0.0
            segment_count = 0
            
            for segment in segment_list:
                # Skip segments with very low confidence or high no-speech probability
                if hasattr(segment, 'no_speech_prob') and segment.no_speech_prob > 0.8:
                    continue
                    
                # Skip very short segments with little content
                if len(segment.text.strip()) < 3:
                    continue
                    
                start_time = self.format_timestamp(segment.start)
                end_time = self.format_timestamp(segment.end)
                timestamped_transcript += f"[{start_time}-{end_time}] {segment.text.strip()}\n"
                total_duration = max(total_duration, segment.end)
                segment_count += 1
            
            # Check if we have meaningful content
            if segment_count == 0 or len(timestamped_transcript.strip()) < 10:
                self.logger.info(f"No meaningful transcript found for {filename} - skipping database entry")
                return None
            
            # Extract file info
            frequency, modulation, box = self.extract_file_info(filename)
            
            # Generate current DTG and UUID
            dtg = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            task_uuid = str(uuid.uuid4())
            
            # Language checks before returning transcription
            if hasattr(info, 'language') and info.language == 'nn':
                self.logger.info(f"Detected language 'nn', skipping DB and file append...")
                return None
            if hasattr(info, 'language') and info.language == 'ro':
                self.logger.info(f"Detected language 'ro', skipping DB and file append...")
                return None
            if hasattr(info, 'language_probability') and info.language_probability < 0.5:
                self.logger.info(f"Language confidence {getattr(info, 'language_probability', 0):.2f} < 0.5, skipping...")
                return None

            # Build transcription result in database format
            transcription = {
                "dtg": dtg,
                "frequency": frequency,
                "language": info.language,
                "duration": total_duration,
                "transcript": timestamped_transcript.strip(),
                "modulation": modulation,
                "box": box,
                "filename": filename,
                "uuid": task_uuid
            }

            self.logger.info(f"Successfully transcribed {filename} ({info.language}, {total_duration:.2f}s, {segment_count} segments)")
            return transcription
            
        except Exception as e:
            self.logger.error(f"Error processing audio file: {e}")
            self.logger.error(traceback.format_exc())
            return None
            
    def submit_result(self, task_id: int, success: bool, result: Optional[Dict] = None, error_message: Optional[str] = None) -> bool:
        """Submit task result to server"""
        try:
            payload = {
                "task_id": task_id,
                "worker_id": self.worker_id,
                "success": success
            }
            
            if success and result:
                payload["transcript_data"] = result
            elif not success and error_message:
                payload["error_message"] = error_message
            
            response = requests.post(
                f"{self.server_url}/api/workers/submit-result",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                server_result = response.json()
                if server_result.get("success"):
                    self.logger.info(f"Successfully submitted result for task {task_id}")
                    return True
                else:
                    self.logger.error(f"Failed to submit result: {server_result.get('message')}")
                    return False
            else:
                self.logger.error(f"Failed to submit result with status {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error submitting result: {e}")
            return False
            
    def submit_embedding_result(self, task_id: int, success: bool, result: Optional[Dict] = None, error_message: Optional[str] = None) -> bool:
        """Submit speaker embedding task result to server"""
        try:
            payload = {
                "task_id": task_id,
                "worker_id": self.worker_id,
                "success": success
            }
            
            if result:
                payload["embedding_data"] = result
            if error_message:
                payload["error_message"] = error_message
            
            response = requests.post(
                f"{self.server_url}/api/tasks/embedding/submit",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                server_result = response.json()
                if server_result.get("success"):
                    self.logger.info(f"Successfully submitted embedding result for task {task_id}")
                    return True
                else:
                    self.logger.error(f"Failed to submit embedding result: {server_result.get('message')}")
                    return False
            else:
                self.logger.error(f"Failed to submit embedding result with status {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error submitting embedding result: {e}")
            return False
            
    def run(self):
        """Main worker loop using producer-consumer pattern"""
        self.logger.info(f"Starting WhisperWatch worker {self.worker_id} in {self.mode} mode")
        
        # Initialize appropriate models based on mode
        if self.mode == "transcription":
            if not self.initialize_whisper():
                self.logger.error("Failed to initialize Whisper model, exiting")
                return 1
        elif self.mode == "embedding":
            if not HAS_SPEAKER_DEPS:
                self.logger.error("Speaker processing dependencies not available for embedding mode")
                return 1
            if not self.initialize_speaker_models():
                self.logger.error("Failed to initialize speaker models, exiting")
                return 1
        else:
            self.logger.error(f"Unknown mode: {self.mode}")
            return 1
            
        # Register with server
        if not self.register_worker():
            self.logger.error("Failed to register with server, exiting")
            return 1
            
        self.running = True
        last_heartbeat = 0
        
        self.logger.info(f"Worker is now running with {self.max_concurrent} concurrent task processors")
        
        # Start task producer thread
        producer_thread = threading.Thread(target=self.task_producer, daemon=True)
        producer_thread.start()
        
        # Start worker threads
        worker_threads = []
        for i in range(self.max_concurrent):
            worker_thread = threading.Thread(target=self.process_task_worker, daemon=True)
            worker_thread.start()
            worker_threads.append(worker_thread)
        
        try:
            while self.running:
                current_time = time.time()
                
                # Send heartbeat every 30 seconds
                if current_time - last_heartbeat > 30:
                    if self.send_heartbeat():
                        last_heartbeat = current_time
                    else:
                        self.logger.warning("Heartbeat failed, will retry")
                
                # Log status every 60 seconds
                with self._workers_lock:
                    active_count = self._active_workers
                
                if current_time % 60 < 1:  # Roughly every 60 seconds
                    queue_size = self._task_queue.qsize()
                    self.logger.info(f"Status: {active_count}/{self.max_concurrent} workers active, {queue_size} tasks queued")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            self.running = False
            self.logger.info("Worker shutting down")
            
        return 0


def main():

    print("WhisperWorker Configuration")
    print("=" * 50)

    server_url = input("Enter WhisperWatch URL: ").strip()
    if not server_url:
        print("URL is required. Exiting...")
        exit(1)

    username = input("Enter username: ").strip()
    if not username:
        print("Username is required. Exiting...")
        exit(1)

    password = input("Enter password: ").strip()  
    if not password:
        print("Password is required. Exiting...")
        exit(1)

    worker_name = input("Enter worker ID: ").strip()
    if not worker_name:
        print("Worker ID is required. Exiting...")
        exit(1)

    try:
        max_concurrent = int(input("Enter max concurrent tasks (1-10, default: 4): ").strip() or "4")
    except ValueError:
        print("Error: Max concurrent must be an integer.")
        return 1
    if max_concurrent < 1 or max_concurrent > 10:
        print("Error: Max concurrent must be between 1 and 10")
        return 1

    mode = input("Enter mode (transcription/embedding, default: transcription): ").strip().lower() or "transcription"
    if mode not in ["transcription", "embedding"]:
        print("Error: Mode must be 'transcription' or 'embedding'")
        return 1
    
    hf_token = input("Enter Hugging Face token (required if using embedding mode): ").strip() or None
    if mode == "embedding" and not hf_token and not os.getenv("HUGGINGFACE_TOKEN"):
        print("Error: Hugging Face token is required for embedding mode")
        return 1

    worker = WorkerClient(server_url, worker_name, max_concurrent, username, password)
    return worker.run()

if __name__ == "__main__":
    sys.exit(main())
