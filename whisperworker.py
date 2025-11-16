#!/usr/bin/env python3
import logging
import os
import queue
import signal
import socket
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

# Check internet connectivity early and set offline mode if needed
def _early_internet_check() -> bool:
    """Quick connectivity check before importing heavy libraries"""
    test_hosts = [
        ("huggingface.co", 443),
        ("8.8.8.8", 53),
    ]
    for host, port in test_hosts:
        try:
            socket.setdefaulttimeout(2)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            sock.close()
            return True
        except (socket.error, socket.timeout, socket.gaierror):
            continue
    return False

# Set HuggingFace offline mode early if no internet
if not _early_internet_check():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'

from faster_whisper import WhisperModel, BatchedInferencePipeline
import argparse
import re
from easynmt import EasyNMT

class WorkerClient:
    def __init__(self, server_url: str, worker_name: str, max_concurrent: int = 4, username: str = None, password: str = None, whisper_batch_size: int = 8):
        self.server_url = server_url.rstrip('/')
        self.worker_name = worker_name
        self.worker_id = nanoid.generate(size=10)
        self.max_concurrent = max_concurrent  # Maximum concurrent downloads/processing
        self.mode = "transcription"
        self.running = False
        self.whisper_model = None
        self.translation_model = None
        self._model_lock = threading.Lock()  # Protect model access
        self._translation_lock = threading.Lock()  # Protect translation model access
        self._task_queue = queue.Queue(maxsize=self.max_concurrent * 2)  # Buffer for tasks
        self._active_workers = 0
        self._workers_lock = threading.Lock()
        self.username = username
        self.password = password
        # Optional secret for worker-authenticated endpoints (env overrideable)
        self.worker_secret = os.getenv('WORKER_SECRET') or None
        self.whisper_batch_size = whisper_batch_size
        # PostgreSQL notification listener flags
        self.listening_for_tasks = False
        self.notification_conn = None
        # Cache internet connectivity status
        self._internet_available = None

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
    
    def _check_internet_connectivity(self) -> bool:
        """Check if internet is available by attempting to connect to common hosts"""
        if self._internet_available is not None:
            return self._internet_available
        
        # Test connectivity to multiple reliable hosts
        test_hosts = [
            ("huggingface.co", 443),  # Primary: HuggingFace (needed for model downloads)
            ("8.8.8.8", 53),           # Google DNS
            ("1.1.1.1", 53),           # Cloudflare DNS
        ]
        
        for host, port in test_hosts:
            try:
                socket.setdefaulttimeout(3)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((host, port))
                sock.close()
                self.logger.debug(f"Internet connectivity confirmed via {host}:{port}")
                self._internet_available = True
                return True
            except (socket.error, socket.timeout, socket.gaierror):
                continue
        
        self.logger.info("No internet connectivity detected - running in offline mode")
        self._internet_available = False
        return False
        
    def initialize_whisper(self):
        """Initialize the Whisper model with batched inference"""
        try:
            self.logger.info("Initializing Whisper model with batched inference...")
            # Try to use GPU if available
            device = "cuda" if self._check_cuda() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            # Determine batch size: prefer instance setting, fall back to env or default
            if getattr(self, 'whisper_batch_size', None):
                batch_size = int(self.whisper_batch_size)
            else:
                batch_size = int(os.getenv('WHISPER_BATCH_SIZE', '8'))
            
            # Get model size from environment or use default
            model_size = os.getenv('WHISPER_MODEL', 'small')
            
            self.logger.info(f"Using device: {device}, compute_type: {compute_type}, batch_size: {batch_size}, model: {model_size}")
            
            # Initialize base model
            # The download_root parameter ensures models are stored in a persistent location
            # Models will be downloaded on first run and cached for offline use
            base_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=os.getenv('WHISPER_CACHE_DIR', None)  # Uses default cache if not set
            )
            
            # Wrap with batched inference pipeline
            self.whisper_model = BatchedInferencePipeline(model=base_model)
            # store resolved batch size
            self.whisper_batch_size = batch_size
            
            self.logger.info("Whisper model with batched inference initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Whisper model: {e}")
            return False

    def initialize_translation(self):
        """Initialize the EasyNMT translation model and pre-download common OPUS-MT models"""
        try:
            self.logger.info("Initializing EasyNMT translation model...")
            
            # Check internet connectivity
            has_internet = self._check_internet_connectivity()
            
            # Download required NLTK data if not already present (offline-safe)
            try:
                import nltk
                try:
                    nltk.data.find('tokenizers/punkt_tab')
                    self.logger.debug("NLTK punkt_tab data already available")
                except LookupError:
                    if has_internet:
                        self.logger.debug("Attempting to download NLTK punkt_tab data...")
                        nltk.download('punkt_tab', quiet=True, raise_on_error=False)
                    else:
                        self.logger.debug("Skipping NLTK download (offline mode)")
            except Exception as e:
                self.logger.warning(f"NLTK data check failed (translation may still work): {e}")
            
            # Pre-download common OPUS-MT models to avoid runtime delays (offline-safe)
            if has_internet:
                self.logger.info("Pre-downloading common OPUS-MT translation models...")
                common_models = [
                    "Helsinki-NLP/opus-mt-fr-en",  # French → English
                    "Helsinki-NLP/opus-mt-es-en",  # Spanish → English
                    "Helsinki-NLP/opus-mt-de-en",  # German → English
                    "Helsinki-NLP/opus-mt-it-en",  # Italian → English
                    "Helsinki-NLP/opus-mt-ru-en",  # Russian → English
                    "Helsinki-NLP/opus-mt-uk-en",  # Ukrainian → English
                    "Helsinki-NLP/opus-mt-zh-en",  # Chinese → English
                    "Helsinki-NLP/opus-mt-pl-en",  # Polish → English
                    "Helsinki-NLP/opus-mt-ar-en",  # Arabic → English
                    "Helsinki-NLP/opus-mt-hi-en",  # Hindi → English
                ]
                
                try:
                    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                    import requests.exceptions
                    downloaded_count = 0
                    failed_count = 0
                    
                    for model_id in common_models:
                        short_name = model_id.split("/")[-1]
                        try:
                            self.logger.debug(f"Downloading {short_name}...")
                            # Set timeout and disable retries to fail fast if network is unavailable
                            AutoTokenizer.from_pretrained(
                                model_id, 
                                local_files_only=False,
                                force_download=False
                            )
                            AutoModelForSeq2SeqLM.from_pretrained(
                                model_id, 
                                local_files_only=False,
                                force_download=False
                            )
                            downloaded_count += 1
                            self.logger.debug(f"✓ Cached {short_name}")
                        except (requests.exceptions.ConnectionError, 
                                requests.exceptions.Timeout,
                                OSError) as e:
                            failed_count += 1
                            # Check if model exists locally
                            try:
                                AutoTokenizer.from_pretrained(model_id, local_files_only=True)
                                self.logger.debug(f"Using cached {short_name}")
                            except:
                                self.logger.debug(f"Could not download {short_name} (not cached)")
                        except Exception as e:
                            failed_count += 1
                            self.logger.debug(f"Could not download {short_name}: {type(e).__name__}")
                    
                    if downloaded_count > 0:
                        self.logger.info(f"Pre-downloaded {downloaded_count}/{len(common_models)} OPUS-MT models")
                    if failed_count == len(common_models):
                        self.logger.warning("Could not download any models (using cached models if available)")
                except Exception as e:
                    self.logger.warning(f"Error during model pre-download (will use cached models if available): {e}")
            else:
                self.logger.info("Skipping model download (offline mode - using cached models)")
            
            # Use a lightweight model suitable for general translation
            # Options: 'opus-mt', 'mbart50_m2m', 'm2m_100_418M', 'm2m_100_1.2B'
            self.translation_model = EasyNMT('opus-mt')
            self.logger.info("EasyNMT translation model initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize EasyNMT model: {e}")
            self.logger.warning("Translation will be disabled for this worker")
            return False

    def try_get_and_queue_task(self):
        try:
            if not self._task_queue.full():
                task = self.get_task()
                if task:
                    try:
                        self._task_queue.put_nowait(task)
                    except queue.Full:
                        pass
        except Exception as e:
            self.logger.debug(f"Notification-triggered fetch failed: {e}")

    # Speaker ID functionality removed per project requirements
            
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
                    "mode": self.mode,
                    "secret": self.worker_secret,
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
                        label = task.get('id') or task.get('filename') or 'unknown'
                        self.logger.debug(f"Added task {label} to queue")
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
                    label = task.get('id') or task.get('filename') or 'unknown'
                    filename = task.get("filename")
                    task_id = task.get("id")
                    file_path = task.get("file_path")
                    if not file_path and filename:
                        # Construct a path for legacy download fallback if needed
                        file_path = os.path.join(os.getenv('AUDIO_FOLDER', 'audio'), filename)
                    self.logger.info(f"Worker processing task {label}: {filename}")
                    
                    # Create temporary file for download
                    temp_file = tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False)
                    temp_path = temp_file.name
                    temp_file.close()
                    
                    try:
                        # Download the file
                        if self.download_file(file_path or filename, temp_path, filename_hint=filename):
                            # Process based on task type
                            # Transcription only
                            result = self.process_audio_file(temp_path, filename)
                            if result and self.is_coherent_transcript(result.get("transcript", "")):
                                if task_id is not None:
                                    self.submit_result(task_id, True, result)
                                else:
                                    self.submit_transcription_result_filename(filename, True, result)
                            else:
                                # If we processed the file but transcript isn't coherent, mark as complete with empty transcript.
                                # If result is None (e.g., skipped/processing failed), record failure as before to allow retry logic.
                                if result is not None:
                                    empty_result = {
                                        "transcript": "",
                                        "language": result.get("language", ""),
                                        "duration": result.get("duration", 0.0),
                                    }
                                    if task_id is not None:
                                        self.submit_result(task_id, True, empty_result)
                                    else:
                                        self.submit_transcription_result_filename(filename, True, empty_result)
                                else:
                                    if task_id is not None:
                                        self.submit_result(task_id, False, error_message="Processing failed or skipped")
                                    else:
                                        self.submit_transcription_result_filename(filename, False, error_message="Processing failed or skipped")
                        else:
                            # Download failed
                            if task_id is not None:
                                self.submit_result(task_id, False, error_message="Failed to download file")
                            else:
                                self.submit_transcription_result_filename(filename, False, error_message="Failed to download file")
                    
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
            endpoint = "/api/workers/get-task"
            response = requests.post(
                f"{self.server_url}{endpoint}",
                json={"worker_id": self.worker_id, "mode": self.mode},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success") and result.get("task"):
                    task = result["task"]
                    # Normalize task shape as transcription-only
                    if 'filename' in task and 'file_path' not in task:
                        task.setdefault('task_type', 'transcription')
                    return task
            else:
                self.logger.debug(f"No tasks available (status: {response.status_code})")
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting task: {e}")
            return None
            
    def download_file(self, file_path_or_filename: str, local_path: str, filename_hint: Optional[str] = None) -> bool:
        """Download file, preferring worker-authenticated endpoint by filename; fallback to legacy /download."""
        try:
            # Try worker endpoint by filename first
            filename = filename_hint or os.path.basename(file_path_or_filename)
            try:
                headers = {}
                if self.worker_id:
                    headers['X-Worker-Id'] = str(self.worker_id)
                if self.worker_secret:
                    headers['X-Worker-Secret'] = str(self.worker_secret)
                url = f"{self.server_url}/api/workers/audio/{filename}"
                resp = requests.get(url, headers=headers, stream=True, timeout=60)
                if resp.status_code == 200:
                    with open(local_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    return True
                else:
                    self.logger.error(f"Worker audio endpoint 404/err for {filename}: status={resp.status_code}")
            except Exception as e:
                self.logger.debug(f"Worker audio endpoint failed, will fallback: {e}")

            # Fallback to legacy /download with a file path
            from urllib.parse import quote
            quoted = quote(file_path_or_filename, safe='')
            response = requests.get(
                f"{self.server_url}/download/{quoted}",
                stream=True,
                timeout=60
            )
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            else:
                self.logger.error(f"Legacy download endpoint failed for {file_path_or_filename}: status={response.status_code}")
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
    
    def translate_text(self, text: str, source_lang: str, target_lang: str = 'en') -> Optional[str]:
        """Translate text from source language to target language using EasyNMT"""
        if not self.translation_model:
            self.logger.debug("Translation model not available, skipping translation")
            return None
        
        # Don't translate if already in English
        if source_lang == 'en' or source_lang == target_lang:
            return None
        
        try:
            self.logger.info(f"Translating text from {source_lang} to {target_lang}")
            
            # Use lock to ensure thread-safe access to translation model
            with self._translation_lock:
                # Split by lines to preserve timestamp structure
                lines = text.strip().split('\n')
                translated_lines = []
                
                for line in lines:
                    if not line.strip():
                        translated_lines.append(line)
                        continue
                    
                    # Extract timestamp prefix if present
                    timestamp_match = re.match(r'(\[\d{1,2}:\d{2}-\d{1,2}:\d{2}\]\s*)', line)
                    if timestamp_match:
                        timestamp_prefix = timestamp_match.group(1)
                        text_to_translate = line[len(timestamp_prefix):]
                    else:
                        timestamp_prefix = ""
                        text_to_translate = line
                    
                    # Skip empty lines
                    if not text_to_translate.strip():
                        translated_lines.append(line)
                        continue
                    
                    # Translate the text (without timestamp)
                    try:
                        translated = self.translation_model.translate(
                            text_to_translate,
                            source_lang=source_lang,
                            target_lang=target_lang
                        )
                        # Reconstruct line with timestamp
                        translated_lines.append(timestamp_prefix + translated)
                    except Exception as e:
                        self.logger.warning(f"Failed to translate line, keeping original: {e}")
                        translated_lines.append(line)
                
                translated_text = '\n'.join(translated_lines)
                self.logger.info(f"Successfully translated text from {source_lang} to {target_lang}")
                return translated_text
                
        except Exception as e:
            self.logger.error(f"Error during translation: {e}")
            self.logger.error(traceback.format_exc())
            return None
    

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
                # Transcribe the audio file with batched inference
                segments, info = self.whisper_model.transcribe(
                    file_path,
                    batch_size=self.whisper_batch_size,
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
            
            # Extract file info
            frequency, modulation, box = self.extract_file_info(filename)
            # Generate current DTG and UUID
            dtg = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            task_uuid = str(uuid.uuid4())

            # Check if we have meaningful content; if not, return an empty but successful transcription result
            if segment_count == 0 or len(timestamped_transcript.strip()) < 10:
                self.logger.info(f"No meaningful transcript found for {filename} - marking complete with empty transcript")
                transcription = {
                    "dtg": dtg,
                    "frequency": frequency,
                    "language": "",  # No language when there's no content
                    "duration": total_duration,
                    "transcript": "",
                    "transcript_original": "",
                    "modulation": modulation,
                    "box": box,
                    "filename": filename,
                    "uuid": task_uuid
                }
                return transcription

            # Language checks AFTER deciding no-content completion
            # If there is content but language confidence is low, mark complete with empty transcript
            if hasattr(info, 'language') and info.language == 'nn':
                self.logger.info(f"Detected language 'nn' with content, marking complete with empty transcript")
                transcription = {
                    "dtg": dtg,
                    "frequency": frequency,
                    "language": "",  # No language for filtered content
                    "duration": total_duration,
                    "transcript": "",
                    "transcript_original": "",
                    "modulation": modulation,
                    "box": box,
                    "filename": filename,
                    "uuid": task_uuid
                }
                return transcription
            if hasattr(info, 'language') and info.language == 'ro':
                self.logger.info(f"Detected language 'ro' with content, marking complete with empty transcript")
                transcription = {
                    "dtg": dtg,
                    "frequency": frequency,
                    "language": "",  # No language for filtered content
                    "duration": total_duration,
                    "transcript": "",
                    "transcript_original": "",
                    "modulation": modulation,
                    "box": box,
                    "filename": filename,
                    "uuid": task_uuid
                }
                return transcription
            if hasattr(info, 'language_probability') and info.language_probability < 0.5:
                self.logger.info(f"Language confidence {getattr(info, 'language_probability', 0):.2f} < 0.5 with content, marking complete with empty transcript")
                transcription = {
                    "dtg": dtg,
                    "frequency": frequency,
                    "language": "",  # No language for low-confidence content
                    "duration": total_duration,
                    "transcript": "",
                    "transcript_original": "",
                    "modulation": modulation,
                    "box": box,
                    "filename": filename,
                    "uuid": task_uuid
                }
                return transcription

            # Translate if not in English
            translated_transcript = None
            if info.language != 'en':
                self.logger.info(f"Detected non-English language '{info.language}', attempting translation")
                translated_transcript = self.translate_text(timestamped_transcript.strip(), info.language, 'en')
            
            # Build transcription result in database format
            transcription = {
                "dtg": dtg,
                "frequency": frequency,
                "language": info.language,
                "duration": total_duration,
                "transcript": translated_transcript if translated_transcript else timestamped_transcript.strip(),
                "transcript_original": timestamped_transcript.strip() if translated_transcript else None,
                "modulation": modulation,
                "box": box,
                "filename": filename,
                "uuid": task_uuid
            }

            self.logger.info(f"Successfully transcribed {filename} ({info.language}, {total_duration:.2f}s, {segment_count} segments{', translated to English' if translated_transcript else ''})")
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
            

    def submit_transcription_result_filename(self, filename: str, success: bool, result: Optional[Dict] = None, error_message: Optional[str] = None) -> bool:
        """Submit transcription result using filename-based (Postgres) endpoint."""
        try:
            payload = {
                "filename": filename,
                "worker_id": self.worker_id,
                "success": success,
            }
            if success:
                # Accept empty/none result by providing defaults
                result = result or {}
                payload["transcript"] = result.get("transcript", "")
                payload["transcript_original"] = result.get("transcript_original", None)
                payload["language"] = result.get("language", "")
                payload["duration"] = result.get("duration", 0.0)
            elif not success and error_message:
                payload["error_message"] = error_message
            resp = requests.post(f"{self.server_url}/api/workers/submit-transcription", json=payload, timeout=30)
            ok = (resp.status_code == 200 and resp.json().get('success'))
            if ok:
                self.logger.info(f"Submitted transcription (PG) for {filename} (success={success})")
                return True
            self.logger.error(f"Failed to submit transcription (PG) for {filename}: {resp.status_code} {resp.text}")
            return False
        except Exception as e:
            self.logger.error(f"Error submitting transcription (PG): {e}")
            return False

            
    def run(self):
        """Main worker loop using producer-consumer pattern + Postgres notifications when available."""
        self.logger.info(f"Starting WhisperWatch worker {self.worker_id} in {self.mode} mode")
        
        # Initialize Whisper model (transcription-only)
        if not self.initialize_whisper():
            self.logger.error("Failed to initialize Whisper model, exiting")
            return 1
        
        # Initialize translation model (optional, worker continues if this fails)
        self.initialize_translation()
            
        # Register with server
        if not self.register_worker():
            self.logger.error("Failed to register with server, exiting")
            return 1
            
        self.running = True
        last_heartbeat = 0
        
        self.logger.info(f"Worker is now running with {self.max_concurrent} concurrent task processors")
        
        # Start task producer thread (HTTP polling fallback)
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
    # If no CLI parameters are provided, enter interactive mode
    if len(sys.argv) == 1:
        print("WhisperWorker Configuration")
        print("=" * 50)
        # interactive prompts (allow environment variables as defaults)
        server_url = input("Enter WhisperWatch URL: ").strip()
        username = input("Enter username: ").strip()
        password = input("Enter password: ").strip()
        worker_name = input("Enter worker name: ").strip()
        max_concurrent_str = input("Enter max concurrent threads (1-10) [allow more for larger GPUs, default 4]: ").strip()
        whisper_batch_str = input("Enter Whisper batch size (1-16) [allow more for larger GPUs, default 8]: ").strip()
        try:
            max_concurrent = int(max_concurrent_str) if max_concurrent_str else 4
        except Exception:
            print("Invalid max concurrent, defaulting to 4")
            max_concurrent = 4
        try:
            whisper_batch_size = int(whisper_batch_str) if whisper_batch_str else 8
        except Exception:
            print("Invalid whisper batch size, defaulting to 8")
            whisper_batch_size = 8
        print("=" * 50)
    else:
        parser = argparse.ArgumentParser(description="WhisperWorker")
        parser.add_argument("--server-url", required=True, help="WhisperWatch server URL")
        parser.add_argument("--username", required=True, help="Username for authentication")
        parser.add_argument("--password", required=True, help="Password for authentication")
        parser.add_argument("--worker-name", required=True, help="Worker ID/name")
        parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent tasks (1-10, default: 4)")
        parser.add_argument("--whisper-batch-size", type=int, default=8, help="Batch size for Whisper batched inference (1-16, default: 8)")
        args = parser.parse_args()

        server_url = args.server_url
        username = args.username
        password = args.password
        worker_name = args.worker_name
        max_concurrent = args.max_concurrent
    # Speaker ID functionality removed; worker runs in transcription-only mode
    # parser.add_argument("--worker-secret", help="Optional worker secret to authenticate /api/workers/audio downloads")

    if max_concurrent < 1 or max_concurrent > 10:
        print("Error: Max concurrent must be between 1 and 10")
        return 1

    # Determine whisper_batch_size from args (if present) or interactive variable
    try:
        whisper_batch_size_arg = args.whisper_batch_size
    except Exception:
        # args may not exist in interactive mode
        whisper_batch_size_arg = None

    whisper_batch_size = whisper_batch_size_arg if whisper_batch_size_arg is not None else (locals().get('whisper_batch_size', None))

    worker = WorkerClient(
        server_url,
        worker_name,
        max_concurrent,
        username,
        password,
        whisper_batch_size=whisper_batch_size,
    )

    return worker.run()


if __name__ == "__main__":
    sys.exit(main())
