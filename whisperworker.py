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

import requests
from faster_whisper import WhisperModel


class WorkerClient:
    def __init__(self, server_url: str, worker_name: str, capabilities: str = "transcription", max_concurrent: int = 4, username: str = None, password: str = None):
        self.server_url = server_url.rstrip('/')
        self.worker_name = worker_name
        self.capabilities = capabilities
        self.worker_id = nanoid.generate(size=10)
        self.max_concurrent = max_concurrent  # Maximum concurrent downloads/processing
        self.running = False
        self.whisper_model = None
        self._model_lock = threading.Lock()  # Protect Whisper model access
        self._task_queue = queue.Queue(maxsize=self.max_concurrent * 2)  # Buffer for tasks
        self._active_workers = 0
        self._workers_lock = threading.Lock()
        self.username = username
        self.password = password
        
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
                "base",  # You can make this configurable
                device=device,
                compute_type=compute_type
            )
            self.logger.info("Whisper model initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Whisper model: {e}")
            return False
            
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
                    "capabilities": self.capabilities
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    self.logger.info(f"Successfully registered worker {self.worker_id}")
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
                            # Process the audio file
                            result = self.process_audio_file(temp_path, filename)
                            
                            if result:
                                # Submit successful result
                                self.submit_result(task_id, True, result)
                            else:
                                # No meaningful transcript found
                                self.submit_result(task_id, True, None)
                        else:
                            # Download failed
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
            response = requests.post(
                f"{self.server_url}/api/workers/get-task",
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
            
    def run(self):
        """Main worker loop using producer-consumer pattern"""
        self.logger.info(f"Starting WhisperWatch worker {self.worker_id}")
        
        # Initialize Whisper model
        if not self.initialize_whisper():
            self.logger.error("Failed to initialize Whisper model, exiting")
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

    # For manual capabilities input
    #capabilities = input("Enter worker capabilities (default: transcription): ").strip() or "transcription"
    try:
        max_concurrent = int(input("Enter max concurrent tasks (1-10, default: 4): ").strip() or "4")
    except ValueError:
        print("Error: Max concurrent must be an integer.")
        return 1
    if max_concurrent < 1 or max_concurrent > 10:
        print("Error: Max concurrent must be between 1 and 10")
        return 1

    
    capabilities = "transcription"  # Default capability
    worker = WorkerClient(server_url, worker_name, capabilities, max_concurrent, username, password)
    return worker.run()


if __name__ == "__main__":
    sys.exit(main())
