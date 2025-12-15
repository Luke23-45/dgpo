# FILE: utils/async_saver.py

import threading
import shutil
import os
import logging
from pathlib import Path
import pytorch_lightning as pl

log = logging.getLogger("AsyncSaver")

class AsyncDriveUploader:
    """
    State-of-the-art Colab-safe checkpoint uploader.
    
    Saves to fast local NVMe first to unblock the GPU immediately, 
    then uploads to Google Drive in a background thread.
    """

    def __init__(self, local_dir: str, drive_dir: str):
        self.local_dir = Path(local_dir)
        self.drive_dir = Path(drive_dir)
        
        # Ensure directories exist
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.drive_dir.mkdir(parents=True, exist_ok=True)
        
        self._threads = []

    def _upload_worker(self, src: Path, dst: Path):
        """Background task to copy file."""
        try:
            shutil.copy2(src, dst)
            log.info(f"✅ [AsyncSaver] Uploaded: {dst.name}")
            
            # Optional: Delete local copy to save VM space
            # if src.exists():
            #     src.unlink()
        except Exception as e:
            log.error(f"❌ [AsyncSaver] Upload failed for {src.name}: {e}")

    def save_async(self, trainer: pl.Trainer, filename: str):
        """
        1. Saves checkpoint locally (Blocking but fast - seconds).
        2. Spawns thread to upload to Drive (Non-blocking).
        """
        local_path = self.local_dir / filename
        drive_path = self.drive_dir / filename

        # --- Phase 1: Fast local serialization ---
        # This captures the state safely.
        trainer.save_checkpoint(local_path)
        log.info(f"💾 [AsyncSaver] Saved locally: {local_path}")

        # --- Phase 2: Async background upload ---
        # Clean up old dead threads
        self._threads = [t for t in self._threads if t.is_alive()]
        
        # Start new upload
        t = threading.Thread(
            target=self._upload_worker, 
            args=(local_path, drive_path),
            daemon=True
        )
        t.start()
        self._threads.append(t)

    def flush(self):
        """Call this at the end of training to ensure pending uploads finish."""
        if not self._threads:
            return
            
        log.info(f"⏳ [AsyncSaver] Waiting for {len(self._threads)} pending uploads...")
        for t in self._threads:
            t.join()
        log.info("✅ [AsyncSaver] All uploads complete.")