"""FunASR Implementation

Local offline ASR using FunASR model with automatic retry and memory management.

Features:
- Local inference (no cloud dependency)
- Automatic language detection
- Memory usage monitoring
- Retry mechanism for transient failures
- ITN (Inverse Text Normalization) support

Reference: migration/core/providers/asr/fun_local.py
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import psutil

from loom.adapters.audio.models import TranscriptionResult
from loom.adapters.audio.asr.base import BaseASR
from loom.interfaces.audio_adapter import AudioProcessingError
from loom.core.structured_logger import get_logger

logger = get_logger("audio.asr")

# Retry configuration
MAX_RETRIES = 2
RETRY_DELAY = 1  # seconds


class FunASR(BaseASR):
    """FunASR local speech recognition.

    Configuration:
        model_dir: Path to FunASR model directory
        output_dir: Directory for temporary audio files
        vad_max_single_segment: Max single segment duration (ms, default: 30000)
        use_itn: Enable Inverse Text Normalization (default: True)
        batch_size_s: Batch size in seconds (default: 60)

    Example:
        >>> asr = FunASR(
        ...     model_dir="models/paraformer-zh",
        ...     output_dir="temp/audio",
        ...     delete_temp_files=True
        ... )
        >>> result = await asr.transcribe(pcm_audio_data, language="zh")
        >>> print(result.text)
    """

    def __init__(
        self,
        model_dir: str,
        output_dir: str = "temp/audio",
        delete_temp_files: bool = True,
        vad_max_single_segment: int = 30000,
        use_itn: bool = True,
        batch_size_s: int = 60,
        min_memory_gb: float = 2.0,
    ):
        """Initialize FunASR.

        Args:
            model_dir: Path to FunASR model directory
            output_dir: Directory for temporary audio files
            delete_temp_files: Whether to delete temporary files after processing
            vad_max_single_segment: Maximum single segment duration (ms)
            use_itn: Enable Inverse Text Normalization
            batch_size_s: Batch size in seconds
            min_memory_gb: Minimum required memory (GB)
        """
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.delete_temp_files = delete_temp_files
        self.vad_max_single_segment = vad_max_single_segment
        self.use_itn = use_itn
        self.batch_size_s = batch_size_s

        # Memory check
        total_mem = psutil.virtual_memory().total
        total_mem_gb = total_mem / (1024**3)
        if total_mem_gb < min_memory_gb:
            logger.warning(
                f"Low memory: {total_mem_gb:.2f}GB < {min_memory_gb}GB. FunASR may fail to load.",
                total_mem_gb=total_mem_gb,
                min_memory_gb=min_memory_gb,
            )

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Lazy load model (defer until first use to save startup time)
        self.model = None
        self._model_loaded = False

    def _load_model(self) -> None:
        """Load FunASR model (lazy initialization)."""
        if self._model_loaded:
            return

        try:
            # Import FunASR only when needed
            from funasr import AutoModel

            logger.info("Loading FunASR model", model_dir=self.model_dir)

            self.model = AutoModel(
                model=self.model_dir,
                vad_kwargs={"max_single_segment_time": self.vad_max_single_segment},
                disable_update=True,
                hub="hf",
                # device="cuda:0",  # Enable GPU if available
            )

            self._model_loaded = True
            logger.info("FunASR model loaded successfully")

        except Exception as e:
            raise AudioProcessingError(f"Failed to load FunASR model: {e}") from e

    async def transcribe(self, audio_data: bytes, language: str = "auto") -> TranscriptionResult:
        """Transcribe audio to text with retry.

        Args:
            audio_data: Raw PCM audio data
            language: Language code ('auto' for auto-detection)

        Returns:
            TranscriptionResult with text and metadata
        """
        self._load_model()  # Ensure model is loaded

        retry_count = 0
        last_error = None

        while retry_count < MAX_RETRIES:
            try:
                start_time = time.time()

                # Run FunASR
                result = self.model.generate(
                    input=audio_data,
                    cache={},
                    language=language,
                    use_itn=self.use_itn,
                    batch_size_s=self.batch_size_s,
                )

                # Extract text with post-processing
                from funasr.utils.postprocess_utils import rich_transcription_postprocess

                text = rich_transcription_postprocess(result[0]["text"])

                latency_ms = (time.time() - start_time) * 1000

                logger.debug(
                    "ASR transcription complete",
                    text=text[:100],
                    latency_ms=latency_ms,
                    language=language,
                )

                return TranscriptionResult(
                    text=text,
                    confidence=1.0,  # FunASR doesn't provide confidence scores
                    language=language if language != "auto" else "zh",  # Default to Chinese
                    latency_ms=latency_ms,
                )

            except OSError as e:
                # Disk/memory errors - retry
                retry_count += 1
                last_error = e
                if retry_count >= MAX_RETRIES:
                    logger.error(
                        f"ASR failed after {retry_count} retries",
                        error=str(e),
                        exc_info=e,
                    )
                    break

                logger.warning(
                    f"ASR failed, retrying ({retry_count}/{MAX_RETRIES})",
                    error=str(e),
                )
                time.sleep(RETRY_DELAY)

            except Exception as e:
                # Other errors - fail immediately
                logger.error("ASR transcription error", error=str(e), exc_info=e)
                last_error = e
                break

        # All retries failed
        raise AudioProcessingError(f"ASR transcription failed: {last_error}") from last_error

    def reset(self) -> None:
        """Reset ASR state.

        Note: FunASR model is stateless, so reset is a no-op.
        """
        logger.debug("ASR reset (no-op for FunASR)")
