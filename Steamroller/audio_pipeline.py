"""
audio_pipeline.py — Upload audio + optimized Whisper ASR transcription.
"""

import io
import os
import wave
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

SAMPLE_RATE = 16000
CHANNELS = 1

# Pitch thresholds (Hz) for gender classification.
# Female fundamental frequency (F0) typically: 160–300 Hz
# Male   fundamental frequency (F0) typically:  80–165 Hz
_FEMALE_F0_THRESHOLD = 160.0


def detect_speaker_gender(file_bytes: bytes, filename: str = "audio.wav") -> str:
    """Estimate speaker gender from audio using pitch (F0) analysis.

    Uses autocorrelation on short frames to find the dominant fundamental
    frequency. Returns ``"female"`` when the median F0 is above the threshold,
    otherwise ``"male"``.

    Args:
        file_bytes: Raw bytes of the audio file.
        filename:   Original filename (used for the temp-file suffix).

    Returns:
        ``"female"`` or ``"male"``.
    """
    from pathlib import Path

    suffix = Path(filename).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(file_bytes)
    tmp.flush()
    tmp.close()

    try:
        import scipy.io.wavfile as wavfile
        from scipy.signal import resample_poly
        import math

        sr, data = wavfile.read(tmp.name)

        # Convert to mono float32
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if data.max() > 1.0:
            data /= 32768.0

        # Resample to 16 kHz if needed
        if sr != SAMPLE_RATE:
            gcd = math.gcd(sr, SAMPLE_RATE)
            data = resample_poly(data, SAMPLE_RATE // gcd, sr // gcd)
            sr = SAMPLE_RATE

        # Autocorrelation-based pitch detection over 30 ms frames
        frame_len = int(sr * 0.03)        # 30 ms
        hop_len   = int(sr * 0.01)        # 10 ms hop
        min_period = int(sr / 300)        # 300 Hz upper bound
        max_period = int(sr / 60)         # 60  Hz lower bound

        pitches = []
        for start in range(0, len(data) - frame_len, hop_len):
            frame = data[start:start + frame_len]
            rms = np.sqrt(np.mean(frame ** 2))
            if rms < 0.01:               # skip silent frames
                continue
            # Normalised autocorrelation
            ac = np.correlate(frame, frame, mode="full")
            ac = ac[len(ac) // 2:]
            if ac[0] == 0:
                continue
            ac = ac / ac[0]
            # Find the first peak in the valid period range
            segment = ac[min_period:max_period]
            if segment.size == 0:
                continue
            peak_idx = int(np.argmax(segment)) + min_period
            if ac[peak_idx] > 0.3:       # confidence threshold
                f0 = sr / peak_idx
                pitches.append(f0)

        if not pitches:
            return "female"              # conservative default for children/female

        median_f0 = float(np.median(pitches))
        return "female" if median_f0 >= _FEMALE_F0_THRESHOLD else "male"

    except Exception:
        return "female"                  # safe default
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def record_audio(duration_seconds: int = 10) -> np.ndarray:
    audio = sd.rec(int(duration_seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32")
    sd.wait()
    return audio.flatten()


def normalize_audio(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Peak-normalize a float32 audio array so Whisper gets a full-volume signal.

    Microphone recordings are often quiet; Whisper hallucinates heavily on
    low-amplitude input. Normalizing to ``target_peak`` fixes this without
    clipping.
    """
    peak = np.abs(audio).max()
    if peak < 1e-6:          # completely silent — return as-is
        return audio
    return audio * (target_peak / peak)


def check_audio_has_speech(audio: np.ndarray, rms_threshold: float = 0.01) -> bool:
    """Return True if the audio contains enough energy to likely have speech.

    Uses RMS energy on the loudest 50 % of frames as a simple VAD proxy.
    Rejects recordings that are pure silence or very faint ambient noise.
    """
    if audio.size == 0:
        return False
    # Split into 20 ms frames and take the top-half by energy
    frame_len = int(SAMPLE_RATE * 0.02)
    frames = [
        audio[i : i + frame_len]
        for i in range(0, len(audio) - frame_len, frame_len)
    ]
    if not frames:
        rms = float(np.sqrt(np.mean(audio ** 2)))
        return rms >= rms_threshold
    energies = [float(np.sqrt(np.mean(f ** 2))) for f in frames]
    energies.sort(reverse=True)
    top_half = energies[: max(1, len(energies) // 2)]
    return float(np.mean(top_half)) >= rms_threshold


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def save_audio_to_temp(audio: np.ndarray) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_write(tmp.name, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    return tmp.name


_whisper_models = {}


def _load_whisper(model_size: str = "small"):
    if model_size not in _whisper_models:
        import whisper
        _whisper_models[model_size] = whisper.load_model(model_size)
    return _whisper_models[model_size]


# Whisper anti-hallucination options applied to every transcription call.
# - temperature=0          → greedy decoding; no random sampling = no hallucinations
# - no_speech_threshold    → treats segment as silent if no-speech prob > this value
# - logprob_threshold      → rejects segments with avg log-prob below this value
# - compression_ratio_threshold → flags suspiciously repetitive (hallucinated) output
_WHISPER_OPTS = dict(
    verbose=False,
    fp16=False,
    beam_size=1,
    best_of=1,
    temperature=0,                    # deterministic — eliminates sampling hallucinations
    condition_on_previous_text=False,
    no_speech_threshold=0.6,          # reject near-silent segments
    logprob_threshold=-1.0,           # reject low-confidence predictions
    compression_ratio_threshold=2.4,  # catch repetitive hallucinated loops
)

def transcribe_uploaded_file(
    file_bytes: bytes,
    filename: str = "audio.wav",
    model_size: str = "small",
    language: Optional[str] = None,
    is_live: bool = False,
    **kwargs,
) -> Tuple[str, dict]:
    """
    Transcribe an uploaded audio file.

    Args:
        file_bytes: Raw uploaded audio bytes.
        filename: Original filename.
        model_size: Whisper model size.
        language: Language code (None = auto detect).
        is_live: True when audio came from live recording.

    Returns:
        (text, whisper_result)
    """

    suffix = Path(filename).suffix or ".wav"

    tmp = tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False
    )

    try:
        tmp.write(file_bytes)
        tmp.flush()
        tmp.close()

        import whisper

        # Whisper converts mp3/m4a/wav/etc. internally
        audio = whisper.load_audio(tmp.name)

        return transcribe_audio(
            audio,
            model_size=model_size,
            language=language,
        )

    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass



def transcribe_audio(
    audio: np.ndarray,
    model_size: str = "small",
    language: Optional[str] = None,
    is_live: bool = False,
    **kwargs,
) -> Tuple[str, dict]:
    """
    Transcribe numpy audio using Whisper.

    Args:
        audio: Float32 numpy audio array.
        model_size: Whisper model size.
        language: Language code or None for auto detection.
        is_live: True for microphone/live recordings.

    Returns:
        (text, whisper_result)
    """

    # Normalize microphone/upload audio
    audio = normalize_audio(audio)

    if not check_audio_has_speech(audio):
        raise ValueError(
            "No speech detected. Please speak clearly and try again."
        )

    model = _load_whisper(model_size)

    tmp_path = save_audio_to_temp(audio)

    try:
        result = model.transcribe(
            tmp_path,
            language=language,
            initial_prompt=(
                "Transcribe exactly what is spoken. "
                "Do not guess missing words. "
                "Do not complete unfinished sentences."
            ),
            **_WHISPER_OPTS,
        )

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


    text = result.get("text", "").strip()

    if not text:
        raise ValueError(
            "Whisper could not detect speech."
        )


    # Optional confidence protection
    segments = result.get("segments", [])

    if segments:
        avg_logprob = np.mean(
            [
                s.get("avg_logprob", -10)
                for s in segments
            ]
        )

        if avg_logprob < -1.5:
            raise ValueError(
                "Low confidence transcription. "
                "Please record again."
            )

    return text, result