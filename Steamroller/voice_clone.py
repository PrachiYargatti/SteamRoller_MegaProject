"""
voice_clone.py — Remote F5-TTS voice cloning client.

Voice cloning itself runs on a Kaggle notebook (GPU) using F5-TTS. This
module just calls that notebook's HTTP endpoint (exposed via ngrok) and
saves the returned audio locally. See README.md for how to set up the
Kaggle side and where to put its public URL.
"""

import os
import uuid
import requests


class VoiceClone:
    def __init__(self, api_url=None):
        self.api_url = (api_url or os.environ.get("KAGGLE_TTS_URL", "")).rstrip("/")
        if not self.api_url:
            raise ValueError(
                "Set the KAGGLE_TTS_URL environment variable to your Kaggle "
                "notebook's ngrok URL (e.g. https://abcd-1234.ngrok-free.app)."
            )
        print(f"[VOICE CLONE] Using F5-TTS endpoint: {self.api_url}")

    def generate(
        self,
        text,
        speaker_wav,
        ref_text="",
        language="en",
        speed=0.9,
    ):
        """
        text:
            Text to convert into speech (gen_text for F5-TTS).

        speaker_wav:
            Path to the user's reference voice recording.

        ref_text:
            Transcript of speaker_wav. If you already ran ASR (e.g. Whisper,
            as app.py does), pass that transcript here for best quality.
            Leave empty to let F5-TTS auto-transcribe the reference audio.

        language:
            Reserved for compatibility with app.py / future backends.

        speed:
            Playback speed multiplier passed straight to F5-TTS's --speed
            flag. F5-TTS often sounds rushed by default; try 0.8-0.9 to
            slow it down, or 1.0+ to speed it up.
        """

        if not os.path.exists(speaker_wav):
            raise FileNotFoundError(
                f"Reference voice not found: {speaker_wav}"
            )

        os.makedirs("outputs", exist_ok=True)
        output_file = os.path.join(
            "outputs",
            f"clone_{uuid.uuid4().hex}.wav"
        )

        print("[VOICE CLONE] Sending request to Kaggle F5-TTS server...")
        print(f"[VOICE CLONE] Reference: {speaker_wav}")

        with open(speaker_wav, "rb") as f:
            try:
                resp = requests.post(
                    f"{self.api_url}/clone",
                    data={"text": text, "ref_text": ref_text, "speed": speed},
                    files={"speaker_wav": f},
                    timeout=180,
                )
            except requests.exceptions.RequestException as e:
                raise RuntimeError(
                    f"Could not reach Kaggle F5-TTS server at {self.api_url}: {e}"
                )

        if resp.status_code != 200:
            raise RuntimeError(f"F5-TTS server error ({resp.status_code}): {resp.text}")

        with open(output_file, "wb") as out:
            out.write(resp.content)

        print(f"[VOICE CLONE] Saved: {output_file}")

        return output_file