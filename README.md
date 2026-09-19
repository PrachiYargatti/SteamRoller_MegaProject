# STEAMROLLER — Fast End-to-End Speech Repair

## UI Design
<img width="1919" height="842" alt="image" src="https://github.com/user-attachments/assets/ecfe3d08-85c4-4fcc-8483-6d1a712dd9b4" />

## Flow
Upload stuttered audio -> Whisper ASR -> NLP cleaning -> local repair agents -> repaired audio output.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Speed Notes
- Default Whisper model is `tiny` for faster processing.
- Use short clips around 5–30 seconds for near/under-1-minute processing.
- The repair stage is local rule-based, not cloud LLM-based, so it is fast.

## Voice Note
The app uses local system TTS by default to avoid the default female Google TTS voice as much as possible.

## F5-TTS Voice Cloning (Kaggle)
True voice cloning is powered by F5-TTS running on a Kaggle GPU notebook, exposed
over the internet with ngrok. `voice_clone.py` is a thin HTTP client — it does not
run any model locally.

1. Open your Kaggle notebook, enable a GPU accelerator, and add a Flask + ngrok
   cell (see `kaggle_server_cell.py` for the exact code to paste in) after your
   existing F5-TTS install/setup cells. Run it — it prints a public URL like
   `https://abcd-1234.ngrok-free.app`.
2. Before running the app locally, set that URL as an environment variable:
   - macOS/Linux: `export KAGGLE_TTS_URL="https://abcd-1234.ngrok-free.app"`
   - Windows PowerShell: `$env:KAGGLE_TTS_URL="https://abcd-1234.ngrok-free.app"`
3. Run `streamlit run app.py` and pick "F5-TTS Voice Clone" as the output voice.
4. The Kaggle notebook must stay open and running for cloning to work — the URL
   changes every time you restart the notebook, so update `KAGGLE_TTS_URL` again
   when that happens.
