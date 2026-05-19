# AudioMark Annotator v1.0
### Mahindra Connector Lock Detection

## Start

### Windows
Double-click `start_windows.bat`

### Mac / Linux
```bash
bash start_mac_linux.sh
```

Open browser: **http://localhost:8100**

---

## Workflow

### Step 1 — Annotate (browser tool)
1. Upload WAV files
2. Draw segments on waveform → assign labels
3. Labels to use:
   - `connector_click` — the click sound when connector locks
   - `background_noise` — factory background (no click)
4. Press **S** to save each file
5. Apply audio effects if needed (High-Pass filter recommended)
6. Export JSON when done

### Step 2 — Train (Python script)
```bash
# Default: SVM + MFCC features
python train.py

# Better for noisy factory audio:
python train.py --model svm --features pcen

# Best accuracy (needs tensorflow):
python train.py --model yamnet

# Test your model on a file:
python train.py --predict your_audio.wav
```

### Step 3 — Use the model
The trained `data/models/model.pkl` can be loaded in any Python script:
```python
import pickle
with open("data/models/model.pkl", "rb") as f:
    md = pickle.load(f)
# md["model"]     — the classifier
# md["scaler"]    — the feature scaler
# md["labels"]    — list of label names
# md["accuracy"]  — training accuracy
```

---

## Audio Effects (in browser)
| Effect | What it does | Best use |
|---|---|---|
| Amplify | Increase volume by N dB | Quiet recordings |
| Normalize | Set peak to max | Inconsistent volumes |
| Fade In | Silence → full volume | Remove click at start |
| Fade Out | Full volume → silence | Remove click at end |
| Fade Both | Both ends | Clean clips |
| Echo | Add delayed copy | Data augmentation |
| Pitch Up/Down | Shift pitch ±2 semitones | Data augmentation |
| Speed Up/Down | Change tempo | Data augmentation |
| High-Pass 300 Hz | Remove factory hum | ⭐ Recommended for your data |
| Trim Silence | Remove silent edges | Cleaner training clips |

## Keyboard Shortcuts
| Key | Action |
|---|---|
| Space | Play / Pause |
| S | Save annotations |
| Delete | Remove selected segment |
| 1–9 | Select label by number |
| [ ] | Speed down / up |
| Ctrl+Z | Undo last segment |
| Ctrl+Scroll | Zoom in/out |
