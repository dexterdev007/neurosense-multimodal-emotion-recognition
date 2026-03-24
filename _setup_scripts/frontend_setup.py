import os
import shutil

FRONTEND_DIR = "NeuroSense/webdev/frontend"
os.makedirs(FRONTEND_DIR, exist_ok=True)

# 1. Copy and modify styles.css
REF_CSS = "/Users/devashishsingh/Desktop/eeg/web dev/frontend/styles.css"
NEW_CSS = os.path.join(FRONTEND_DIR, "styles.css")
shutil.copy(REF_CSS, NEW_CSS)

fusion_css = """
/* Fusion page specific */
.fusion-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 18px;
  margin-top: 16px;
}

.fusion-result-final {
  background: linear-gradient(135deg, rgba(127,123,255,0.15), rgba(76,201,240,0.15));
  border: 2px solid rgba(127,123,255,0.4);
  border-radius: 20px;
  padding: 28px;
  text-align: center;
  margin-top: 20px;
}

.fusion-result-final .emotion-label-big {
  font-size: 48px;
  font-weight: 700;
  background: linear-gradient(135deg, #7f7bff, #4cc9f0);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

/* Probability bars */
.prob-bars { display: grid; gap: 10px; margin-top: 14px; }
.prob-row { display: flex; align-items: center; gap: 10px; }
.prob-label { min-width: 90px; font-size: 13px; color: var(--muted); }
.prob-bar-track { flex: 1; height: 8px; border-radius: 999px; background: rgba(127,123,255,0.15); overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 999px; background: var(--accent); transition: width 0.6s ease; }
.prob-pct { min-width: 44px; font-size: 12px; color: var(--muted); text-align: right; }

/* Image upload zone for MRI/Face */
.upload-zone {
  border: 2px dashed rgba(127,123,255,0.4);
  border-radius: 18px;
  padding: 32px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.2s, background 0.2s;
  background: rgba(127,123,255,0.04);
}
.upload-zone:hover { border-color: #7f7bff; background: rgba(127,123,255,0.08); }
.upload-zone.dragover { border-color: #4cc9f0; background: rgba(76,201,240,0.1); }

/* Image preview */
.preview-wrap {
  min-height: 240px; border-radius: 16px;
  border: 1px solid rgba(148,163,184,0.2);
  overflow: hidden; display: grid; place-items: center;
  background: rgba(255,255,255,0.4);
}
.preview-img { max-width: 100%; max-height: 400px; object-fit: contain; display: block; }
"""
with open(NEW_CSS, "a") as f:
    f.write(fusion_css)

# 2. Extract reference eeg.html completely
REF_EEG = "/Users/devashishsingh/Desktop/eeg/web dev/frontend/eeg.html"
NEW_EEG = os.path.join(FRONTEND_DIR, "eeg.html")
shutil.copy(REF_EEG, NEW_EEG)

print("styles.css copied and modified.")
print("eeg.html copied.")
