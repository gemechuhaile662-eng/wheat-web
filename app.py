"""Wheat leaf check: web version for free hosting (Render, Streamlit-free alternatives, any Linux server).

Upload a close-up photo of one wheat leaf. The answer is one line:
  Has Yellow Rust / Has Brown Rust / Healthy leaf / Not sure / Not a wheat leaf
Photos are not stored on the server.

Files needed in this folder:
  wheat3.tflite  labels.txt  config.json  ood_stats.npz
Run locally:  python app.py
On a host:    gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120
"""
import json
import os
import threading

import numpy as np
from flask import Flask, jsonify, render_template_string, request
from PIL import Image, ImageOps

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

BASE = os.path.dirname(os.path.abspath(__file__))

LABELS = open(os.path.join(BASE, "labels.txt")).read().splitlines()
CFG = json.load(open(os.path.join(BASE, "config.json")))
MULT = np.array([1.0, 1.0, float(CFG.get("yellow_multiplier", 1.0))])
THRESH = float(CFG.get("threshold", 0.70))      # below this: "Not sure"
LOOSENESS = float(CFG.get("looseness", 1.0))    # raise if real leaves are rejected

_stats = np.load(os.path.join(BASE, "ood_stats.npz"))
MEANS, PREC = _stats["means"], _stats["prec"]
LIMIT = float(_stats["thr"]) * LOOSENESS

interp = Interpreter(model_path=os.path.join(BASE, "wheat3.tflite"), num_threads=2)
interp.allocate_tensors()
IN = interp.get_input_details()[0]
_outs = interp.get_output_details()
PROB_IDX = next(o["index"] for o in _outs if o["shape"][-1] == len(LABELS))
FEAT_IDX = next(o["index"] for o in _outs if o["shape"][-1] != len(LABELS))
lock = threading.Lock()

ADVICE = {
    "healthy": "No rust found in this photo. Check a few more leaves from different plants.",
    "brown": "Signs of brown rust (leaf rust). Check nearby plants and ask an agronomist about treatment.",
    "yellow": "Signs of yellow rust (stripe rust). It spreads fast, so check nearby plants soon and ask an agronomist about treatment.",
    "unsure": "Retake the photo: one leaf filling the frame, in daylight, in focus.",
    "other": "This does not look like a wheat leaf. Take a close-up photo of one wheat leaf.",
}


def distance(feat):
    return min(float(((feat - m) @ PREC) @ (feat - m)) for m in MEANS)


def classify(img):
    x = np.asarray(img.resize((224, 224), Image.BILINEAR), dtype=np.float32)[None, ...]
    with lock:
        interp.set_tensor(IN["index"], x)       # raw 0-255 pixels, the model rescales itself
        interp.invoke()
        probs = interp.get_tensor(PROB_IDX)[0].copy()
        feat = interp.get_tensor(FEAT_IDX)[0].copy()
    probs = probs * MULT
    return probs / probs.sum(), distance(feat)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.get("/health")
def health():
    return "ok"


@app.post("/predict")
def predict_route():
    f = request.files.get("photo")
    if not f:
        return jsonify(error="No photo received. Choose or take a photo first."), 400
    try:
        img = ImageOps.exif_transpose(Image.open(f.stream)).convert("RGB")
    except Exception:
        return jsonify(error="That file could not be read as an image."), 400
    img.thumbnail((1200, 1200))

    p, dist = classify(img)
    best = int(p.argmax())
    first = LABELS[best].lower().split()[0]

    if dist > LIMIT:
        key, text = "other", "Not a wheat leaf"
    elif p[best] < THRESH:
        key, text = "unsure", "Not sure"
    elif first == "healthy":
        key, text = "healthy", "Healthy leaf"
    else:
        key = first if first in ("brown", "yellow") else "unsure"
        text = f"Has {LABELS[best]}"
    return jsonify(key=key, text=text, advice=ADVICE[key])


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wheat leaf check</title>
<style>
  :root {
    --bg:#eef2ea; --ink:#16241b; --muted:#566a5c; --line:#cbd7c6;
    --brown:#8b4a1c; --on-brown:#fff8f1;
    --yellow:#e0b100; --on-yellow:#2a2100;
    --healthy:#2f7d4e; --on-healthy:#f2fbf5;
    --unsure:#5d6670; --on-unsure:#f4f6f8;
    --action:#16241b; --on-action:#f4f8f2;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#111912; --ink:#e8efe6; --muted:#9bb0a1; --line:#2c3d31;
            --action:#e8efe6; --on-action:#111912; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:17px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  main { max-width:520px; margin:0 auto; padding:20px 16px 40px; }
  h1 { font-size:1.5rem; margin:4px 0 2px; letter-spacing:-0.01em; }
  .sub { color:var(--muted); margin:0 0 18px; }
  .btn { display:flex; align-items:center; justify-content:center; min-height:56px;
         border-radius:12px; background:var(--action); color:var(--on-action);
         font-weight:650; cursor:pointer; text-align:center; padding:0 16px; }
  .btn.busy { opacity:.5; pointer-events:none; }
  .btn:focus-within { outline:3px solid var(--healthy); outline-offset:3px; }
  input[type=file] { position:absolute; width:1px; height:1px; opacity:0; }
  #result { margin-top:22px; display:none; }
  .band { border-radius:14px; padding:20px 18px; }
  .band[data-key=brown]   { background:var(--brown);   color:var(--on-brown); }
  .band[data-key=yellow]  { background:var(--yellow);  color:var(--on-yellow); }
  .band[data-key=healthy] { background:var(--healthy); color:var(--on-healthy); }
  .band[data-key=unsure], .band[data-key=other] { background:var(--unsure); color:var(--on-unsure); }
  .band .name { font-size:2rem; font-weight:760; line-height:1.1; letter-spacing:-0.02em; }
  .photo { display:block; width:100%; border-radius:14px; margin-top:12px; background:var(--line); }
  .advice { margin:14px 2px 6px; }
  .error { margin-top:18px; padding:14px 16px; border-radius:12px; border:2px solid var(--brown); display:none; }
  footer { margin-top:28px; color:var(--muted); font-size:.88rem; }
</style>
</head>
<body>
<main>
  <h1>Wheat leaf check</h1>
  <p class="sub">Brown rust, yellow rust or healthy. Take a close-up photo of one leaf.</p>

  <label class="btn" id="btn" for="file">Take or choose a photo</label>
  <input type="file" id="file" accept="image/*">

  <div class="error" id="error" role="alert"></div>

  <section id="result" aria-live="polite">
    <div class="band" id="band"><div class="name" id="name"></div></div>
    <img class="photo" id="photo" alt="The photo that was checked">
    <p class="advice" id="advice"></p>
  </section>

  <footer>This is a photo-based estimate, not a diagnosis. It only knows these three results, so other diseases can give wrong answers. Photos are not stored.</footer>
</main>
<script>
const $ = s => document.querySelector(s);
$('#file').addEventListener('change', async e => {
  const f = e.target.files[0];
  if (!f) return;
  $('#btn').classList.add('busy'); $('#btn').textContent = 'Checking the leaf…';
  $('#error').style.display = 'none';
  const fd = new FormData(); fd.append('photo', f);
  try {
    const r = await fetch('/predict', {method: 'POST', body: fd});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || 'The check failed. Try again.');
    $('#band').dataset.key = j.key;
    $('#name').textContent = j.text;
    $('#advice').textContent = j.advice;
    $('#photo').src = URL.createObjectURL(f);
    $('#result').style.display = 'block';
    $('#result').scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (err) {
    $('#result').style.display = 'none';
    $('#error').textContent = err.message; $('#error').style.display = 'block';
  } finally {
    $('#btn').classList.remove('busy'); $('#btn').textContent = 'Take or choose a photo';
    e.target.value = '';
  }
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True)
