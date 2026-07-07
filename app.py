import os
import faiss
import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify, render_template
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# --- GLOBAL VARIABLES ---
model_cross = None
model_mono  = None
cross_index = None
mono_index  = None
fc_ids_list = []
fc_lookup_eng  = {}
fc_lookup_orig = {}


# --------------------------------------------------------------------------- #
#  SYSTEM LOADER
# --------------------------------------------------------------------------- #

def load_system():
    global model_cross, model_mono, cross_index, mono_index
    global fc_ids_list, fc_lookup_eng, fc_lookup_orig

    print("=" * 60)
    print("  Initializing CrisLens Web App  (Baseline)")
    print("=" * 60)

    # 1. Download artifacts
    local_artifact_dir = "./artifacts"
    os.makedirs(local_artifact_dir, exist_ok=True)

    print("[1/3] Fetching Data from Hugging Face...")
    artifact_dir = snapshot_download(
        repo_id="Sayyam-1/crislens-artifacts",
        repo_type="dataset",
        local_dir=local_artifact_dir,
    )

    # 2. Load Bi-Encoder Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[2/3] Loading Bi-Encoder Models onto {device.upper()}...")
    model_cross = SentenceTransformer("Sayyam-1/crislens-cross-mpnet", device=device)
    model_mono  = SentenceTransformer("Sayyam-1/crislens-mono-mpnet",  device=device)

    # 3. Load FAISS Indexes & Fact-Check Lookup
    print("[3/3] Loading FAISS Indexes & Fact-Check Data...")
    cross_index = faiss.read_index(os.path.join(artifact_dir, "cross_faiss_mpnet.index"))
    mono_index  = faiss.read_index(os.path.join(artifact_dir, "mono_faiss_mpnet.index"))

    fc_ids      = np.load(os.path.join(artifact_dir, "fc_ids.npy"), allow_pickle=True)
    fc_ids_list = [int(x) for x in fc_ids]

    train_fc = pd.read_csv(os.path.join(artifact_dir, "train_fc_parsed.csv"))
    test_fc  = pd.read_csv(os.path.join(artifact_dir, "test_fc_parsed.csv"))
    all_fc   = (
        pd.concat([train_fc, test_fc], ignore_index=True)
        .drop_duplicates(subset=["fact_check_id"])
    )
    fc_lookup_eng  = dict(zip(all_fc["fact_check_id"], all_fc["fc_text_eng"]))
    fc_lookup_orig = dict(zip(all_fc["fact_check_id"], all_fc["fc_text_orig"]))

    print("System Ready! Server starting...\n")


# Load everything before starting the web server
load_system()


# --------------------------------------------------------------------------- #
#  ROUTES
# --------------------------------------------------------------------------- #

@app.route("/")
def home():
    """Serve the frontend UI."""
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def search_api():
    """Original FAISS Dense Retrieval."""
    data  = request.json
    query = data.get("query", "").strip()
    mode  = data.get("mode", "cross").lower()
    top_k = int(data.get("top_k", 5))

    if not query:
        return jsonify({"error": "Empty query"}), 400

    if mode == "cross":
        q_vec = model_cross.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        _, I_dense = cross_index.search(q_vec, top_k)
    else:
        q_vec = model_mono.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        _, I_dense = mono_index.search(q_vec, top_k)

    results = []
    for rank, idx in enumerate(I_dense[0], 1):
        fc_id = fc_ids_list[idx]
        
        display_text = (
            fc_lookup_orig.get(fc_id, "N/A")
            if mode == "mono"
            else fc_lookup_eng.get(fc_id, "N/A")
        )
        
        results.append({
            "rank":  rank,
            "id":    fc_id,
            "text":  display_text,
            "score": 0.0, # Baseline didn't return FAISS scores in UI initially
        })

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
