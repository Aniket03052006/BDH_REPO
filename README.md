<p align="center">
  <img src="icon.png" width="80" alt="BDH Logo" />
</p>

<h1 align="center">🧠 Neural Explorer — BDH Interactive Visualizer</h1>

<p align="center">
  <strong>Real-time 3D visualization and interactive exploration of the Biologically-Derived Heuristic (BDH) neural architecture</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#comparison-mode">Comparison</a> •
  <a href="#training">Training</a> •
  <a href="#api-reference">API</a>
</p>

---

## Live Deployment

- URL: https://bdh-repo.vercel.app
- Note: The app may take about 90-120 seconds to connect on first load.

---

## Overview

Neural Explorer is an interactive web application for visualizing and experimenting with the **BDH (Baby Dragon Hatchling / Biologically-Derived Heuristic)** neural architecture — a novel approach to language modeling that replaces traditional dense MLP layers with **sparse, biologically-inspired activation patterns**. The system provides a live 3D globe visualization of neural firing, Hebbian memory tracking, neuron profiling, ablation studies, weight consolidation, and a side-by-side comparison mode against GPT-2 (Transformer).

> **Copyright © 2025 Pathway Technology, Inc.**

---

## Features

### 🌐 3D Neural Globe
- **Fibonacci-sphere point cloud** rendered with Three.js showing all neurons in real-time
- **Color-coded activation**: dark → indigo → cyan → yellow → white based on firing intensity
- **Interactive controls**: rotate, zoom, pan, freeze, auto-rotate, top/side views
- **Timeline scrubber** to replay token-by-token activation history

### 💬 Chat Interface
- Byte-level character generation with live token streaming
- Word reconstruction and word-span tracking
- Temperature and Top-K sampling controls
- Scrollable chat history with user/AI message bubbles

### 🔬 Neuron Profiling
- Click any neuron to view its full profile
- **Activation timeline** showing firing history across tokens
- **Top associated words** — learned through word-level activation mapping
- **Head/Position info** — which attention head and internal index

### 🧬 Hebbian Learning
- Real-time Hebbian memory accumulation (`α=0.5, η=0.05`)
- Memory statistics display (max, mean)
- Persistent or per-prompt memory modes
- Influence on neuron activation patterns

### ✂️ Ablation Studies
- Toggle individual neurons on/off
- Observe how ablation affects generation
- Batch ablation of top-N active neurons

### 🔄 Weight Consolidation
- **Option-B Consolidation**: reinforces encoder, encoder_v, and decoder weights
- Selects top-quantile features per attention head
- Reports **ΔE** (encoder), **ΔEv** (encoder_v), **ΔD** (decoder) weight norm changes
- Configurable alpha and quantile thresholds

### ⚡ BDH vs Transformer Comparison
- **Dual 3D globes** — BDH (sparse blue) vs Transformer (dense red)
- **Memory tracking** — Transformer O(T) KV cache vs BDH O(1) fixed memory
- **Synchronized generation** — same prompt, same token count
- **Live logs** — real-time terminal output for both models
- **Generated output display** — see both models' text in the Memory tab

### 🧪 Experiment B
- Teach-then-test paradigm
- Feed a teaching sentence, then test if the model learned the pattern
- Quantitative results with similarity metrics

---

## Architecture

### BDH Model (`bdh.py`)

The BDH architecture is a character-level language model with biologically-inspired sparse activation:

```
Input (byte tokens)
  │
  ▼
┌─────────────────┐
│   Embedding      │  vocab_size=256 → n_embd
│   (byte-level)   │
└────────┬────────┘
         │
    ╔════╧════════════════════════════════════════╗
    ║           × n_layer (6 layers)              ║
    ║                                              ║
    ║  ┌──────────┐                                ║
    ║  │ Encoder   │  n_embd → n_head × N          ║
    ║  │ (W_enc)   │  + ReLU → sparse xs           ║
    ║  └────┬─────┘                                ║
    ║       │                                      ║
    ║  ┌────▼─────┐                                ║
    ║  │ Phase     │  Rotary-style attention         ║
    ║  │ Attention │  Q=xs, K=xs, V=x               ║
    ║  └────┬─────┘                                ║
    ║       │                                      ║
    ║  ┌────▼──────┐                               ║
    ║  │ Encoder_V  │  n_embd → n_head × N         ║
    ║  │ (W_enc_v)  │  + ReLU → sparse ys          ║
    ║  └────┬──────┘                               ║
    ║       │                                      ║
    ║  ┌────▼─────┐                                ║
    ║  │ Gate      │  xys = xs ⊙ ys (element-wise) ║
    ║  │ (Sparse)  │  + Dropout                     ║
    ║  └────┬─────┘                                ║
    ║       │                                      ║
    ║  ┌────▼─────┐                                ║
    ║  │ Decoder   │  n_head × N → n_embd          ║
    ║  │ (W_dec)   │  + LayerNorm + Residual        ║
    ║  └────┬─────┘                                ║
    ╚═══════╧══════════════════════════════════════╝
         │
  ┌──────▼────────┐
  │  LM Head       │  n_embd → vocab_size
  │  (tied weights)│
  └───────────────┘
         │
         ▼
  Token Prediction (256-way softmax)
```

**Key dimensions:**
- `n_head` = 4 attention heads
- `N` = `n_embd × mlp_internal_dim_multiplier / n_head` neurons per head
- Total neurons = `n_head × N` (e.g., 4 × 8192 = 32,768)

### Server Architecture (`server.py`)

```
                    ┌─────────────────────┐
                    │    FastAPI Server     │
                    │    (port 8003)        │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐ ┌──────▼───────┐ ┌──────▼──────┐
    │  HTTP Routes    │ │  WebSocket   │ │  REST API   │
    │  (static files) │ │  /ws         │ │  /consolidate│
    └────────────────┘ │              │ └─────────────┘
                       │              │
         ┌─────────────┼──────────────┼──────────────┐
         │             │              │              │
    ┌────▼───┐   ┌─────▼────┐  ┌─────▼───┐   ┌─────▼────┐
    │ Prompt  │   │ Neuron   │  │ Ablation│   │Comparison│
    │ Handler │   │ Profile  │  │ Handler │   │  "both"  │
    └────┬───┘   └──────────┘  └─────────┘   └────┬─────┘
         │                                         │
    ┌────▼───────────────────────────────────┐     │
    │         BDH Model (CPU/GPU)            │     │
    │  • Token generation                    │     │
    │  • Sparse activation extraction        │     │
    │  • Hebbian memory update               │     │
    │  • Word-span reconstruction            │     │
    └────────────────────────────────────────┘     │
                                                   │
                                    ┌──────────────▼──────┐
                                    │  TransformerLens     │
                                    │  GPT-2 (lazy-loaded) │
                                    │  • Dense generation   │
                                    │  • KV cache tracking  │
                                    └─────────────────────┘
```

### Frontend Architecture

```
index.html
  ├── Chat Sidebar (left panel)
  │     ├── Brand header + Compare button
  │     ├── Chat history (scrollable)
  │     └── Prompt input + controls
  │
  ├── 3D Globe Viewport (center)
  │     ├── Three.js WebGL canvas
  │     ├── Timeline scrubber
  │     └── View controls (rotate, zoom, freeze)
  │
  ├── Right Panel (tabbed)
  │     ├── Controls tab (temperature, top-k, Hebbian)
  │     ├── Neuron Profile tab
  │     └── Experiment B tab
  │
  └── Comparison Modal (full-screen overlay)
        ├── Header with Globe/Memory tabs
        ├── Globe tab: dual Three.js canvases
        ├── Memory tab: TF/BDH cards + output + logs
        └── Comparison prompt input
```

---

## Project Structure

```
finalised_model/
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
│
├── bdh.py                  # BDH model architecture
├── server.py               # FastAPI backend + WebSocket
├── train.py                # Local training script
├── remote_train.py         # Remote GPU training script
│
├── index.html              # Main HTML page
├── main.js                 # Frontend logic (Three.js, WebSocket, UI)
├── style.css               # All CSS styles
├── icon.png                # Brand icon
│
├── bdh_wikipedia_final.pt  # Pre-trained model checkpoint
└── static/                 # Static file directory
```

---

## Quickstart

### Prerequisites

- **Python** 3.10+
- **pip** (Python package manager)
- **GPU** (optional, CUDA-compatible for faster inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/neural-explorer.git
cd neural-explorer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
python server.py
```

The server starts on **http://localhost:8003**. Open this URL in your browser.

### What You'll See

1. **3D Globe** — neurons arranged on a sphere, initially dark
2. **Chat** — type a prompt (e.g., "The quick brown fox") and press Send
3. **Live Generation** — tokens stream in, neurons light up on the globe
4. **Explore** — hover neurons, click for profiles, use the timeline scrubber

---

## Comparison Mode

Click the **⚡ Compare** button in the top-left to open the comparison modal:

1. **Globe Tab** — dual globes showing BDH (sparse blue, left) vs Transformer (dense red, right)
2. **Memory Tab** — live memory usage: Transformer's O(T) KV cache growing vs BDH's O(1) constant memory
3. **Generated Output** — each model's text appears under its memory card
4. **Live Logs** — real-time token-by-token logs with memory stats

> **Note:** The Transformer model (GPT-2 via TransformerLens) loads lazily on first comparison use. This may take 30-60 seconds on first run.

### What the Comparison Shows

| Metric | Transformer (GPT-2) | BDH |
|--------|---------------------|-----|
| **Activation** | Dense — all neurons fire | Sparse — selective firing |
| **Memory** | O(T) — grows linearly with tokens | O(1) — constant |
| **Globe** | Full red glow | Selective blue sparks |
| **KV Cache** | 2 × layers × seq_len × d_model × 4B | Fixed parameter memory |

---

## Training

### Local Training

```bash
# Downloads Aesop's Fables dataset automatically
python train.py
```

**Config** (in `train.py`):
- `n_head=8`, `n_embd=512`, `n_layer=6`, `mlp_internal_dim_multiplier=16`
- 30,000 iterations, batch size 4, block size 256
- Saves checkpoint to `ckpt.pt`

### Remote GPU Training

For training on a remote server (e.g., cloud GPU):

```bash
# Upload bdh.py, remote_train.py, and your dataset
python remote_train.py --data dataset.txt
```

**Config** (in `remote_train.py`):
- `n_head=4`, `n_embd=256`, `n_layer=4`, `mlp_internal_dim_multiplier=64`
- 5,000 steps, batch size 64
- Saves checkpoint to `bdh_wikipedia_final.pt` (visualizer-compatible format)

### Checkpoint Format

The server expects a checkpoint with:
```python
{
    'model_state_dict': model.state_dict(),
    'config': BDHConfig(...),  # Must be included
    'optimizer_state_dict': ...,  # Optional
}
```

---

## API Reference

### WebSocket `/ws`

All communication happens over a single WebSocket connection.

#### Client → Server Messages

| Message Type | Fields | Description |
|-------------|--------|-------------|
| `prompt` | `text`, `model_type?` | Generate text. `model_type='both'` for comparison |
| `reset` | — | Reset session state |
| `set_temperature` | `value` | Set generation temperature (0.1–2.0) |
| `set_topk` | `value` | Set top-k sampling (1–100) |
| `toggle_ablation` | `neuron_id` | Toggle neuron ablation |
| `request_neuron_profile` | `neuron_id` | Get neuron profile data |
| `consolidate_weights` | — | Trigger weight consolidation |
| `set_hebb_persist` | `persist` | Toggle Hebbian memory persistence |
| `experiment_b` | `prompt`, `teach` | Run Experiment B |

#### Server → Client Messages

| Message Type | Key Fields | Description |
|-------------|------------|-------------|
| `config` | `total_neurons` | Initial neuron count |
| `token` | `character`, `xy_vis`, `model?`, `mem_info?` | Generated token + activations |
| `neuron_profile` | `neuron_id`, `timeline`, `top_words` | Neuron profile data |
| `log` | `model`, `text` | Comparison mode log line |
| `done` | `status` | Generation complete |

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /` | GET | Serve `index.html` |
| `GET /{file}` | GET | Serve static files |
| `GET /consolidate` | GET | Trigger consolidation, returns diagnostics |

---

## Consolidation Diagnostics

When you click "Consolidate Weights", the system:

1. **Selects top-quantile features** per head based on accumulated activation stats
2. **Reinforces three weight matrices**:
   - **Encoder** (W_enc): input → sparse representation
   - **Encoder_V** (W_enc_v): attention output → sparse value
   - **Decoder** (W_dec): sparse → output
3. **Reports weight norm changes**:
   - **ΔE**: encoder change magnitude
   - **ΔEv**: encoder_v change magnitude
   - **ΔD**: decoder change magnitude

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Python, FastAPI, Uvicorn |
| **ML Framework** | PyTorch |
| **Model** | BDH (custom), GPT-2 (TransformerLens) |
| **Frontend** | HTML5, CSS3, JavaScript (ES modules) |
| **3D Rendering** | Three.js + OrbitControls |
| **Communication** | WebSocket (real-time streaming) |
| **Memory Tracking** | `tracemalloc` (comparison mode) |

---

## Configuration

Key environment variables and settings:

| Settings | Default | Location |
|---------|---------|----------|
| Server port | `8003` | `server.py` bottom |
| Device | Auto (CUDA/CPU) | `server.py:30` |
| Checkpoint file | `bdh_wikipedia_final.pt` | `server.py:55` |
| Hebbian α | `0.5` | `server.py:175` |
| Hebbian η | `0.05` | `server.py:176` |
| Max comparison tokens | `50` | `server.py:488` |
| Default temperature | `0.7` | `server.py` |

---

## License

Copyright © 2025 Pathway Technology, Inc. All rights reserved.
