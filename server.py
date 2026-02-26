# Copyright 2025 Pathway Technology, Inc.
# Merged server: BDH_Proto (neuron profiling, word reconstruction, chat) +
#                src_bdh_2 (Hebbian, ablation, consolidation, Experiment B)

import asyncio
import json
import logging
import os
import struct
import dataclasses
from collections import deque, defaultdict
from pathlib import Path
from typing import List, Optional, Set

import bdh
import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("bdh-server")
logging.basicConfig(level=logging.INFO)

# ──────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────
app = FastAPI()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# Serve static files (style.css, icon.png etc.)
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def get():
    return FileResponse("index.html")

@app.get("/{file_name}")
async def get_file(file_name: str):
    path = os.path.join(".", file_name)
    if os.path.isfile(path):
        if file_name.endswith(".js"):
            return FileResponse(path, media_type="application/javascript")
        return FileResponse(path)
    return {"error": "File not found"}

# ──────────────────────────────────────────
# Load Model
# ──────────────────────────────────────────
def load_model():
    checkpoint_path = "bdh_wikipedia_final.pt"
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        torch.serialization.add_safe_globals([bdh.BDHConfig])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        model = bdh.BDH(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Model loaded: {config.n_layer} layers, {config.n_embd} embd, "
                     f"{config.n_head} heads, mlp_mult={config.mlp_internal_dim_multiplier}")
    else:
        logger.warning(f"Checkpoint {checkpoint_path} not found. Initializing random model.")
        config = bdh.BDHConfig()
        model = bdh.BDH(config)

    model.eval()
    model = model.to(DEVICE)

    nh = config.n_head
    D = config.n_embd
    N = D * config.mlp_internal_dim_multiplier // nh
    total_neurons = nh * N
    logger.info(f"Neuron architecture: {nh} heads × {N} per head = {total_neurons} total (flat index)")
    return model

MODEL = load_model()


# ──────────────────────────────────────────
# Transformer (TransformerLens) — Lazy Load for Comparison
# ──────────────────────────────────────────
import tracemalloc

TF_MODEL = None
TF_TOKENIZER = None

def lazy_load_transformer():
    global TF_MODEL, TF_TOKENIZER
    if TF_MODEL is not None:
        return
    try:
        from transformer_lens import HookedTransformer
        logger.info("Loading TransformerLens GPT-2 small for comparison...")
        TF_MODEL = HookedTransformer.from_pretrained("gpt2", device=str(DEVICE))
        TF_MODEL.eval()
        TF_TOKENIZER = TF_MODEL.tokenizer
        logger.info("Transformer model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load TransformerLens: {e}")


async def stream_transformer_tokens(ws, prompt_text, max_tokens=100):
    """Run GPT-2 via TransformerLens, stream tokens with activations for comparison globe."""
    lazy_load_transformer()
    if TF_MODEL is None:
        await manager.send_json(ws, {"type": "log", "model": "transformer", "text": "Transformer model not available"})
        return 0

    import numpy as np
    tokens = TF_MODEL.to_tokens(prompt_text)
    n_neurons = TF_MODEL.cfg.d_mlp or 3072
    generated_count = 0

    tracemalloc.start()
    for step in range(max_tokens):
        with torch.no_grad():
            logits = TF_MODEL(tokens)
        next_logit = logits[0, -1, :]
        probs = torch.softmax(next_logit / 0.7, dim=-1)
        top_vals, top_ids = torch.topk(probs, 10)
        probs_filtered = torch.zeros_like(probs)
        probs_filtered[top_ids] = top_vals
        probs_filtered = probs_filtered / probs_filtered.sum()
        next_token = torch.multinomial(probs_filtered, 1).unsqueeze(0)
        tokens = torch.cat([tokens, next_token], dim=1)
        generated_count += 1

        # Decode character
        char = TF_TOKENIZER.decode(next_token[0].tolist())

        # Dense activations — all neurons firing
        vis = [1.0] * min(n_neurons, 16384)

        # Memory info
        seq_len = tokens.shape[1]
        # KV cache: 2 * n_layers * seq_len * d_model * bytes_per_float
        n_layers = TF_MODEL.cfg.n_layers
        d_model = TF_MODEL.cfg.d_model
        kv_bytes = 2 * n_layers * seq_len * d_model * 4
        current, peak = tracemalloc.get_traced_memory()

        msg = {
            "type": "token",
            "model": "transformer",
            "character": char,
            "xy_vis": vis[:200],  # send subset for perf
            "mem_info": {
                "model": "transformer",
                "seq_len": int(seq_len),
                "bytes": kv_bytes,
            }
        }

        from starlette.websockets import WebSocketState
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.send_text(json.dumps(msg))
        else:
            break

        await manager.send_json(ws, {
            "type": "log",
            "model": "transformer",
            "text": f"token {generated_count}: '{char}' | KV={kv_bytes} B | seq={seq_len}"
        })
        await asyncio.sleep(0.05)
    tracemalloc.stop()
    return generated_count


# ──────────────────────────────────────────
# Hebbian Constants
# ──────────────────────────────────────────
HEBB_ALPHA = 0.5
HEBB_ETA = 0.05
HEBB_DECAY = 0.005
HEBB_MAX = 5.0


# ──────────────────────────────────────────
# Connection Manager
# ──────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_json(self, websocket: WebSocket, data: dict):
        await websocket.send_text(json.dumps(data))

    async def send_bytes(self, websocket: WebSocket, data: bytes):
        await websocket.send_bytes(data)

manager = ConnectionManager()


# ──────────────────────────────────────────
# Word Reconstruction Helpers
# ──────────────────────────────────────────
def build_word_spans(byte_chars):
    """Group byte characters into word spans based on whitespace/punctuation."""
    spans = []
    current_word = ""
    start_idx = 0
    for i, ch in enumerate(byte_chars):
        if ch in (' ', '\t', '\n', '\r'):
            if current_word:
                spans.append({"word": current_word, "start": start_idx, "end": i - 1})
                current_word = ""
        elif ch in ('.',',','!','?',';',':','"',"'", '(', ')', '[', ']', '{', '}', '/', '\\'):
            if current_word:
                spans.append({"word": current_word, "start": start_idx, "end": i - 1})
                current_word = ""
            spans.append({"word": ch, "start": i, "end": i})
        else:
            if not current_word:
                start_idx = i
            current_word += ch
    if current_word:
        spans.append({"word": current_word, "start": start_idx, "end": len(byte_chars) - 1})
    return spans


# ──────────────────────────────────────────
# Session State (merged)
# ──────────────────────────────────────────
class SessionState:
    def __init__(self, model: bdh.BDH):
        self.model = model
        C = model.config
        nh = C.n_head
        D = C.n_embd
        N = D * C.mlp_internal_dim_multiplier // nh
        total_n = nh * N

        # Inference state
        self.ablation_indices: Set[int] = set()
        self.history_tokens: torch.Tensor = torch.zeros((1, 0), dtype=torch.long, device=DEVICE)
        self.temperature: float = 0.7
        self.top_k: int = 10
        self.activation_threshold: float = 0.0

        # Hebbian state
        self.hebb_memory = torch.zeros(total_n, device=DEVICE, dtype=DTYPE)
        self.hebb_learn = False
        self.hebb_apply = False
        self.hebb_persist = False

        # Neuron profiling state (from BDH_Proto)
        self.neuron_word_map = defaultdict(list)
        self.xy_timeline = []
        self.token_chars = []
        self.word_spans = []
        self.sentences = []
        self.current_sentence_tokens = []

    def reset(self):
        self.history_tokens = torch.zeros((1, 0), dtype=torch.long, device=DEVICE)
        self.ablation_indices.clear()
        self.reset_hebb()
        self.reset_profiling()

    def reset_hebb(self):
        self.hebb_memory.zero_()

    def reset_profiling(self):
        self.neuron_word_map.clear()
        self.xy_timeline.clear()
        self.token_chars.clear()
        self.word_spans.clear()
        self.sentences.clear()
        self.current_sentence_tokens.clear()

    def set_ablation(self, indices: List[int]):
        self.ablation_indices = set(indices)

    def toggle_ablation(self, index: int):
        if index in self.ablation_indices:
            self.ablation_indices.discard(index)
        else:
            self.ablation_indices.add(index)
        return index in self.ablation_indices

    def _update_neuron_word_map(self, word, word_vec):
        """Update neuron -> word mapping with word-level activations."""
        top_vals, top_ids = torch.topk(word_vec, min(200, word_vec.shape[0]))
        for val, nid in zip(top_vals.tolist(), top_ids.tolist()):
            if val <= 0:
                continue
            entries = self.neuron_word_map[nid]
            found = False
            for e in entries:
                if e[0] == word:
                    e[1] = max(e[1], val)
                    found = True
                    break
            if not found:
                entries.append([word, val])
            entries.sort(key=lambda x: -x[1])
            if len(entries) > 50:
                del entries[50:]

    def get_neuron_profile(self, neuron_id):
        """Build full neuron profile for the dashboard."""
        nid = neuron_id

        # A. Top triggering words
        top_words = []
        if nid in self.neuron_word_map:
            for word, score in self.neuron_word_map[nid][:20]:
                top_words.append({"word": word, "score": round(score, 4)})

        # B. Activation timeline
        activation_timeline = []
        for xy in self.xy_timeline:
            if nid < len(xy):
                activation_timeline.append(round(xy[nid].item(), 4))
            else:
                activation_timeline.append(0.0)

        # C. Word-level contribution breakdown
        word_contributions = []
        for span in self.word_spans:
            s, e = span["start"], span["end"]
            if s < len(self.xy_timeline) and e < len(self.xy_timeline):
                max_act = 0.0
                for t in range(s, e + 1):
                    if t < len(self.xy_timeline) and nid < len(self.xy_timeline[t]):
                        max_act = max(max_act, self.xy_timeline[t][nid].item())
                word_contributions.append({"word": span["word"], "score": round(max_act, 4)})
        word_contributions.sort(key=lambda x: -x["score"])
        word_contributions = word_contributions[:15]

        # D. Co-firing neurons
        co_fire_scores = defaultdict(float)
        for xy in self.xy_timeline:
            if nid < len(xy) and xy[nid].item() > 0.1:
                top_vals, top_ids = torch.topk(xy, min(50, xy.shape[0]))
                for v, oid in zip(top_vals.tolist(), top_ids.tolist()):
                    if oid != nid and v > 0.1:
                        co_fire_scores[oid] += v
        co_firing = sorted(co_fire_scores.items(), key=lambda x: -x[1])[:20]
        co_firing = [{"id": cid, "score": round(sc, 4)} for cid, sc in co_firing]

        # E. Top triggering sentences
        top_sentences = self.sentences[:5] if self.sentences else []

        # F. Concept label
        concept_label = ""
        if top_words:
            concept_label = " / ".join([w["word"] for w in top_words[:3]])

        return {
            "type": "neuron_profile",
            "neuron_id": nid,
            "concept_label": concept_label,
            "top_words": top_words,
            "activation_timeline": activation_timeline,
            "word_contributions": word_contributions,
            "co_firing": co_firing,
            "top_sentences": top_sentences,
            "token_chars": self.token_chars[-200:],
        }


# ──────────────────────────────────────────
# WebSocket Endpoint (merged)
# ──────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    session = SessionState(MODEL)
    logger.info("Client connected")

    C = MODEL.config
    nh = C.n_head
    D = C.n_embd
    N = D * C.mlp_internal_dim_multiplier // nh
    total_n = nh * N

    try:
        # Send config to client on connect
        await manager.send_json(websocket, {
            "type": "config",
            "n_head": nh,
            "n_embd": D,
            "n_per_head": N,
            "total_neurons": total_n,
            "n_layer": C.n_layer,
        })

        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type", "")

            # ── Reset ──
            if msg_type == "reset":
                session.reset()
                await manager.send_json(websocket, {"status": "reset"})
                continue

            # ── Neuron Profile ──
            if msg_type == "neuron_profile":
                neuron_id = data.get("neuron_id", 0)
                profile = session.get_neuron_profile(neuron_id)
                await manager.send_json(websocket, profile)
                continue

            # ── Hebbian Controls ──
            if msg_type == "hebb_learn":
                session.hebb_learn = data.get("value", False)
                logger.info(f"Hebbian learn: {session.hebb_learn}")
                continue
            if msg_type == "hebb_apply":
                session.hebb_apply = data.get("value", False)
                logger.info(f"Hebbian apply: {session.hebb_apply}")
                continue
            if msg_type == "hebb_persist":
                session.hebb_persist = data.get("value", False)
                logger.info(f"Hebbian persist: {session.hebb_persist}")
                continue
            if msg_type == "reset_hebb":
                session.reset_hebb()
                mem_max = session.hebb_memory.max().item()
                await manager.send_json(websocket, {"type": "hebb_status", "mem_max": mem_max, "mem_mean": 0.0})
                continue

            # ── Ablation Controls ──
            if msg_type == "toggle_ablation":
                idx = data.get("index", 0)
                is_ablated = session.toggle_ablation(idx)
                await manager.send_json(websocket, {
                    "type": "ablation_update",
                    "index": idx,
                    "ablated": is_ablated,
                    "ablation_list": sorted(session.ablation_indices),
                })
                continue
            if msg_type == "set_ablation":
                indices = data.get("indices", [])
                session.set_ablation(indices)
                await manager.send_json(websocket, {
                    "type": "ablation_update",
                    "ablation_list": sorted(session.ablation_indices),
                })
                continue
            if msg_type == "clear_ablation":
                session.ablation_indices.clear()
                await manager.send_json(websocket, {
                    "type": "ablation_update",
                    "ablation_list": [],
                })
                continue

            # ── Temperature / Top-K ──
            if msg_type == "set_temperature":
                session.temperature = max(0.1, min(2.0, data.get("value", 0.7)))
                continue
            if msg_type == "set_topk":
                session.top_k = max(1, min(256, data.get("value", 10)))
                continue

            # ── Activation Threshold ──
            if msg_type == "threshold":
                session.activation_threshold = max(0.0, data.get("value", 0.0))
                logger.info(f"Activation threshold: {session.activation_threshold}")
                continue

            # ── Generate (prompt) ──
            if msg_type == "prompt":
                prompt_text = data.get("text", "")
                if not prompt_text:
                    continue

                model_type = data.get("model_type", "bdh")

                # ── COMPARISON MODE ──
                if model_type == "both":
                    logger.info(f"Comparison mode: '{prompt_text}'")

                    # 1) Run Transformer first
                    tf_count = await stream_transformer_tokens(websocket, prompt_text, max_tokens=50)
                    logger.info(f"Transformer generated {tf_count} tokens")

                    # 2) Run BDH for the same number of tokens
                    bdh_tokens_generated = 0
                    prompt_bytes = bytearray(prompt_text, "utf-8")
                    bdh_prompt_tokens = torch.tensor(
                        [list(prompt_bytes)], dtype=torch.long, device=DEVICE
                    )
                    C_cmp = MODEL.config
                    nh_cmp = C_cmp.n_head
                    D_cmp = C_cmp.n_embd
                    N_cmp = D_cmp * C_cmp.mlp_internal_dim_multiplier // nh_cmp

                    torch.manual_seed(42)
                    tracemalloc.start()
                    for step_cmp in range(tf_count):
                        idx_cmp = bdh_prompt_tokens
                        Bc, Tc = idx_cmp.size()
                        with torch.no_grad():
                            xc = MODEL.embed(idx_cmp).unsqueeze(1)
                            xc = MODEL.ln(xc)
                            for level_c in range(C_cmp.n_layer):
                                xl = xc @ MODEL.encoder
                                xs = F.relu(xl)
                                ykv = MODEL.attn(Q=xs, K=xs, V=xc)
                                ykv = MODEL.ln(ykv)
                                yl = ykv @ MODEL.encoder_v
                                ys = F.relu(yl)
                                xys = xs * ys
                                xys = MODEL.drop(xys)
                                ymlp = xys.transpose(1, 2).reshape(Bc, 1, Tc, N_cmp * nh_cmp) @ MODEL.decoder
                                yc = MODEL.ln(ymlp)
                                xc = MODEL.ln(xc + yc)
                            logits_c = xc.view(Bc, Tc, D_cmp) @ MODEL.lm_head

                        logits_last_c = logits_c[:, -1, :]
                        probs_c = F.softmax(logits_last_c / 0.7, dim=-1)
                        kc = min(10, probs_c.size(-1))
                        v_c, _ = torch.topk(probs_c, kc)
                        min_v_c = v_c[:, [-1]]
                        probs_c = torch.where(probs_c < min_v_c, torch.zeros_like(probs_c), probs_c)
                        probs_c = probs_c / probs_c.sum(dim=-1, keepdim=True)
                        next_c = torch.multinomial(probs_c, num_samples=1)
                        bdh_prompt_tokens = torch.cat((bdh_prompt_tokens, next_c), dim=1)
                        bdh_tokens_generated += 1

                        # Decode character
                        tb = next_c.item()
                        ch = chr(tb) if 32 <= tb <= 126 else ('\n' if tb == 10 else '?')

                        # Sparse activations for BDH globe
                        flat_xy = (xs * ys)[:, :, -1, :].reshape(-1).float().detach().cpu()
                        # Normalize for visualization
                        mx = flat_xy.max().item()
                        vis_list = (flat_xy / max(mx, 1e-6)).tolist()

                        # BDH memory: constant O(1)
                        bdh_mem = nh_cmp * D_cmp * N_cmp * 4  # fixed param size
                        current_b, peak_b = tracemalloc.get_traced_memory()

                        msg_bdh = {
                            "type": "token",
                            "model": "bdh",
                            "character": ch,
                            "xy_vis": vis_list[:200],
                            "mem_info": {
                                "model": "bdh",
                                "seq_len": bdh_tokens_generated,
                                "bytes": bdh_mem,
                            }
                        }
                        from starlette.websockets import WebSocketState
                        if websocket.application_state == WebSocketState.CONNECTED:
                            await websocket.send_text(json.dumps(msg_bdh))
                        else:
                            break

                        await manager.send_json(websocket, {
                            "type": "log",
                            "model": "bdh",
                            "text": f"token {bdh_tokens_generated}: '{ch}' | mem={bdh_mem} B | O(1)"
                        })
                        await asyncio.sleep(0.05)

                    tracemalloc.stop()
                    await manager.send_json(websocket, {"status": "done"})
                    logger.info(f"Comparison complete: TF={tf_count}, BDH={bdh_tokens_generated}")
                    continue

                if not session.hebb_persist:
                    session.reset_hebb()

                # Encode prompt
                prompt_bytes = bytearray(prompt_text, "utf-8")
                prompt_tokens = torch.tensor(
                    [list(prompt_bytes)], dtype=torch.long, device=DEVICE
                )
                session.history_tokens = prompt_tokens

                prompt_chars = list(prompt_text)
                session.current_sentence_tokens = list(prompt_text)
                generated_chars = []
                ablation_list = sorted(session.ablation_indices) if session.ablation_indices else []

                # Seed RNG for deterministic output (same prompt+params → same output)
                torch.manual_seed(42)

                for step in range(100):  # max tokens
                    idx = session.history_tokens
                    B, T = idx.size()

                    with torch.no_grad():
                        x = MODEL.embed(idx).unsqueeze(1)
                        x = MODEL.ln(x)

                        last_xy_sparse = None

                        for level in range(C.n_layer):
                            x_latent = x @ MODEL.encoder
                            x_sparse = F.relu(x_latent)

                            yKV = MODEL.attn(Q=x_sparse, K=x_sparse, V=x)
                            yKV = MODEL.ln(yKV)

                            y_latent = yKV @ MODEL.encoder_v
                            y_sparse = F.relu(y_latent)
                            xy_sparse = x_sparse * y_sparse  # [B, nh, T, N]

                            # Ablation at every layer
                            if ablation_list:
                                mask = torch.ones(nh * N, device=xy_sparse.device, dtype=xy_sparse.dtype)
                                mask[ablation_list] = 0.0
                                xy_sparse = xy_sparse * mask.view(nh, N).unsqueeze(0).unsqueeze(2)

                            if level == C.n_layer - 1:
                                # Activation threshold
                                if session.activation_threshold > 0:
                                    xy_sparse = xy_sparse.clone()
                                    xy_sparse[xy_sparse < session.activation_threshold] = 0.0

                                # Capture last-token activations (pre-Hebbian, post-ablation)
                                last_xy_sparse = xy_sparse[:, :, -1:, :].detach()  # [B, nh, 1, N]

                                # Stats accumulation (consolidation) — before Hebbian
                                if getattr(MODEL, "accumulate_stats", False):
                                    current_stats = xy_sparse.detach().sum(dim=(0, 2))
                                    if not hasattr(MODEL, "xy_sparse_stats"):
                                        MODEL.xy_sparse_stats = torch.zeros_like(current_stats)
                                        MODEL.stats_count = 0
                                    MODEL.xy_sparse_stats += current_stats
                                    MODEL.stats_count += (B * T)

                                # Hebbian modulation (last token only)
                                if session.hebb_apply:
                                    mem = session.hebb_memory.reshape(1, nh, 1, N).to(
                                        device=xy_sparse.device, dtype=xy_sparse.dtype
                                    )
                                    xy_sparse = xy_sparse.clone()
                                    xy_sparse[:, :, -1:, :] = xy_sparse[:, :, -1:, :] * (1.0 + HEBB_ALPHA * mem)

                            xy_sparse = MODEL.drop(xy_sparse)
                            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ MODEL.decoder
                            y = MODEL.ln(yMLP)
                            x = MODEL.ln(x + y)

                        logits = x.view(B, T, D) @ MODEL.lm_head

                    # Hebbian memory update
                    flat_act = last_xy_sparse.reshape(-1).float()
                    if session.hebb_learn:
                        session.hebb_memory = session.hebb_memory + HEBB_ETA * flat_act
                        session.hebb_memory = session.hebb_memory * (1.0 - HEBB_DECAY)
                        session.hebb_memory = torch.clamp(session.hebb_memory, 0.0, HEBB_MAX)

                    # Flat activations for globe: [nh, N] → [nh*N]
                    flat_activations = flat_act.detach().cpu()

                    # Sampling
                    logits_last = logits[:, -1, :]
                    probs = F.softmax(logits_last / session.temperature, dim=-1)
                    k = min(session.top_k, probs.size(-1))
                    v, _ = torch.topk(probs, k)
                    min_v = v[:, [-1]]
                    probs = torch.where(probs < min_v, torch.zeros_like(probs), probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    next_idx = torch.multinomial(probs, num_samples=1)

                    session.history_tokens = torch.cat((session.history_tokens, next_idx), dim=1)

                    # Decode character
                    token_byte = next_idx.item()
                    if 32 <= token_byte <= 126:
                        token_str = chr(token_byte)
                    elif token_byte == 10:
                        token_str = "\n"
                    elif token_byte == 9:
                        token_str = "\t"
                    else:
                        token_str = ""

                    # Store in profiling timeline
                    session.xy_timeline.append(flat_activations.clone())
                    session.token_chars.append(token_str)
                    generated_chars.append(token_str)
                    session.current_sentence_tokens.append(token_str)

                    # Word reconstruction
                    all_chars = prompt_chars + generated_chars
                    word_spans = build_word_spans(all_chars)
                    session.word_spans = word_spans

                    # Check if word just completed
                    current_word = None
                    word_complete = False
                    current_word_idx = len(word_spans) - 1 if word_spans else -1

                    if word_spans:
                        if token_str in (' ', '\t', '\n', '.', ',', '!', '?', ';', ':') and current_word_idx > 0:
                            prev_span = word_spans[-2] if len(word_spans) >= 2 else word_spans[-1]
                            current_word = prev_span["word"]
                            word_complete = True
                            s_idx = prev_span["start"] - len(prompt_chars)
                            e_idx = prev_span["end"] - len(prompt_chars)
                            if s_idx >= 0 and e_idx >= 0:
                                start_t = max(0, s_idx)
                                end_t = min(len(session.xy_timeline) - 1, e_idx)
                                if start_t <= end_t and start_t < len(session.xy_timeline):
                                    word_vecs = torch.stack([session.xy_timeline[t] for t in range(start_t, end_t + 1)])
                                    word_vec = word_vecs.max(dim=0).values
                                    session._update_neuron_word_map(current_word, word_vec)
                        else:
                            last_span = word_spans[-1]
                            current_word = last_span["word"]

                    # Top neurons
                    top_vals, top_ids = torch.topk(flat_activations, min(100, flat_activations.shape[0]))
                    top_neurons = [{"id": i.item(), "value": round(v.item(), 4)} for i, v in zip(top_ids, top_vals) if v.item() > 0]

                    # Hebbian status
                    mem_max = session.hebb_memory.max().item()
                    mem_mean = session.hebb_memory.mean().item()
                    firing_count = int((flat_activations > 0.01).sum().item())

                    # Send token message
                    token_index = len(session.xy_timeline) - 1
                    message = {
                        "type": "token",
                        "token_index": token_index,
                        "character": token_str,
                        "word_index": current_word_idx,
                        "word_string": current_word or "",
                        "word_complete": word_complete,
                        "xy_vis": flat_activations.tolist(),
                        "top_neurons": top_neurons[:100],
                        "hebb_mem_max": round(mem_max, 4),
                        "hebb_mem_mean": round(mem_mean, 4),
                        "firing_count": firing_count,
                    }

                    from starlette.websockets import WebSocketState
                    if websocket.application_state == WebSocketState.CONNECTED:
                        await websocket.send_text(json.dumps(message))
                    else:
                        break

                    if token_byte == 10:
                        break
                    await asyncio.sleep(0.05)

                # Store sentence
                full_sentence = "".join(session.current_sentence_tokens)
                session.sentences.append(full_sentence)

                # Final word aggregation
                if session.word_spans:
                    last_span = session.word_spans[-1]
                    s_idx = last_span["start"] - len(prompt_chars)
                    e_idx = last_span["end"] - len(prompt_chars)
                    if s_idx >= 0 and e_idx >= 0:
                        start_t = max(0, s_idx)
                        end_t = min(len(session.xy_timeline) - 1, e_idx)
                        if start_t <= end_t and start_t < len(session.xy_timeline):
                            word_vecs = torch.stack([session.xy_timeline[t] for t in range(start_t, end_t + 1)])
                            word_vec = word_vecs.max(dim=0).values
                            session._update_neuron_word_map(last_span["word"], word_vec)

                # Signal done
                from starlette.websockets import WebSocketState
                if websocket.application_state == WebSocketState.CONNECTED:
                    await manager.send_json(websocket, {"status": "done"})
                continue

            # ── Experiment B ──
            if msg_type == "experiment_b":
                logger.info(f"Exp B: Starting... prompt={data.get('prompt','(default)')!r}, teach={data.get('teach','(default)')!r}")
                prompt_q = data.get("prompt", "A = ")
                teach_text = data.get("teach", "A = 1 B = 2 C = 3\n")
                teach_reps = data.get("reps", 50)
                gen_len = data.get("gen_len", 30)
                exp_temp = data.get("temperature", 0.5)
                ablation_list = sorted(session.ablation_indices) if session.ablation_indices else []

                # Helper: forward pass for activations
                def _forward_activations(prompt_str, apply_hebb=False):
                    tokens = torch.tensor(
                        bytearray(prompt_str, "utf-8"), dtype=torch.long, device=DEVICE
                    ).unsqueeze(0)
                    x = MODEL.embed(tokens).unsqueeze(1)
                    x = MODEL.ln(x)
                    Bt, Tt = tokens.size()
                    for level in range(C.n_layer):
                        x_latent = x @ MODEL.encoder
                        x_sparse = F.relu(x_latent)
                        yKV = MODEL.attn(Q=x_sparse, K=x_sparse, V=x)
                        yKV = MODEL.ln(yKV)
                        y_latent = yKV @ MODEL.encoder_v
                        y_sparse = F.relu(y_latent)
                        xy_s = x_sparse * y_sparse
                        if ablation_list:
                            m = torch.ones(nh * N, device=xy_s.device, dtype=xy_s.dtype)
                            m[ablation_list] = 0.0
                            xy_s = xy_s * m.view(nh, N).unsqueeze(0).unsqueeze(2)
                        if level == C.n_layer - 1:
                            if session.activation_threshold > 0:
                                xy_s = xy_s.clone()
                                xy_s[xy_s < session.activation_threshold] = 0.0
                            if apply_hebb and session.hebb_apply:
                                mem = session.hebb_memory.reshape(1, nh, 1, N).to(device=xy_s.device, dtype=xy_s.dtype)
                                xy_s = xy_s.clone()
                                xy_s[:, :, -1:, :] = xy_s[:, :, -1:, :] * (1.0 + HEBB_ALPHA * mem)
                        xy_s = MODEL.drop(xy_s)
                        yMLP = xy_s.transpose(1, 2).reshape(Bt, 1, Tt, N * nh) @ MODEL.decoder
                        y = MODEL.ln(yMLP)
                        x = MODEL.ln(x + y)
                    logits = x.view(Bt, Tt, D) @ MODEL.lm_head
                    last_act = xy_s[:, :, -1, :].reshape(-1)
                    return last_act, logits

                # Helper: generate text
                def _generate_text(prompt_str, apply_hebb, temp, n_tokens):
                    torch.manual_seed(42)  # deterministic generation
                    tokens = torch.tensor(
                        bytearray(prompt_str, "utf-8"), dtype=torch.long, device=DEVICE
                    ).unsqueeze(0)
                    result_chars = []
                    for _ in range(n_tokens):
                        Bt, Tt = tokens.size()
                        x = MODEL.embed(tokens).unsqueeze(1)
                        x = MODEL.ln(x)
                        for level in range(C.n_layer):
                            x_latent = x @ MODEL.encoder
                            x_sparse = F.relu(x_latent)
                            yKV = MODEL.attn(Q=x_sparse, K=x_sparse, V=x)
                            yKV = MODEL.ln(yKV)
                            y_latent = yKV @ MODEL.encoder_v
                            y_sparse = F.relu(y_latent)
                            xy_s = x_sparse * y_sparse
                            if ablation_list:
                                m = torch.ones(nh * N, device=xy_s.device, dtype=xy_s.dtype)
                                m[ablation_list] = 0.0
                                xy_s = xy_s * m.view(nh, N).unsqueeze(0).unsqueeze(2)
                            if level == C.n_layer - 1 and apply_hebb and session.hebb_apply:
                                mem = session.hebb_memory.reshape(1, nh, 1, N).to(device=xy_s.device, dtype=xy_s.dtype)
                                xy_s = xy_s.clone()
                                xy_s[:, :, -1:, :] = xy_s[:, :, -1:, :] * (1.0 + HEBB_ALPHA * mem)
                            xy_s = MODEL.drop(xy_s)
                            yMLP = xy_s.transpose(1, 2).reshape(Bt, 1, Tt, N * nh) @ MODEL.decoder
                            y = MODEL.ln(yMLP)
                            x = MODEL.ln(x + y)
                        logits = x.view(Bt, Tt, D) @ MODEL.lm_head
                        logits_last = logits[:, -1, :] / temp
                        probs = F.softmax(logits_last, dim=-1)
                        nxt = torch.multinomial(probs, 1)
                        tokens = torch.cat([tokens, nxt], dim=1)
                        b = nxt.item()
                        result_chars.append(chr(b) if 32 <= b <= 126 else ('\\n' if b == 10 else '?'))
                    return "".join(result_chars)

                # Helper: teach one rep
                def _teach_one(text_str):
                    tokens = torch.tensor(
                        bytearray(text_str, "utf-8"), dtype=torch.long, device=DEVICE
                    ).unsqueeze(0)
                    Bt, Tt = tokens.size()
                    x = MODEL.embed(tokens).unsqueeze(1)
                    x = MODEL.ln(x)
                    for level in range(C.n_layer):
                        x_latent = x @ MODEL.encoder
                        x_sparse = F.relu(x_latent)
                        yKV = MODEL.attn(Q=x_sparse, K=x_sparse, V=x)
                        yKV = MODEL.ln(yKV)
                        y_latent = yKV @ MODEL.encoder_v
                        y_sparse = F.relu(y_latent)
                        xy_s = x_sparse * y_sparse
                        if ablation_list:
                            m = torch.ones(nh * N, device=xy_s.device, dtype=xy_s.dtype)
                            m[ablation_list] = 0.0
                            xy_s = xy_s * m.view(nh, N).unsqueeze(0).unsqueeze(2)
                        if level == C.n_layer - 1:
                            if session.activation_threshold > 0:
                                xy_s = xy_s.clone()
                                xy_s[xy_s < session.activation_threshold] = 0.0
                        xy_s = MODEL.drop(xy_s)
                        yMLP = xy_s.transpose(1, 2).reshape(Bt, 1, Tt, N * nh) @ MODEL.decoder
                        y = MODEL.ln(yMLP)
                        x = MODEL.ln(x + y)
                    act = xy_s[:, :, -1, :].reshape(-1).float()
                    session.hebb_memory = session.hebb_memory + HEBB_ETA * act
                    session.hebb_memory = session.hebb_memory * (1.0 - HEBB_DECAY)
                    session.hebb_memory = torch.clamp(session.hebb_memory, 0.0, HEBB_MAX)

                # ── Run Experiment ──
                await manager.send_json(websocket, {"type": "exp_phase", "phase": "baseline"})

                baseline_act, _ = _forward_activations(prompt_q, apply_hebb=False)
                baseline_text = _generate_text(prompt_q, False, exp_temp, gen_len)

                await manager.send_json(websocket, {
                    "type": "exp_progress",
                    "phase": "baseline",
                    "text": baseline_text,
                })

                # Teach phase
                await manager.send_json(websocket, {"type": "exp_phase", "phase": "teaching"})
                for rep in range(teach_reps):
                    _teach_one(teach_text)
                    if rep % 10 == 0:
                        await manager.send_json(websocket, {
                            "type": "exp_progress",
                            "phase": "teaching",
                            "rep": rep,
                            "total": teach_reps,
                            "mem_max": round(session.hebb_memory.max().item(), 4),
                            "mem_mean": round(session.hebb_memory.mean().item(), 4),
                        })
                        await asyncio.sleep(0)

                # Test phase
                await manager.send_json(websocket, {"type": "exp_phase", "phase": "testing"})
                test_act, _ = _forward_activations(prompt_q, apply_hebb=True)
                test_text = _generate_text(prompt_q, True, exp_temp, gen_len)

                # Metrics
                cos_sim = F.cosine_similarity(baseline_act.unsqueeze(0), test_act.unsqueeze(0)).item()
                n_changed = int((torch.abs(test_act - baseline_act) > 0.01).sum().item())

                result = {
                    "type": "experiment_result",
                    "baseline_text": baseline_text,
                    "test_text": test_text,
                    "cosine_similarity": round(cos_sim, 4),
                    "changed_neurons": n_changed,
                    "mem_max": round(session.hebb_memory.max().item(), 4),
                    "mem_mean": round(session.hebb_memory.mean().item(), 4),
                    "mem_sparsity": round(int((session.hebb_memory > 0.01).sum()) / total_n * 100, 1),
                    "teach_reps": teach_reps,
                    "prompt": prompt_q,
                    "teach_text": teach_text,
                }
                await manager.send_json(websocket, result)
                logger.info(f"Exp B: Complete. cosine_sim={cos_sim:.4f}, changed={n_changed}")
                continue

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected")


# ──────────────────────────────────────────
# Consolidation Endpoint
# ──────────────────────────────────────────
@app.post("/consolidate")
async def consolidate_endpoint():
    """Triggers Option-B consolidation and returns diagnostics."""
    try:
        if not hasattr(MODEL, "xy_sparse_stats") or MODEL.stats_count == 0:
            return {"status": "skipped", "reason": "No stats accumulated. Generate some text first."}
        result = MODEL.consolidate()
        logger.info(f"Consolidation: status={result.get('status')}, "
                     f"active_frac={result.get('active_fraction', 0):.4f}, "
                     f"tokens={result.get('accumulated_tokens', 0)}")
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))


# Enable stats accumulation
MODEL.accumulate_stats = True
logger.info("Stats accumulation enabled.")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port)
