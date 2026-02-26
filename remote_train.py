import os
import time
import math
import torch
import torch.nn.functional as F
from bdh import BDH, BDHConfig

# 1. 100% COMPATIBLE ARCHITECTURE CONFIG
# DO NOT CHANGE THESE OR THE WEIGHTS WILL NOT LOAD
config = BDHConfig(
    n_layer=4,
    n_embd=256,
    n_head=4,
    mlp_internal_dim_multiplier=64,
    vocab_size=256,
    dropout=0.1
)

# 2. HYPERPARAMETERS (Adjust to your remote GPU power)
BATCH_SIZE = 64
BLOCK_SIZE = 256
LEARNING_RATE = 1e-3
MAX_STEPS = 5000 # Increase this for better training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(data, block_size, batch_size):
    # Standard random sampling for character LM
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

def train(dataset_path):
    print(f"Loading data from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        text_bytes = f.read()
    # Byte-level encoding (matches visualizer's vocab 256)
    data = torch.tensor(list(text_bytes), dtype=torch.long)
    
    model = BDH(config).to(DEVICE)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")
    print(f"Total Neurons: {config.n_head * (config.n_embd * config.mlp_internal_dim_multiplier // config.n_head)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
    
    model.train()
    start_time = time.time()
    
    for step in range(MAX_STEPS):
        xb, yb = get_batch(data, BLOCK_SIZE, BATCH_SIZE)
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            dt = time.time() - start_time
            print(f"Step {step}: loss {loss.item():.4f}, time {dt:.2f}s")
            start_time = time.time()

    # 3. SAVE IN VISUALIZER-COMPATIBLE FORMAT
    # The 'config' object MUST be inside the checkpoint for the server to load it.
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,  
        'optimizer_state_dict': optimizer.state_dict(),
        'step': MAX_STEPS,
    }
    
    output_path = "bdh_wikipedia_final.pt"
    torch.save(checkpoint, output_path)
    print(f"\n--- SUCCESS ---")
    print(f"Training complete. Weights saved to: {output_path}")
    print("Replace your local bdh_wikipedia_final.pt with this file.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset.txt", help="Path to text file")
    args = parser.parse_args()
    
    if os.path.exists(args.data):
        train(args.data)
    else:
        print(f"Error: {args.data} not found. Please provide a text file.")
