from models import GPTJForCausalLMWithHooks
from transformers import AutoTokenizer
import torch
import time

# for future ref:
# 250s to run, 25GB VRAM usage

start = time.time()

# Track a single layer
TRACK_LAYER = 0

print("Loading model...")
model = GPTJForCausalLMWithHooks.from_pretrained(
    "EleutherAI/gpt-j-6B", 
    track_layer_idx=TRACK_LAYER,
    dtype=torch.float32,  # float16
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

config = model.config
hidden_dim = config.hidden_size
max_context_length = config.n_positions
print(f"Hidden Dim: {hidden_dim}") # 4096
print(f"Max Context Length: {max_context_length}") #2048

print("Setting up gradient computation...")
# CRITICAL: Disable gradients for ALL parameters
for param in model.parameters():
    param.requires_grad = False

# ONLY enable gradients for the tracked layer's Q, K, V projections
tracked_attn = model.transformer.h[TRACK_LAYER].attn
tracked_attn.q_proj.weight.requires_grad = True
tracked_attn.k_proj.weight.requires_grad = True
tracked_attn.v_proj.weight.requires_grad = True

# Setup optimizer - ONLY for the 3 parameters we're tracking
optimizer = torch.optim.AdamW([
    tracked_attn.q_proj.weight,
    tracked_attn.k_proj.weight,
    tracked_attn.v_proj.weight
], lr=1e-5)

text = "This is transformer exercise"
inp = tokenizer(text, return_tensors="pt")
model.train(True)

# Forward pass (in float16)
print("Running forward pass...")
output = model(**inp)
loss = output.logits.sum()

# Backward pass - only computes gradients for Q, K, V of tracked layer
print("Running backward pass...")
optimizer.zero_grad()
loss.backward()

# Optimizer step
optimizer.step()

# Capture weights after optimizer
model.capture_weights_after_optimizer()

# Print summary
model.print_summary()

elapsed = time.time() - start
print(f"\nTime taken: {elapsed:.2f} seconds")

# Clean up
model.remove_all_hooks()