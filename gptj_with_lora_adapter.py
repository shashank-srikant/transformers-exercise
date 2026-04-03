from transformers import GPTJForCausalLM, GPTJConfig, GPT2TokenizerFast
import torch
from torch import nn

# Auto Model Loading
# from transformers import AutoModelForCausalLM, AutoTokenizer
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-j-6B")

# for name, module in model.named_modules():
#     print(name, module)


# freeze original weights
for param in model.parameters():
    param.requires_grad = False

def add_lora_adapter(input_linear_layer, alpha, rank):
    """
    Adds LoRA layer corresponding to the Linear Layer.
    """
    input_linear_layer.A = nn.Parameter(torch.randn((input_linear_layer.in_features, rank)))
    input_linear_layer.B = nn.Parameter(torch.zeros((rank, input_linear_layer.out_features)))
    
    input_forward = input_linear_layer.forward
    def lora_forward(self, x):
        orig_out = input_forward(x)
        result = orig_out + (x@self.A)@self.B * (alpha/rank)
        return result

    input_linear_layer.forward = lora_forward.__get__(input_linear_layer, nn.Linear)

alpha = 4
rank = 2
epochs = 3

# add layers
for layer_idx in range(len(model.transformer.h)):
    attn = model.transformer.h[layer_idx].attn
    add_lora_adapter(attn.q_proj, alpha, rank)
    add_lora_adapter(attn.k_proj, alpha, rank)
    add_lora_adapter(attn.v_proj, alpha, rank)
    add_lora_adapter(attn.out_proj, alpha, rank)


# params info
trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad] # 28 layers, 4 projs each, 2 matrices each = 224 params tensors
print(f"\nTrainable parameter tensors: {len(trainable)}")
for n, s in trainable[:5]:
    print(n, s)
print('.....')

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n Trainable parameters: {trainable_params}")

frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Frozen parameters: {frozen_params}")

train_data  = ["This is the first sample that i will use to train lora adapter. Bless me!", "This is the second sample."]

tokenizer.pad_token = tokenizer.eos_token
inp = tokenizer(train_data, return_tensors="pt", padding=True, truncation=True, max_length=64)
inp["labels"] = inp["input_ids"].clone()

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4
)

model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(**inp)
    loss = output.loss
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    loss.backward()
    optimizer.step()