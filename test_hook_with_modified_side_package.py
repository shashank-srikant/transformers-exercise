from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
import time

# model = AutoModel.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

start = time.time()
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

text = "This is transformer exercise"
inp = tokenizer(text, return_tensors="pt")
model.train(True)

output = model(**inp) # forward pass
loss = output.logits.sum()
loss.backward() # backward pass

# verifying grads of Q.K,V weights for a layer
# att = model.transformer.h[0].attn
# print("Q weight.grad[0,:5]:", att.q_proj.weight.grad[0,:5] if att.q_proj.weight.grad is not None else None)
# print("K weight.grad[0,:5]:", att.k_proj.weight.grad[0,:5] if att.k_proj.weight.grad is not None else None)
# print("V weight.grad[0,:5]:", att.v_proj.weight.grad[0,:5] if att.v_proj.weight.grad is not None else None)


elapsed = time.time() - start
print(f"\nTime taken: {elapsed:.2f} seconds") # 6-7 min

# clean hooks
for block in model.transformer.h:
    block.attn.remove_hooks()
