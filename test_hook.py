from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
import time

start = time.time()
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

text = "This is transformer exercise"
inp = tokenizer(text, return_tensors="pt")
model.train(True)

cache = {
    "q_out": None, "k_out": None, "v_out": None,
    "q_grad_in": None, "k_grad_in": None, "v_grad_in": None,
    "q_grad": None, "k_grad": None, "v_grad": None,
    "q_grad_zeroed": None, "k_grad_zeroed": None, "v_grad_zeroed": None,
}

# q,k,v matrices
def make_forward_hook(name):
    def hook(module, inp, out):
        cache[f"{name}_out"] = out.detach()
        print(f"(forward pas) Calculated {name} shape {out.shape}")
    return hook

# gradients
def make_backward_hook(name):
    def hook(module, grad_in, grad_out):

        cache[f"{name}_grad_in"] = grad_in[0].detach().clone()
        
        # grad_out[0] is gradient of module output
        cache[f"{name}_grad"] = grad_out[0].detach().clone()

        # zero out gradient
        new_grad = grad_out[0] * 0
        cache[f"{name}_grad_zeroed"] = new_grad.detach().clone()

        # print(f"[BACKWARD BEFORE EDIT] {name} grad → {grad_out[0][0,0,:5]}")
        # print(f"[BACKWARD AFTER EDIT]  {name} grad → {new_grad[0,0,:5]}")

        return (new_grad,)  # pass modified gradients back
    return hook

# utils
# for name, module in model.named_modules():
#     print(name, module)

# for name, module in model.named_modules():
#     if "attention.self" in name:
#         print(name, module)


# attaching hook to attention layer 0
att = model.transformer.h[0].attn

handles = []

handles.append(att.q_proj.register_forward_hook(make_forward_hook("q")))
handles.append(att.k_proj.register_forward_hook(make_forward_hook("k")))
handles.append(att.v_proj.register_forward_hook(make_forward_hook("v")))

handles.append(att.q_proj.register_full_backward_hook(make_backward_hook("q")))
handles.append(att.k_proj.register_full_backward_hook(make_backward_hook("k")))
handles.append(att.v_proj.register_full_backward_hook(make_backward_hook("v")))

output = model(**inp) # forward pass
loss = output.logits.sum()

loss.backward() # backward pass

elapsed = time.time() - start
print(f"\nTime taken: {elapsed:.2f} seconds")

print("\nQ, K, V before back prop")
print("Q:", cache["q_out"][0,0,:5]) # batch, time/tokens, features [B,T,D]/check first 5 features
print("K:", cache["k_out"][0,0,:5])
print("V:", cache["v_out"][0,0,:5])

print("\ngrads input to the layer")
print("Q grad:", cache["q_grad_in"][0,0,:5])
print("K grad:", cache["k_grad_in"][0,0,:5])
print("V grad:", cache["v_grad_in"][0,0,:5])

print("\ngrads after backprop before zeroing")
print("Q grad:", cache["q_grad"][0,0,:5])
print("K grad:", cache["k_grad"][0,0,:5])
print("V grad:", cache["v_grad"][0,0,:5])

print("\ngrads after zeroing")
print("Q grad:", cache["q_grad_zeroed"][0,0,:5])
print("K grad:", cache["k_grad_zeroed"][0,0,:5])
print("V grad:", cache["v_grad_zeroed"][0,0,:5])

# rm hooks
for h in handles:
    h.remove()
