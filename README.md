# PyTorch Hooks Exercise

## Overview
This exercise explores PyTorch hooks in transformer attention modules. You'll inspect Q, K, V values and modify its gradients during backpropagation.

## Model
Use **GPT-2** or **GPT-J-6B**. Don't use toy model like gpt-nano.

## Tasks

### 1. Set Up Local Model Code
- Install transformers: `pip install transformers torch`
- Use the model class from relevant modeling file from your site-packages:
  - For GPT-2: `modeling_gpt2.py`
  - For GPT-J: `modeling_gptj.py`
- Create a custom model class with hooks set up, extending existing hugging face model class.

### 2. Add Hooks to Attention Linear Layers
Add forward and backward hooks to the Q, K, V projection layers in the attention module.

**Before backprop:**
- Print Q, K, V layers weight matrices
- Print Q, K, V layers weight gradient values

**After backprop:**
- Zero out Weight gradients
- Print updated Q, K, V layer weights after optimization

### 3. Run One Pass
Execute **one forward-backward pass** on a single batch and one params update by optimizer.
This should run on CPU without taking long.

## Deliverable
Working code that demonstrates:
- Forward hook capturing Q, K, V matrices
- Relevant hook capturing and modifying gradients
- Before/after comparison of Q, K, V weights
