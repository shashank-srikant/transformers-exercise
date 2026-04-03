# Exercise 0 - Pytorch, Loss functions fundamentals

**1. Model choice:**  
M1: Create a simple feed forward network containing two intermediate layers  
M2: use a bert-style encoder (and find out the difference between bert style encoder vs. gpt style decoder)  

**2. Data:** 200 total sentences with ground-truth emotion (positive/negative/neutral)  
Data split:  
100: train  
50: val  
50: test  

The input will be of multiple sentences: W x N x B  
W: vocab length  
N: tokens per batch  
B: batch size  

**3. Write a loss function which calculates batch-wise loss and optimizes the network.**  
a. Tell me what the choice of loss function is.  
b. Tell me how you will combine per sentence loss into a batch loss.  
Run this until convergence on val-set, and report train, val, test-set results.  

# Exercise 1 - PyTorch Hooks Exercise

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
