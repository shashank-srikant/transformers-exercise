"""
Find two numbers A and B such that A + B = 10
"""

import torch

A = torch.tensor(0, dtype=torch.float32, requires_grad=True)
B = torch.tensor(0, dtype=torch.float32, requires_grad=True)
target = 10


lr = 0.01

for epoch in range(101):
    prediction = A + B
    loss = (prediction - target) ** 2

    loss.backward()

    with torch.no_grad():
        A -= lr * A.grad
        B -= lr * B.grad

    A.grad.zero_()
    B.grad.zero_()

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch}: A={A.item():.4f}, B={B.item():.4f}, Loss={loss.item():.4f}"
        )


"""

Q1: Why lr is set to 0.01? What happens with a higher or lower learning rate?
- It allows for steady convergence towards the target.
- Larger lr may cause overshooting and divergence.
- Smaller lr may lead to slow convergence or get stuck in a local minima.

Q2: Why do we need to zero the gradients after each update?
- By default, pytorch accumulates gradients. So if we don't zero them, the gradients from previous iterations add up.

Q3: What is the purpose of loss.backward()?
- It computes the gradients of the loss with respect to the parameters (A and B in this case) using backpropagation.

Q4: Why should the computation graph be acyclic? How does computation graph work for recurrent neural networks?
- An acyclic graph ensures that there are no loops, which allows for proper gradient computation.

Chain Rule:
Say we have function:
- y = A + B
- loss = (y - target)^2 = ((A + B) - target)^2 = (A+B-10)^2 = u^2, where u = A + B - 10
- target = 10

Then, the gradients are:
d(loss)/dA
    = d(loss)/du * du/dA
    = d(u^2) / du * du/dA
    = 2u * 1
    = 2(A + B - 10)

Similarly, calculate d(loss)/dB, which comes equal to 2(A + B - 10) as well.

- This calculation is done by autograd in PyTorch when we call loss.backward()
- What values is A or B updated with is determined by the learning rate and the calculated gradients.
- For eg: A is updated as: A = A - lr * d(loss)/dA
    - If gradient is negative, increasing A will decrease the loss.
    - If gradient is positive, decreasing A will decrease the loss.
"""
