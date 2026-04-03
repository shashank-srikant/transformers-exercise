## Backprop Gradient Flow
activation grad->weight grad->input grad

## Issue
Initial assumption was that making the activation grad zero, weight grad should be zero (given weight grad is computed using activation grad).
But, still weight grad was non-zero. This was because of order in which these grads are computed.

     
## Order of Computation while using register_full_backward_hook
  1. Weight Gradient Computation: ∂L/∂W = (∂L/∂Q)^T @ X -> still non-zero
  2. Input Gradient Computation: ∂L/∂W = (∂L/∂Q)^T @ X    
  3. Activation Gradient Computation: ∂L/∂Q, register_full_backward_hook updates the activation gradient to zero ∂L/∂Q=0
  
## Order of Computation while using register_full_backward_pre_hook
  1. Activation Gradient Computation: ∂L/∂Q, register_full_backward_pre_hook updates the activation gradient to zero ∂L/∂Q=0
  2. Weight Gradient Computation: ∂L/∂W = (∂L/∂Q)^T @ X -> gets to be zero
  3. Input Gradient Computation: ∂L/∂W = (∂L/∂Q)^T @ X  
  
  When used register_full_backward_pre_hook, it set the weight grad zero.

## Notes:
## Module vs Tensor Wise Hooks
Modules based hooks require additional code to cleanup these hook handles.  
Tensor based hook `{tensor}.register_hook`:  query.register_hook(), self.q_proj.weight.register_hook()
  
## forward vs pre_forward
  forward_hook
  runs after the forward pass
  
  pre_forward_hook
  before the forward pass

##  Think through what setting grad to zero does
  `Weight grad=0 and Activation grad!=0`: weight update will stop, activation flow is still there/activation learning still continues.  
`Weight grad=!0 and Activation grad=0`: weight update still contine, will eventually be zero


## Choice of hooks for tracking weight gradients and updated weights after optimization
### Why weight hook for tracking weight gradients?  
Backward hooks (`register_full_backward_pre_hook`) intercept gradients flowing through the module's output (i.e., activation gradients), not the gradients of the module's parameters (weight gradients).

### Why manual capture of updated Q,K,V after optimizer step?  	
Hooks only fire during forward/backward passes, not during optimizer updates. The `optimizer.step()` happens *outside the computational graph* - it's a direct tensor manipulation that doesn't trigger any hooks.