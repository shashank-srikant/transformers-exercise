from transformers.models.gptj.modeling_gptj import (
    GPTJAttention,
    GPTJForCausalLM,
    GPTJBlock,
    GPTJModel,
    GPTJ_ATTENTION_CLASSES
)
import torch.nn as nn
import torch

class GPTJAttentionWithHooks(GPTJAttention):
    """Extended GPTJAttention with custom hooks for gradient analysis."""
    
    def __init__(self, config, layer_idx=None, enable_hooks=False):
        super().__init__(config, layer_idx)
        
        self.hook_handles = []
        self.hooks_enabled = enable_hooks
        
        # Minimal storage: just 5 values per snapshot
        self.weight_snapshots = {
            'Q': {'before_backprop': None, 'after_optimizer': None},
            'K': {'before_backprop': None, 'after_optimizer': None},
            'V': {'before_backprop': None, 'after_optimizer': None}
        }
        self.grad_snapshots = {
            'Q': {'before_zero': None, 'after_zero': None},
            'K': {'before_zero': None, 'after_zero': None},
            'V': {'before_zero': None, 'after_zero': None}
        }
        
        if self.hooks_enabled:
            self._setup_hooks()
    
    def _setup_hooks(self):
        """Setup forward and weight gradient hooks."""
        self._register_forward_hooks()
        self._register_weight_hooks()
    
    def _register_forward_hooks(self):
        """Capture weights BEFORE backprop."""
        def module_forward_hook(name):
            def hook(module, input, output):
                # STAGE 1: Capture 5 weight values BEFORE backprop
                if name == 'Q':
                    self.weight_snapshots['Q']['before_backprop'] = self.q_proj.weight.data[0, :5].clone()
                elif name == 'K':
                    self.weight_snapshots['K']['before_backprop'] = self.k_proj.weight.data[0, :5].clone()
                elif name == 'V':
                    self.weight_snapshots['V']['before_backprop'] = self.v_proj.weight.data[0, :5].clone()
            return hook
        
        self.hook_handles.append(
            self.q_proj.register_forward_hook(module_forward_hook("Q"))
        )
        self.hook_handles.append(
            self.k_proj.register_forward_hook(module_forward_hook("K"))
        )
        self.hook_handles.append(
            self.v_proj.register_forward_hook(module_forward_hook("V"))
        )
    
    def _register_weight_hooks(self):
        """Register hooks on weight tensors to capture and zero gradients."""
        def weight_grad_hook(name):
            def hook(grad):
                # STAGE 2: Capture 5 gradient values BEFORE zeroing
                self.grad_snapshots[name]['before_zero'] = grad[0, :5].clone()
                
                # Zero the gradient
                new_grad = grad * 0
                
                # STAGE 3: Capture 5 gradient values AFTER zeroing
                self.grad_snapshots[name]['after_zero'] = new_grad[0, :5].clone()
                
                return new_grad
            return hook
        
        self.hook_handles.append(
            self.q_proj.weight.register_hook(weight_grad_hook("Q"))
        )
        self.hook_handles.append(
            self.k_proj.weight.register_hook(weight_grad_hook("K"))
        )
        self.hook_handles.append(
            self.v_proj.weight.register_hook(weight_grad_hook("V"))
        )
    
    def capture_weights_after_optimizer(self):
        """STAGE 4: Capture weights AFTER optimizer update."""
        if not self.hooks_enabled:
            return
            
        self.weight_snapshots['Q']['after_optimizer'] = self.q_proj.weight.data[0, :5].clone()
        self.weight_snapshots['K']['after_optimizer'] = self.k_proj.weight.data[0, :5].clone()
        self.weight_snapshots['V']['after_optimizer'] = self.v_proj.weight.data[0, :5].clone()
    
    def print_summary(self):
        """Print summary matching the desired output format."""
        if not self.hooks_enabled:
            print("Hooks not enabled for this layer")
            return
            
        print("WEIGHT AND GRADIENT TRACKING:")
        
        # Stage 1: Q,K,V weights before backprop
        print("\nQ, K, V before back prop")
        for name in ['Q', 'K', 'V']:
            if self.weight_snapshots[name]['before_backprop'] is not None:
                print(f"{name}: {self.weight_snapshots[name]['before_backprop']}")
        
        # Stage 2: Gradients after backprop before zeroing
        print("\ngrads after backprop before zeroing")
        for name in ['Q', 'K', 'V']:
            if self.grad_snapshots[name]['before_zero'] is not None:
                print(f"{name} grad: {self.grad_snapshots[name]['before_zero']}")
        
        
        # Stage 3: Gradients after zeroing
        print("\nQ,K,V after zeroing")
        for name in ['Q', 'K', 'V']:
            if self.grad_snapshots[name]['after_zero'] is not None:
                print(f"{name}: {self.grad_snapshots[name]['after_zero']}")
        
        # Stage 4: Weights after optimizer step
        print("\nQ,K,V weights after optimizer")
        for name in ['Q', 'K', 'V']:
            if self.weight_snapshots[name]['after_optimizer'] is not None:
                print(f"{name}: {self.weight_snapshots[name]['after_optimizer']}")
            else:
                print(f"{name}: Need to call capture_weights_after_optimizer()")
        
        # Show weight changes
        print("\nWeight changes (after_optimizer - before_backprop)")
        for name in ['Q', 'K', 'V']:
            if (self.weight_snapshots[name]['after_optimizer'] is not None and 
                self.weight_snapshots[name]['before_backprop'] is not None):
                change = (self.weight_snapshots[name]['after_optimizer'] - 
                         self.weight_snapshots[name]['before_backprop'])
                print(f"{name}: {change}")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()


class GPTJBlockWithHooks(GPTJBlock):
    """GPTJBlock that uses our custom attention class."""
    
    def __init__(self, config, layer_idx=None, enable_hooks=False):
        super().__init__(config, layer_idx)
        self.attn = GPTJAttentionWithHooks(config, layer_idx, enable_hooks=enable_hooks)


class GPTJModelWithHooks(GPTJModel):
    """GPTJModel that uses blocks with custom attention."""
    
    def __init__(self, config, track_layer_idx=0):
        super().__init__(config)
        self.track_layer_idx = track_layer_idx
        
        self.h = nn.ModuleList([
            GPTJBlockWithHooks(config, layer_idx=i, enable_hooks=(i == track_layer_idx)) 
            for i in range(config.n_layer)
        ])
        self.post_init()


class GPTJForCausalLMWithHooks(GPTJForCausalLM):
    """GPTJForCausalLM with hook support."""
    
    def __init__(self, config, track_layer_idx=0):
        super().__init__(config)
        self.transformer = GPTJModelWithHooks(config, track_layer_idx=track_layer_idx)
        self.track_layer_idx = track_layer_idx
        self.post_init()
    
    def capture_weights_after_optimizer(self):
        """Capture weights after optimizer update for the tracked layer."""
        self.transformer.h[self.track_layer_idx].attn.capture_weights_after_optimizer()
    
    def print_summary(self):
        """Print summary for the tracked layer."""
        print(f"\n>>> Summary for Layer {self.track_layer_idx} <<<")
        self.transformer.h[self.track_layer_idx].attn.print_summary()
    
    def remove_all_hooks(self):
        """Remove hooks from the tracked layer."""
        if hasattr(self.transformer.h[self.track_layer_idx].attn, 'remove_hooks'):
            self.transformer.h[self.track_layer_idx].attn.remove_hooks()
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, track_layer_idx=0, *args, **kwargs):
        """Override from_pretrained to accept track_layer_idx parameter."""

        # Loads pretrained weights directly into custom model
        # without creating a second copy in memory

        from transformers import AutoConfig, GPTJForCausalLM
        
        # Ensure low_cpu_mem_usage is enabled (avoid duplicates)
        if 'low_cpu_mem_usage' not in kwargs:
            kwargs['low_cpu_mem_usage'] = True
        
        print("Loading pretrained model...")
        pretrained_model = GPTJForCausalLM.from_pretrained(
            pretrained_model_name_or_path, 
            *args, 
            **kwargs
        )
        
        # Get config from loaded model
        config = pretrained_model.config
        
        # Create custom model structure without initializing weights
        print("Creating custom model structure...")
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        
        # Copy ALL attributes from pretrained model
        model.config = config
        model.track_layer_idx = track_layer_idx
        model.transformer = pretrained_model.transformer # Note: copy references only for saving memory
        model.lm_head = pretrained_model.lm_head
        
        # Copy rem attributes
        if hasattr(pretrained_model, 'generation_config'):
            model.generation_config = pretrained_model.generation_config
        if hasattr(pretrained_model, 'model_parallel'):
            model.model_parallel = pretrained_model.model_parallel
        if hasattr(pretrained_model, 'is_parallelizable'):
            model.is_parallelizable = pretrained_model.is_parallelizable
        
        # Replace only the tracked layer with our custom version
        print(f"Replacing layer {track_layer_idx} with hooked version...")
        old_block = model.transformer.h[track_layer_idx]
        
        # Create new block with hooks
        new_block = GPTJBlockWithHooks(config, layer_idx=track_layer_idx, enable_hooks=True) # Note: we are only creating new memory for 1 layer 

        new_block.load_state_dict(old_block.state_dict())
        
        model.transformer.h[track_layer_idx] = new_block
        
        del pretrained_model        
        print("Model ready!")
        return model