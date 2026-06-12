"""geofm.debug.forward_trace

Forward pass tracing utilities for debugging.
"""
from __future__ import annotations

from typing import List, Any, Optional

import torch
import torch.nn as nn


class ForwardTrace:
    """Trace forward passes through the model.

    Usage:
        # At start of forward
        ForwardTrace.trace_enter("Decoder")

        # At end of forward
        ForwardTrace.trace_exit("Decoder", output)
    """

    _indent_level = 0

    @staticmethod
    def trace_enter(name: str) -> None:
        """Mark entry to a module in the forward pass.

        Args:
            name: Name of the module/function
        """
        indent = "  " * ForwardTrace._indent_level
        print(f"{indent}→ {name}")
        ForwardTrace._indent_level += 1

    @staticmethod
    def trace_exit(name: str, output: Any = None) -> None:
        """Mark exit from a module in the forward pass.

        Args:
            name: Name of the module/function
            output: Optional output to display
        """
        ForwardTrace._indent_level -= 1
        indent = "  " * ForwardTrace._indent_level

        if output is None:
            print(f"{indent}← {name} [no output]")
        elif torch.is_tensor(output):
            print(f"{indent}← {name} {tuple(output.shape)}")
        elif isinstance(output, (list, tuple)):
            print(f"{indent}← {name} [{len(output)} items]")
        else:
            print(f"{indent}← {name}")

    @staticmethod
    def print_features(features: List[torch.Tensor]) -> None:
        """Print feature shapes.

        Args:
            features: List of feature tensors
        """
        print("\nFeature Trace")
        print("-" * 40)

        for i, feat in enumerate(features):
            if torch.is_tensor(feat):
                print(f"F{i}: {tuple(feat.shape)}")
            else:
                print(f"F{i}: {type(feat)}")

        print("-" * 40)

    @staticmethod
    def print_decoder_output(output: torch.Tensor) -> None:
        """Print decoder output shape.

        Args:
            output: Decoder output tensor
        """
        print("\nDecoder Output:")
        print("-" * 40)

        if torch.is_tensor(output):
            print(f"Shape: {tuple(output.shape)}")
            print(f"Dtype: {output.dtype}")
        else:
            print(f"Type: {type(output)}")

        print("-" * 40)

    @staticmethod
    def print_batch(batch: dict) -> None:
        """Print batch information.

        Args:
            batch: Batch dictionary
        """
        print("\nBatch Info:")
        print("-" * 40)

        for key, value in batch.items():
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    if torch.is_tensor(subval):
                        print(f"  {key}.{subkey}: {tuple(subval.shape)}")
            elif torch.is_tensor(value):
                print(f"  {key}: {tuple(value.shape)}")

        print("-" * 40)

    @staticmethod
    def trace_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Trace a tensor through the model.

        Args:
            name: Name for the tensor
            tensor: Tensor to trace

        Returns:
            The same tensor (for chaining)
        """
        print(f"  {name}: {tuple(tensor.shape)}")
        return tensor

    @staticmethod
    def reset() -> None:
        """Reset the trace indentation level."""
        ForwardTrace._indent_level = 0

    @staticmethod
    def checkpoint(name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Add a checkpoint in the forward pass.

        Args:
            name: Checkpoint name
            tensor: Tensor to check

        Returns:
            The same tensor
        """
        indent = "  " * ForwardTrace._indent_level
        print(f"{indent}✓ {name}: {tuple(tensor.shape)}")
        return tensor


class ModelTracer:
    """Trace model execution and layer activations."""

    def __init__(self, model: nn.Module):
        """Initialize tracer.

        Args:
            model: Model to trace
        """
        self.model = model
        self.hooks = []
        self.layer_outputs = {}

    def register_hooks(self, layer_names: Optional[List[str]] = None) -> None:
        """Register forward hooks to capture layer outputs.

        Args:
            layer_names: Optional list of layer names to hook
        """
        def hook_fn(name):
            def fn(module, input, output):
                self.layer_outputs[name] = output
            return fn

        for name, module in self.model.named_modules():
            if layer_names is None or name in layer_names:
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)

    def clear(self) -> None:
        """Clear captured outputs."""
        self.layer_outputs = {}

    def get_output(self, name: str) -> Any:
        """Get output from a specific layer.

        Args:
            name: Layer name

        Returns:
            Layer output
        """
        return self.layer_outputs.get(name)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __enter__(self):
        """Context manager entry."""
        self.register_hooks()
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.remove_hooks()


def trace_forward(
    model: nn.Module,
    batch: dict,
    trace_modules: Optional[List[str]] = None,
) -> dict:
    """Trace a forward pass through the model.

    Args:
        model: Model to trace
        batch: Input batch
        trace_modules: Optional list of module names to trace

    Returns:
        Dictionary of module outputs
    """
    outputs = {}

    def hook_fn(name):
        def fn(module, input, output):
            outputs[name] = output
        return fn

    hooks = []
    for name, module in model.named_modules():
        if trace_modules is None or name in trace_modules:
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)

    try:
        with torch.no_grad():
            model(batch)
    finally:
        for hook in hooks:
            hook.remove()

    return outputs