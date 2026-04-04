"""
numl/nn/module.py
-----------------
Base Module class — the building block for all neural network components.

Design mirrors PyTorch's Module:
  - Parameters are registered automatically via __setattr__ interception.
  - Submodules are registered the same way.
  - .parameters() recursively collects all leaf parameters.
  - .state_dict() / .load_state_dict() for serialization.
  - .train() / .eval() toggles training mode (affects BatchNorm, Dropout).
"""

import numpy as np
from numl.core.tensor import Tensor


class Module:
    """Abstract base class for all neural network modules."""

    def __init__(self):
        # Use object.__setattr__ to bypass our override during __init__
        object.__setattr__(self, '_parameters', {})
        object.__setattr__(self, '_modules', {})
        object.__setattr__(self, '_training', True)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.forward() is not implemented.")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # ------------------------------------------------------------------
    # Attribute registration
    # ------------------------------------------------------------------

    def __setattr__(self, name: str, value):
        # Remove from previous registries if re-assigning
        params = object.__getattribute__(self, '_parameters')
        modules = object.__getattribute__(self, '_modules')

        if name in params:
            del params[name]
        if name in modules:
            del modules[name]

        if isinstance(value, Tensor) and value.requires_grad:
            params[name] = value
            object.__setattr__(self, name, value)
        elif isinstance(value, Module):
            modules[name] = value
            object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def parameters(self) -> list:
        """Recursively collect all parameters (Tensors with requires_grad=True)."""
        params = []
        seen = set()

        def _collect(module: "Module"):
            for p in module._parameters.values():
                if id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
            for m in module._modules.values():
                _collect(m)

        _collect(self)
        return params

    def named_parameters(self, prefix: str = "") -> list:
        """Recursively collect (name, parameter) pairs."""
        result = []
        seen = set()

        def _collect(module: "Module", pfx: str):
            for name, p in module._parameters.items():
                full_name = f"{pfx}.{name}" if pfx else name
                if id(p) not in seen:
                    seen.add(id(p))
                    result.append((full_name, p))
            for mod_name, m in module._modules.items():
                full_pfx = f"{pfx}.{mod_name}" if pfx else mod_name
                _collect(m, full_pfx)

        _collect(self, prefix)
        return result

    def zero_grad(self):
        """Zero out gradients on all parameters."""
        for p in self.parameters():
            p.zero_grad()

    # ------------------------------------------------------------------
    # Training mode
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "Module":
        """Set training mode for this module and all submodules."""
        object.__setattr__(self, '_training', mode)
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        """Set evaluation mode (equivalent to train(False))."""
        return self.train(False)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self, prefix: str = "") -> dict:
        """Return {name: np.ndarray} for all parameters."""
        result = {}
        for name, p in self.named_parameters(prefix=prefix):
            result[name] = p.data.copy()
        return result

    def load_state_dict(self, state: dict, prefix: str = ""):
        """Restore parameters from a state dict."""
        named = dict(self.named_parameters(prefix=prefix))
        for name, arr in state.items():
            if name in named:
                named[name].data = arr
            else:
                raise KeyError(f"Unexpected key '{name}' in state_dict.")

    def get_config(self) -> dict:
        """
        Return constructor kwargs for serialization.
        Subclasses should override this to return their init arguments.
        """
        return {}

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}("]
        for name, m in self._modules.items():
            mod_str = repr(m).replace("\n", "\n  ")
            lines.append(f"  ({name}): {mod_str}")
        lines.append(")")
        return "\n".join(lines) if len(lines) > 2 else f"{type(self).__name__}()"
