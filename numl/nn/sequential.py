"""
numl/nn/sequential.py
---------------------
Sequential container: chains modules one after another.
"""

from numl.core.tensor import Tensor
from numl.nn.module import Module


class Sequential(Module):
    """
    A sequential container that passes input through each module in order.

    Example
    -------
    model = Sequential([
        Dense(784, 256),
        ReLU(),
        BatchNorm1d(256),
        Dropout(0.3),
        Dense(256, 10),
    ])
    """

    def __init__(self, layers: list | None = None):
        super().__init__()
        self._layer_list = []
        if layers:
            for i, layer in enumerate(layers):
                self.add(layer, name=str(i))

    def add(self, module: Module, name: str | None = None) -> "Sequential":
        """Append a module to the sequence."""
        if name is None:
            name = str(len(self._layer_list))
        self._layer_list.append(module)
        self._modules[name] = module
        object.__setattr__(self, name, module)
        return self

    def forward(self, x: Tensor) -> Tensor:
        for module in self._layer_list:
            x = module(x)
        return x

    def __getitem__(self, idx):
        return self._layer_list[idx]

    def __len__(self):
        return len(self._layer_list)

    def __iter__(self):
        return iter(self._layer_list)

    def get_config(self):
        return {"layers": [m.get_config() for m in self._layer_list]}

    def __repr__(self) -> str:
        lines = ["Sequential("]
        for i, m in enumerate(self._layer_list):
            lines.append(f"  ({i}): {repr(m)}")
        lines.append(")")
        return "\n".join(lines)
