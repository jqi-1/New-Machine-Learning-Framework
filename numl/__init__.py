"""
numl — A full machine learning framework built on pure Python and NumPy.
"""
from numl.core.tensor import Tensor, tensor, zeros, ones, randn, concatenate
from numl import core, nn, optim, classical, data, preprocessing, decomposition, metrics, serialize

__version__ = "0.1.0"
__all__ = ["Tensor", "tensor", "zeros", "ones", "randn", "concatenate",
           "core", "nn", "optim", "classical", "data",
           "preprocessing", "decomposition", "metrics", "serialize"]
