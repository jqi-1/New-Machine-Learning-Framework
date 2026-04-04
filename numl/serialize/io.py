"""
numl/serialize/io.py
---------------------
Model serialization and deserialization.

Two strategies:
  1. Neural networks (Module subclasses):
       - Weights saved as .npz (compressed NumPy arrays)
       - Architecture config saved as .json
       - load_model() reconstructs from both

  2. Classical models (any Python object):
       - Uses pickle (protocol 4)
       - save_classical() / load_classical()
"""

import json
import pickle
import importlib
import numpy as np
from numl.nn.module import Module


def save_model(model: Module, path: str) -> None:
    """
    Save a neural network Module to disk.

    Writes two files:
      {path}.npz  — compressed weight arrays
      {path}.json — class name, module path, and get_config() output

    Parameters
    ----------
    model : Module subclass
    path  : str — base path (no extension)
    """
    # Save weights
    state = model.state_dict()
    np.savez_compressed(path + '.npz', **state)

    # Save metadata
    meta = {
        'class': type(model).__name__,
        'module': type(model).__module__,
        'config': model.get_config(),
    }
    with open(path + '.json', 'w') as f:
        json.dump(meta, f, indent=2)


def load_model(path: str, model_class=None, **kwargs) -> Module:
    """
    Load a neural network Module from disk.

    Parameters
    ----------
    path        : str — base path (no extension)
    model_class : Module subclass (optional). If None, reconstructed from .json.
    **kwargs    : override constructor arguments

    Returns
    -------
    model : restored Module
    """
    with open(path + '.json') as f:
        meta = json.load(f)

    config = meta.get('config', {})
    config.update(kwargs)

    if model_class is None:
        mod = importlib.import_module(meta['module'])
        cls = getattr(mod, meta['class'])
    else:
        cls = model_class

    model = cls(**config)

    arrays = np.load(path + '.npz')
    state = dict(arrays)
    model.load_state_dict(state)

    return model


def save_classical(model, path: str) -> None:
    """
    Save a classical ML model (non-Module) using pickle.

    Parameters
    ----------
    model : any object (DecisionTree, RandomForest, SVC, etc.)
    path  : str — base path; writes {path}.pkl
    """
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=4)


def load_classical(path: str):
    """
    Load a classical ML model from a pickle file.

    Parameters
    ----------
    path : str — base path; reads {path}.pkl

    Returns
    -------
    model : restored object
    """
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_checkpoint(model: Module, optimizer, epoch: int, path: str,
                    extra: dict | None = None) -> None:
    """
    Save a full training checkpoint (model + optimizer state + epoch).

    Parameters
    ----------
    model     : Module
    optimizer : Optimizer
    epoch     : int
    path      : str — base path; writes {path}_checkpoint.npz + .json
    extra     : optional dict of additional metadata (e.g., best_loss)
    """
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    if extra:
        state['extra'] = extra

    meta = {
        'class': type(model).__name__,
        'module': type(model).__module__,
        'config': model.get_config(),
        'epoch': epoch,
        'extra': extra or {},
    }

    # Flatten numpy arrays for npz
    flat = {}
    for k, v in state['model_state'].items():
        flat[f'model__{k}'] = v

    # Optimizer state can contain nested dicts/arrays
    opt_state = optimizer.state_dict()
    flat['__optimizer_step'] = np.array([opt_state.get('step_count', 0)])
    for key, val in opt_state.get('state', {}).items():
        if isinstance(val, dict):
            for sub_k, sub_v in val.items():
                if isinstance(sub_v, np.ndarray):
                    flat[f'optim__{key}__{sub_k}'] = sub_v
        elif isinstance(val, np.ndarray):
            flat[f'optim__{key}'] = val

    np.savez_compressed(path + '_checkpoint.npz', **flat)
    with open(path + '_checkpoint.json', 'w') as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(path: str, model: Module, optimizer=None) -> dict:
    """
    Load a training checkpoint into an existing model (and optimizer).

    Parameters
    ----------
    path      : str — base path (no suffix)
    model     : Module — will have state loaded in-place
    optimizer : Optimizer (optional) — will have state partially restored

    Returns
    -------
    meta : dict with 'epoch', 'extra', etc.
    """
    with open(path + '_checkpoint.json') as f:
        meta = json.load(f)

    arrays = dict(np.load(path + '_checkpoint.npz'))

    # Restore model
    model_state = {k[len('model__'):]: v
                   for k, v in arrays.items() if k.startswith('model__')}
    model.load_state_dict(model_state)

    # Restore optimizer step count
    if optimizer is not None and '__optimizer_step' in arrays:
        optimizer._step_count = int(arrays['__optimizer_step'][0])

    return meta
