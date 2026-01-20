from .environment import env_creator, make

try:
    import torch as _torch  # noqa: F401
except ImportError:
    pass
else:
    import importlib
    torch = importlib.import_module(__name__ + ".torch")
