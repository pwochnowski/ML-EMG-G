"""Data loaders for various EMG datasets.

Dataset-specific loaders encapsulate all format quirks and defaults.
Use get_loader() to instantiate the correct loader by name.
"""

from typing import Type

from ..base import DataLoader
from .db1_loader import DB1Loader
from .rami_loader import RamiLoader
from .myo_loader import MyoLoader

# Legacy imports for backwards compatibility
from .text_loader import TextFileLoader
from .mat_loader import MatFileLoader


__all__ = [
    # Dataset-specific loaders (preferred)
    "DB1Loader",
    "RamiLoader",
    "MyoLoader",
    # Legacy format-based loaders (deprecated)
    "TextFileLoader",
    "MatFileLoader",
    # Factory function
    "get_loader",
]


# Mapping from loader type names to loader classes
_LOADER_REGISTRY: dict[str, Type[DataLoader]] = {
    # Dataset-specific loaders
    "db1": DB1Loader,
    "rami": RamiLoader,
    "myo": MyoLoader,
}


def get_loader(name: str) -> Type[DataLoader]:
    """Get a loader class by name.
    
    Args:
        name: Loader type name. Supported values:
            - "db1": NinaPro Database 1 loader
            - "rami": Rami dataset loader
            - "myo": GrabMyo dataset loader
            - "mat": Legacy MAT file loader (deprecated)
            - "text": Legacy text file loader (deprecated)
    
    Returns:
        The loader class (not an instance).
    
    Raises:
        ValueError: If the loader name is not recognized.
    
    Example:
        >>> LoaderClass = get_loader("db1")
        >>> loader = LoaderClass(data_dir="datasets/db1/data")
    """
    name = name.lower()
    if name not in _LOADER_REGISTRY:
        available = ", ".join(sorted(_LOADER_REGISTRY.keys()))
        raise ValueError(f"Unknown loader type '{name}'. Available: {available}")
    return _LOADER_REGISTRY[name]
