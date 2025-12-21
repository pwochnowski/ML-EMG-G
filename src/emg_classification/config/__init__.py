"""Configuration module for EMG classification."""

from .schema import (
    DatasetConfig,
    WindowConfig,
    FeatureConfig,
    ModelConfig,
    PipelineConfig,
    load_config,
)

__all__ = [
    "DatasetConfig",
    "WindowConfig",
    "FeatureConfig",
    "ModelConfig",
    "PipelineConfig",
    "load_config",
]
