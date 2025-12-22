"""Pydantic configuration schema for EMG classification pipelines.

Defines structured configuration models for datasets, windowing,
feature extraction, and training pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Literal, Union
from pydantic import BaseModel, Field
import yaml


class DatasetConfig(BaseModel):
    """Configuration for a dataset."""
    
    name: str = Field(..., description="Dataset identifier")
    n_channels: int = Field(..., ge=1, description="Number of EMG channels")
    sampling_rate: float = Field(..., gt=0, description="Sampling rate in Hz")
    data_dir: Path = Field(..., description="Path to data directory")
    file_pattern: str = Field(default="*.txt", description="Glob pattern for data files")
    loader_type: Literal["text", "mat", "csv"] = Field(
        default="text", description="Type of data loader to use"
    )
    label_column: Optional[str] = Field(
        default=None, description="Column name for labels (if applicable)"
    )
    emg_columns: Optional[List[int]] = Field(
        default=None, 
        description="Range of EMG columns to use [start, end]. None = all channels."
    )
    
    # Label mapping: gesture name prefix -> class ID
    label_mapping: Dict[str, int] = Field(
        default_factory=dict, description="Mapping from filename prefix to class ID"
    )
    
    # Subject configuration
    # Can be a list of subject IDs (db1) or a dict mapping folder names to IDs (rami)
    subjects: Optional[Union[List[str], Dict[str, str]]] = Field(
        default=None,
        description="Subject list or mapping. List format: ['s01', 's02', ...]. "
                    "Dict format: {'folder_name': 'subject_id', ...}"
    )


class WindowConfig(BaseModel):
    """Configuration for signal windowing."""
    
    size: int = Field(default=400, gt=0, description="Window size in samples")
    increment: int = Field(default=100, gt=0, description="Window increment (stride) in samples")
    
    @property
    def overlap(self) -> int:
        """Calculate overlap between windows."""
        return self.size - self.increment


class FeatureConfig(BaseModel):
    """Configuration for feature extraction."""
    
    extractor_type: Literal["spectral", "time", "combined"] = Field(
        default="combined", description="Type of features to extract"
    )
    
    # Spectral feature options
    spectral_moments: bool = Field(default=True, description="Include spectral moments M0-M4")
    spectral_flux: bool = Field(default=True, description="Include spectral flux")
    spectral_sparsity: bool = Field(default=True, description="Include spectral sparsity")
    irregularity_factor: bool = Field(default=True, description="Include irregularity factor")
    spectrum_correlation: bool = Field(default=True, description="Include spectrum correlation")
    
    # Time domain feature options
    time_mav: bool = Field(default=True, description="Include Mean Absolute Value")
    time_wl: bool = Field(default=True, description="Include Waveform Length")
    time_zc: bool = Field(default=True, description="Include Zero Crossings")
    time_ssc: bool = Field(default=True, description="Include Slope Sign Changes")
    time_threshold: float = Field(default=0.01, description="Threshold for ZC and SSC")


class ModelConfig(BaseModel):
    """Configuration for model training."""
    
    name: str = Field(..., description="Model name (e.g., 'lda', 'svm', 'rf')")
    hyperparameters: Dict = Field(default_factory=dict, description="Model hyperparameters")


class NormalizerConfig(BaseModel):
    """Configuration for data normalization."""
    
    type: Literal["standard", "percentile", "channel", "adaptive"] = Field(
        default="standard", description="Normalization strategy"
    )
    # Channel normalizer options
    n_channels: Optional[int] = Field(default=None, description="Number of channels for channel normalizer")
    features_per_channel: Optional[int] = Field(default=None, description="Features per channel")
    # Percentile normalizer options
    lower_percentile: float = Field(default=5, description="Lower percentile for clipping")
    upper_percentile: float = Field(default=95, description="Upper percentile for clipping")
    # Adaptive scaler options
    adaptation_strength: float = Field(default=0.5, description="Strength of test-time adaptation")


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""
    
    cv_type: Literal["loso", "within_subject", "none"] = Field(
        default="loso", description="Cross-validation strategy"
    )
    n_folds: int = Field(default=5, description="Number of folds for within-subject CV")
    calibration_samples: int = Field(
        default=0, description="Number of samples per class for calibration"
    )
    
    # Feature selection
    feature_selection: Literal["none", "kbest", "pca"] = Field(
        default="none", description="Feature selection method"
    )
    k_best: Optional[int] = Field(default=None, description="Number of features for SelectKBest")
    pca_components: Optional[int] = Field(default=None, description="Number of PCA components")


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""
    
    dataset: DatasetConfig
    window: WindowConfig = Field(default_factory=WindowConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    normalizer: NormalizerConfig = Field(default_factory=NormalizerConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    models: List[ModelConfig] = Field(default_factory=list, description="Models to train")
    
    # Output configuration
    results_dir: Path = Field(default=Path("results"), description="Directory for results")
    save_models: bool = Field(default=False, description="Whether to save trained models")


def load_config(path: Union[Path, str]) -> PipelineConfig:
    """Load a pipeline configuration from a YAML file.
    
    Args:
        path: Path to the YAML configuration file.
        
    Returns:
        Validated PipelineConfig instance.
    """
    path = Path(path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return PipelineConfig(**data)


def save_config(config: PipelineConfig, path: Union[Path, str]) -> None:
    """Save a pipeline configuration to a YAML file.
    
    Args:
        config: PipelineConfig instance to save.
        path: Path to save the YAML file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
