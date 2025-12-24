"""Pydantic configuration schema for EMG classification pipelines.

Defines structured configuration models for datasets, windowing,
feature extraction, and training pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Literal, Union
from pydantic import BaseModel, Field
import yaml

# Path to the features.yaml configuration file
_FEATURES_CONFIG_PATH = Path(__file__).parent / "features.yaml"


class FeatureSetConfig(BaseModel):
    """Configuration for a named feature set.
    
    Defines which features to extract from EMG signals.
    Loaded from features.yaml by name (e.g., 'default', 'full', 'experimental').
    """
    
    # Spectral features
    spectral_moments: bool = Field(default=True, description="Include spectral moments M0-M4")
    spectral_flux: bool = Field(default=True, description="Include spectral flux")
    spectral_sparsity: bool = Field(default=True, description="Include spectral sparsity")
    spectral_irregularity: bool = Field(default=True, description="Include irregularity factor")
    spectral_correlation: bool = Field(default=True, description="Include spectrum correlation")
    
    # Time-domain features
    time_mav: bool = Field(default=True, description="Include Mean Absolute Value")
    time_wl: bool = Field(default=True, description="Include Waveform Length")
    time_zc: bool = Field(default=True, description="Include Zero Crossings")
    time_ssc: bool = Field(default=True, description="Include Slope Sign Changes")
    time_rms: bool = Field(default=False, description="Include Root Mean Square")
    time_iemg: bool = Field(default=False, description="Include Integrated EMG")
    time_var: bool = Field(default=False, description="Include Variance")
    time_lscale: bool = Field(default=False, description="Include L-Scale (robust dispersion)")
    
    # Histogram features
    hist_enabled: bool = Field(default=False, description="Include histogram features")
    hist_bins: int = Field(default=10, ge=2, description="Number of histogram bins")
    
    # Wavelet features (mDWT)
    wavelet_enabled: bool = Field(default=False, description="Include wavelet features")
    wavelet_name: str = Field(default="db7", description="Wavelet family name")
    wavelet_level: int = Field(default=3, ge=1, description="Wavelet decomposition levels")
    
    # Threshold for ZC/SSC
    threshold: float = Field(default=0.01, description="Threshold for ZC and SSC")


class PreprocessingConfig(BaseModel):
    """Configuration for signal preprocessing (filtering).
    
    Applied to raw EMG signals before windowing and feature extraction.
    """
    
    # Bandpass filter
    bandpass_enabled: bool = Field(default=True, description="Enable bandpass filter")
    bandpass_lowcut: float = Field(default=20.0, gt=0, description="Low cutoff frequency (Hz)")
    bandpass_highcut: float = Field(default=500.0, gt=0, description="High cutoff frequency (Hz)")
    bandpass_order: int = Field(default=4, ge=1, description="Butterworth filter order")
    
    # Notch filter (powerline interference)
    notch_enabled: bool = Field(default=True, description="Enable notch filter")
    notch_freq: float = Field(default=50.0, gt=0, description="Notch frequency (Hz)")
    notch_q: float = Field(default=30.0, gt=0, description="Quality factor")


def load_feature_set(name: str, config_path: Optional[Path] = None) -> FeatureSetConfig:
    """Load a named feature set from features.yaml.
    
    Args:
        name: Name of the feature set (e.g., 'default', 'full', 'experimental').
        config_path: Optional path to features.yaml. Uses default if not specified.
        
    Returns:
        Validated FeatureSetConfig instance.
        
    Raises:
        ValueError: If the feature set name is not found.
        FileNotFoundError: If the config file doesn't exist.
    """
    path = config_path or _FEATURES_CONFIG_PATH
    
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    
    feature_sets = data.get("feature_sets", {})
    if name not in feature_sets:
        available = list(feature_sets.keys())
        raise ValueError(f"Feature set '{name}' not found. Available: {available}")
    
    return FeatureSetConfig(**feature_sets[name])


def load_preprocessing_config(config_path: Optional[Path] = None) -> PreprocessingConfig:
    """Load preprocessing configuration from features.yaml.
    
    Args:
        config_path: Optional path to features.yaml. Uses default if not specified.
        
    Returns:
        Validated PreprocessingConfig instance.
    """
    path = config_path or _FEATURES_CONFIG_PATH
    
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    
    preproc = data.get("preprocessing", {})
    
    # Flatten nested structure from YAML
    config_dict = {}
    if "bandpass" in preproc:
        bp = preproc["bandpass"]
        config_dict["bandpass_enabled"] = bp.get("enabled", True)
        config_dict["bandpass_lowcut"] = bp.get("lowcut", 20.0)
        config_dict["bandpass_highcut"] = bp.get("highcut", 500.0)
        config_dict["bandpass_order"] = bp.get("order", 4)
    if "notch" in preproc:
        notch = preproc["notch"]
        config_dict["notch_enabled"] = notch.get("enabled", True)
        config_dict["notch_freq"] = notch.get("freq", 50.0)
        config_dict["notch_q"] = notch.get("q_factor", 30.0)
    
    return PreprocessingConfig(**config_dict)


def list_feature_sets(config_path: Optional[Path] = None) -> List[str]:
    """List available feature set names.
    
    Args:
        config_path: Optional path to features.yaml. Uses default if not specified.
        
    Returns:
        List of available feature set names.
    """
    path = config_path or _FEATURES_CONFIG_PATH
    
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    
    return list(data.get("feature_sets", {}).keys())


class DatasetConfig(BaseModel):
    """Configuration for a dataset."""
    
    name: str = Field(..., description="Dataset identifier")
    n_channels: int = Field(..., ge=1, description="Number of EMG channels")
    sampling_rate: float = Field(..., gt=0, description="Sampling rate in Hz")
    data_dir: Path = Field(..., description="Path to data directory")
    file_pattern: str = Field(default="*.txt", description="Glob pattern for data files")
    loader_type: str = Field(
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
