from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class DataConfig(BaseModel):
    """Data configuration validation."""
    
    root: Path = Field(description="Root directory for dataset")
    batch_size: int = Field(ge=1, le=512, description="Batch size for training")
    num_workers: int = Field(ge=0, le=16, description="Number of data loader workers")
    val_fraction: float = Field(ge=0.0, le=1.0, description="Validation split fraction")
    use_webdataset: bool = Field(default=False, description="Use WebDataset format")
    
    @field_validator("root")
    @classmethod
    def validate_root_exists(cls, v: Path) -> Path:
        """Validate that the root directory exists."""
        if not v.exists():
            raise ValueError(f"Data root directory does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"Data root must be a directory: {v}")
        return v


class OptimizerConfig(BaseModel):
    """Optimizer configuration validation."""
    
    lr: float = Field(gt=0.0, le=1.0, description="Learning rate")
    weight_decay: float = Field(ge=0.0, le=1.0, description="Weight decay")


class TrainerConfig(BaseModel):
    """Trainer configuration validation."""
    
    max_epochs: int = Field(ge=1, le=1000, description="Maximum training epochs")
    accelerator: Literal["cpu", "gpu", "auto"] = Field(
        default="cpu", description="Accelerator type"
    )
    devices: int = Field(ge=1, le=8, default=1, description="Number of devices")
    precision: str | int = Field(default="32-true", description="Training precision")
    limit_train_batches: int | float | None = Field(
        default=None, description="Limit training batches"
    )
    limit_val_batches: int | float | None = Field(
        default=None, description="Limit validation batches"
    )
    check_val_every_n_epoch: int = Field(
        ge=1, default=1, description="Validation check frequency"
    )
    deterministic: bool = Field(default=True, description="Enable deterministic training")


class Config(BaseModel):
    """Main configuration validation."""
    
    seed: int = Field(ge=0, le=2**32-1, description="Random seed")
    temperature: float = Field(gt=0.0, le=10.0, description="Temperature for contrastive learning")
    omics_input_dim: int = Field(ge=1, le=10000, description="Omics input dimension")
    output_dir: Path = Field(description="Output directory for artifacts")
    
    data: DataConfig
    optimizer: OptimizerConfig  
    trainer: TrainerConfig
    
    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        """Create output directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @model_validator(mode="after")
    def validate_config_consistency(self) -> Config:
        """Validate configuration consistency."""
        if self.trainer.accelerator == "gpu" and not self._gpu_available():
            raise ValueError("GPU accelerator requested but CUDA not available")
        
        if self.data.num_workers > 0 and self.trainer.accelerator == "cpu":
            # Warn about potential inefficiency but don't fail
            pass
            
        return self
    
    def _gpu_available(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


def validate_config(config_dict: dict[str, Any]) -> Config:
    """Validate and parse configuration dictionary.
    
    Args:
        config_dict: Raw configuration dictionary
        
    Returns:
        Validated configuration object
        
    Raises:
        ValueError: If configuration is invalid
    """
    try:
        return Config.model_validate(config_dict)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}") from e