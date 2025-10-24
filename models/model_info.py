"""
Model information data models.
"""
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


class ModelType(Enum):
    """Supported model types."""
    STABLE_DIFFUSION_V1_4 = "stable-diffusion-v1-4"
    STABLE_DIFFUSION_V1_5 = "stable-diffusion-v1-5"
    STABLE_DIFFUSION_XL = "stable-diffusion-xl"


class ModelCategory(Enum):
    """Model categories for organization."""
    REALISTIC = "realistic"
    ANIME = "anime"
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    ABSTRACT = "abstract"
    ARTISTIC = "artistic"
    FANTASY = "fantasy"
    SCI_FI = "sci-fi"
    OTHER = "other"


@dataclass
class ModelInfo:
    """Information about an installed model."""

    name: str  # Unique identifier (filename without extension)
    path: str
    display_name: str = ""  # User-friendly display name
    model_type: Optional[ModelType] = None  # Optional model type
    description: str = ""
    categories: List[ModelCategory] = field(default_factory=list)
    usage_notes: str = ""
    source_url: Optional[str] = None
    license_info: Optional[str] = None
    is_default: bool = False
    size_mb: Optional[float] = None
    installed_date: Optional[str] = None
    last_used: Optional[str] = None
    usage_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "path": self.path,
            "display_name": self.display_name,
            "model_type": self.model_type.value if self.model_type else None,
            "description": self.description,
            "categories": [cat.value for cat in self.categories],
            "usage_notes": self.usage_notes,
            "source_url": self.source_url,
            "license_info": self.license_info,
            "is_default": self.is_default,
            "size_mb": self.size_mb,
            "installed_date": self.installed_date,
            "last_used": self.last_used,
            "usage_count": self.usage_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        """Create from dictionary."""
        model_type = None
        if data.get("model_type"):
            try:
                model_type = ModelType(data["model_type"])
            except ValueError:
                model_type = None

        return cls(
            name=data["name"],
            path=data["path"],
            display_name=data.get("display_name", ""),
            model_type=model_type,
            description=data.get("description", ""),
            categories=[ModelCategory(cat) for cat in data.get("categories", [])],
            usage_notes=data.get("usage_notes", ""),
            source_url=data.get("source_url"),
            license_info=data.get("license_info"),
            is_default=data.get("is_default", False),
            size_mb=data.get("size_mb"),
            installed_date=data.get("installed_date"),
            last_used=data.get("last_used"),
            usage_count=data.get("usage_count", 0),
        )


@dataclass
class GenerationParams:
    """Parameters for image generation."""

    steps: int = 20
    guidance_scale: float = 7.5
    seed: int = 0
    negative_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationParams':
        """Create from dictionary."""
        return cls(
            steps=data.get("steps", 20),
            guidance_scale=data.get("guidance_scale", 7.5),
            seed=data.get("seed", 0),
            negative_prompt=data.get("negative_prompt", ""),
        )
