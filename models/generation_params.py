"""
Image generation parameters data model.
"""
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class GenerationParams:
    """Parameters for image generation."""

    steps: int = 20
    guidance_scale: float = 7.5
    seed: int = 0
    negative_prompt: str = ""
    width: int = 512
    height: int = 512

    # Optimization parameters
    scheduler: str = "DPM++ 2M Karras"  # DDIM, DPM++ 2M, DPM++ 2M Karras, Euler A, UniPC
    quantization: str = "None"  # None, 8-bit, 4-bit
    use_xformers: bool = True  # Use xFormers attention optimization
    vae_tiling: bool = True  # Use VAE tiling for large images

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "scheduler": self.scheduler,
            "quantization": self.quantization,
            "use_xformers": self.use_xformers,
            "vae_tiling": self.vae_tiling,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationParams':
        """Create from dictionary."""
        return cls(
            steps=data.get("steps", 20),
            guidance_scale=data.get("guidance_scale", 7.5),
            seed=data.get("seed", 0),
            negative_prompt=data.get("negative_prompt", ""),
            width=data.get("width", 512),
            height=data.get("height", 512),
            scheduler=data.get("scheduler", "DPM++ 2M Karras"),
            quantization=data.get("quantization", "None"),
            use_xformers=data.get("use_xformers", True),
            vae_tiling=data.get("vae_tiling", True),
        )
