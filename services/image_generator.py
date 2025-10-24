"""
Image generation service.
"""
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from PIL import Image

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generation_params import GenerationParams

# Lazy imports to avoid PyTorch compatibility issues at module level
def _get_diffusers_imports():
    """Lazy import of diffusers to avoid compatibility issues."""
    try:
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
        return StableDiffusionPipeline, StableDiffusionXLPipeline
    except ImportError as e:
        raise ImportError(f"Failed to import diffusers: {e}")

def _get_transformers_imports():
    """Lazy import of transformers to avoid compatibility issues."""
    try:
        from transformers import CLIPTokenizer
        return CLIPTokenizer
    except ImportError as e:
        raise ImportError(f"Failed to import transformers: {e}")


class ModelLoader(QThread):
    """Thread for loading models with progress reporting."""

    progress = pyqtSignal(str, int)  # (message, percentage)
    finished = pyqtSignal(object)  # Can be StableDiffusionPipeline or StableDiffusionXLPipeline
    error = pyqtSignal(str)

    def __init__(self, model_path: str, quantization: str = "None", use_xformers: bool = True):
        super().__init__()
        self.model_path = model_path
        self.quantization = quantization
        self.use_xformers = use_xformers

    def _is_sdxl_model(self, model_path: str) -> bool:
        """Check if the model is SDXL based on model index, config files, or filename."""
        try:
            if os.path.isdir(model_path):
                # Check for SDXL-specific files
                model_index = os.path.join(model_path, "model_index.json")
                if os.path.exists(model_index):
                    import json
                    with open(model_index, 'r') as f:
                        config = json.load(f)
                        # SDXL has different architecture
                        class_name = config.get("_class_name", "").lower()
                        return "_xl" in class_name or "xl" in class_name or "stable_diffusion_xl" in class_name

                # Check for SDXL-specific config files
                unet_config = os.path.join(model_path, "unet", "config.json")
                if os.path.exists(unet_config):
                    with open(unet_config, 'r') as f:
                        config = json.load(f)
                        # SDXL UNet has different structure
                        return "cross_attention_dim" in config and config.get("cross_attention_dim") == 2048

            elif os.path.isfile(model_path):
                # For single files, check filename patterns that often indicate SDXL
                filename = os.path.basename(model_path).lower()
                if any(keyword in filename for keyword in ['xl', 'sdxl', 'stable-diffusion-xl']):
                    print(f"Detected potential SDXL model based on filename: {filename}")
                    return True

                # Try to load a small portion of the safetensors file to check metadata
                if model_path.endswith('.safetensors'):
                    try:
                        from safetensors import safe_open
                        with safe_open(model_path, framework="pt", device="cpu") as f:
                            # Check if any tensor names contain SDXL indicators
                            tensor_names = list(f.keys())[:10]  # Check first 10 tensors
                            for name in tensor_names:
                                if 'xl' in name.lower() or 'stable_diffusion_xl' in name.lower():
                                    print(f"Detected SDXL model based on tensor names: {name}")
                                    return True
                    except Exception as e:
                        print(f"Could not check safetensors metadata: {e}")

            # If we can't determine, assume SD 1.5
            return False

        except Exception as e:
            print(f"Error in SDXL detection: {e}")
            # If we can't determine, assume SD 1.5
            return False

    def run(self):
        """Load model in background thread with progress updates."""
        try:
            self.progress.emit("Initializing model loading...", 0)

            # Detect if this is an SDXL model
            is_sdxl = self._is_sdxl_model(self.model_path)
            StableDiffusionPipeline, StableDiffusionXLPipeline = _get_diffusers_imports()
            pipeline_class = StableDiffusionXLPipeline if is_sdxl else StableDiffusionPipeline

            self.progress.emit(f"Loading {'SDXL' if is_sdxl else 'SD 1.5'} model components...", 25)

            # Set up quantization and dtype
            torch_dtype = torch.float16
            load_in_8bit = False
            load_in_4bit = False

            if self.quantization == "8-bit":
                load_in_8bit = True
                torch_dtype = torch.float16  # 8-bit loads as float16 then quantizes
                print("Using 8-bit quantization")
            elif self.quantization == "4-bit":
                load_in_4bit = True
                torch_dtype = torch.float16  # 4-bit loads as float16 then quantizes
                print("Using 4-bit quantization")

            if os.path.isdir(self.model_path):
                # Load from directory (diffusers format)
                model = pipeline_class.from_pretrained(
                    self.model_path,
                    torch_dtype=torch_dtype,
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit,
                    local_files_only=True
                )
            elif os.path.isfile(self.model_path) and (self.model_path.endswith('.safetensors') or self.model_path.endswith('.ckpt')):
                # Load from single file
                if is_sdxl:
                    # SDXL single-file loading - updated to match sample code
                    model = pipeline_class.from_single_file(
                        self.model_path,
                        torch_dtype=torch.float16,  # Explicit torch.float16
                        variant="fp16",             # Required for SDXL
                        use_safetensors=True,       # Ensure safetensors format
                        local_files_only=True
                    )
                    # Move to GPU immediately after loading (matching sample)
                    if torch.cuda.is_available():
                        model = model.to("cuda")
                else:
                    # SD 1.5 single-file loading - updated for consistency
                    model = StableDiffusionPipeline.from_single_file(
                        self.model_path,
                        torch_dtype=torch.float16,  # Explicit torch.float16
                        use_safetensors=True,       # Ensure safetensors format
                        local_files_only=True
                    )

                    # Re-enable safety checker for single-file models
                    self.progress.emit("Loading safety checker...", 50)
                    if hasattr(model, 'safety_checker') and model.safety_checker is None:
                        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
                        from transformers import CLIPFeatureExtractor
                        try:
                            safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                                "CompVis/stable-diffusion-safety-checker",
                                torch_dtype=torch.float16
                            )
                            feature_extractor = CLIPFeatureExtractor.from_pretrained(
                                "openai/clip-vit-base-patch32"
                            )
                            model.safety_checker = safety_checker
                            model.feature_extractor = feature_extractor
                        except Exception as e:
                            print(f"Warning: Could not load safety checker: {e}")

            self.progress.emit("Enabling optimizations...", 75)

            # Enable optimizations
            model.enable_attention_slicing()

            # Enable xFormers if requested and available
            if self.use_xformers:
                try:
                    # First check if xformers is installed
                    import xformers
                    print(f"xFormers {xformers.__version__} detected")

                    # Check CUDA compatibility
                    if torch.cuda.is_available():
                        cuda_version = torch.version.cuda
                        print(f"CUDA version: {cuda_version}")

                        # Try to enable xFormers
                        try:
                            model.enable_xformers_memory_efficient_attention()
                            print("✅ xFormers memory efficient attention enabled successfully!")
                            print("   Expected performance improvement: 2-3x faster generation")
                        except Exception as xformers_error:
                            print(f"⚠️  xFormers failed to enable: {xformers_error}")
                            print("   This is likely due to version incompatibility with PyTorch 2.6.0")
                            print("   The app will still work with standard attention (just slower)")
                            print("   To fix: Wait for xFormers to support PyTorch 2.6.0, or downgrade PyTorch")
                    else:
                        print("⚠️  CUDA not available - xFormers requires CUDA")
                        print("   Falling back to standard attention")

                except ImportError as e:
                    print("ℹ️  xFormers not installed - this is optional")
                    print("   The app will work with standard attention (just slower)")
                    print("   To install xFormers (optional performance boost):")
                    print("   pip install xformers --index-url https://download.pytorch.org/whl/cu118")
                    print("   Note: May not be compatible with PyTorch 2.6.0 yet")

                except Exception as e:
                    print(f"⚠️  xFormers initialization failed: {e}")
                    print("   Falling back to standard attention - app will still work normally")

            # Move to GPU if available
            if torch.cuda.is_available():
                self.progress.emit("Moving model to GPU...", 90)
                model.to("cuda")

            self.progress.emit("Model loaded successfully!", 100)
            self.finished.emit(model)

        except Exception as e:
            self.error.emit(f"Model loading failed: {str(e)}")


class ImageGenerator(QThread):
    """Thread for generating images using Stable Diffusion or SDXL."""

    finished = pyqtSignal(Image.Image)
    error = pyqtSignal(str)

    def __init__(self, model, prompt: str, params: GenerationParams):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.params = params

    def _get_scheduler_config(self, scheduler_name: str) -> dict:
        """Get scheduler configuration for the specified scheduler."""
        # Lazy import schedulers to avoid compatibility issues
        try:
            # Import diffusers first to get access to schedulers
            import diffusers
            DDIMScheduler = diffusers.DDIMScheduler
            DPMSolverMultistepScheduler = diffusers.DPMSolverMultistepScheduler
            DPMSolverSinglestepScheduler = diffusers.DPMSolverSinglestepScheduler
            EulerAncestralDiscreteScheduler = diffusers.EulerAncestralDiscreteScheduler
            EulerDiscreteScheduler = diffusers.EulerDiscreteScheduler
            HeunDiscreteScheduler = diffusers.HeunDiscreteScheduler
            KDPM2AncestralDiscreteScheduler = diffusers.KDPM2AncestralDiscreteScheduler
            KDPM2DiscreteScheduler = diffusers.KDPM2DiscreteScheduler
            PNDMScheduler = diffusers.PNDMScheduler
            UniPCMultistepScheduler = diffusers.UniPCMultistepScheduler
        except ImportError as e:
            print(f"Failed to import schedulers: {e}")
            return None

        schedulers = {
            "DDIM": {
                'class': DDIMScheduler,
                'kwargs': {}
            },
            "DPM++ 2M": {
                'class': DPMSolverMultistepScheduler,
                'kwargs': {'solver_order': 2}
            },
            "DPM++ 2M Karras": {
                'class': DPMSolverMultistepScheduler,
                'kwargs': {'solver_order': 2, 'use_karras_sigmas': True}
            },
            "DPM++ SDE": {
                'class': DPMSolverSinglestepScheduler,
                'kwargs': {}
            },
            "DPM++ SDE Karras": {
                'class': DPMSolverSinglestepScheduler,
                'kwargs': {'use_karras_sigmas': True}
            },
            "Euler": {
                'class': EulerDiscreteScheduler,
                'kwargs': {}
            },
            "Euler A": {
                'class': EulerAncestralDiscreteScheduler,
                'kwargs': {}
            },
            "Heun": {
                'class': HeunDiscreteScheduler,
                'kwargs': {}
            },
            "KDPM2": {
                'class': KDPM2DiscreteScheduler,
                'kwargs': {}
            },
            "KDPM2 A": {
                'class': KDPM2AncestralDiscreteScheduler,
                'kwargs': {}
            },
            "PNDM": {
                'class': PNDMScheduler,
                'kwargs': {}
            },
            "UniPC": {
                'class': UniPCMultistepScheduler,
                'kwargs': {}
            }
        }

        return schedulers.get(scheduler_name)

    def _truncate_prompt(self, prompt: str, max_tokens: int = 75) -> str:
        """Truncate prompt to fit within token limit."""
        if not prompt:
            return prompt

        # SDXL can handle more tokens than SD 1.5
        _, StableDiffusionXLPipeline = _get_diffusers_imports()
        if isinstance(self.model, StableDiffusionXLPipeline):
            max_tokens = 77  # SDXL token limit

        try:
            # Try to use the model's tokenizer if available
            if hasattr(self.model, 'tokenizer'):
                tokenizer = self.model.tokenizer
            else:
                # Fallback to CLIP tokenizer
                CLIPTokenizer = _get_transformers_imports()
                tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

            # Encode and truncate
            tokens = tokenizer.encode(prompt)
            if len(tokens) > max_tokens:
                truncated_tokens = tokens[:max_tokens]
                truncated_prompt = tokenizer.decode(truncated_tokens)
                print(f"Prompt truncated from {len(tokens)} to {len(truncated_tokens)} tokens (max: {max_tokens})")
                return truncated_prompt
            return prompt

        except Exception as e:
            print(f"Warning: Could not tokenize prompt for truncation: {e}")
            # Fallback: truncate by character count (rough approximation)
            if len(prompt) > 500:  # Rough character limit
                return prompt[:500] + "..."
            return prompt

    def run(self):
        """Generate image in background thread."""
        try:
            # Get pipeline classes for type checking
            _, StableDiffusionXLPipeline = _get_diffusers_imports()
            model_type = 'SDXL' if isinstance(self.model, StableDiffusionXLPipeline) else 'SD 1.5'
            print(f"Starting image generation with {model_type} model")
            print(f"Prompt length: {len(self.prompt)} characters")
            print(f"Parameters: steps={self.params.steps}, guidance={self.params.guidance_scale}, size={self.params.width}x{self.params.height}")

            # Truncate prompt if necessary
            original_prompt = self.prompt
            self.prompt = self._truncate_prompt(self.prompt)

            if self.prompt != original_prompt:
                print(f"Prompt was truncated for generation to fit token limit")
                print(f"Original: {len(original_prompt)} chars, Truncated: {len(self.prompt)} chars")

            # Set up generator for reproducible results
            generator = None
            if self.params.seed != 0:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                generator = torch.Generator(device=device).manual_seed(self.params.seed)
                print(f"Using seed: {self.params.seed} on {device}")

            # Set up scheduler based on user selection
            scheduler_config = self._get_scheduler_config(self.params.scheduler)
            if scheduler_config:
                # Schedulers are already imported in _get_scheduler_config
                scheduler_class = scheduler_config['class']
                scheduler_kwargs = scheduler_config.get('kwargs', {})

                # Create and set scheduler
                scheduler = scheduler_class.from_config(self.model.scheduler.config, **scheduler_kwargs)
                self.model.scheduler = scheduler
                print(f"Using scheduler: {self.params.scheduler}")

            # Enable VAE tiling for large images if requested
            if self.params.vae_tiling and (self.params.width > 1024 or self.params.height > 1024):
                try:
                    self.model.enable_vae_tiling()
                    print(f"VAE tiling enabled for large image: {self.params.width}x{self.params.height}")
                except Exception as e:
                    print(f"VAE tiling not available: {e}")

            print("Starting inference...")

            # Generate image - handle both SD and SDXL pipelines
            if isinstance(self.model, StableDiffusionXLPipeline):
                # SDXL pipeline
                print("Using SDXL pipeline")
                result = self.model(
                    prompt=self.prompt,
                    negative_prompt=self.params.negative_prompt,
                    num_inference_steps=self.params.steps,
                    guidance_scale=self.params.guidance_scale,
                    width=self.params.width,
                    height=self.params.height,
                    generator=generator,
                )
            else:
                # Standard SD pipeline
                print("Using SD 1.5 pipeline")
                result = self.model(
                    prompt=self.prompt,
                    negative_prompt=self.params.negative_prompt,
                    num_inference_steps=self.params.steps,
                    guidance_scale=self.params.guidance_scale,
                    width=self.params.width,
                    height=self.params.height,
                    generator=generator,
                )

            print("Inference completed, processing result...")
            image = result.images[0]
            print(f"Generated image: {image.size} pixels")
            self.finished.emit(image)

        except Exception as e:
            print(f"Image generation failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.error.emit(f"Image generation failed: {str(e)}")


class ImageGenerationService:
    """Service for managing image generation operations."""

    MODELS_DIR = "models"

    def __init__(self):
        self.model = None  # Can be StableDiffusionPipeline or StableDiffusionXLPipeline
        self._current_model_path: str = None
        self._ensure_models_directory()

    def _ensure_models_directory(self) -> None:
        """Ensure models directory exists."""
        os.makedirs(self.MODELS_DIR, exist_ok=True)

    def _get_local_model_path(self, model_name: str) -> str:
        """Get the local path for a cached model."""
        return os.path.join(self.MODELS_DIR, model_name)

    def _download_model_locally(self, model_id: str, local_name: str) -> str:
        """Download and cache a model locally."""
        local_path = self._get_local_model_path(local_name)

        try:
            print(f"Downloading model {model_id} to {local_path}...")

            # Create directory if it doesn't exist
            os.makedirs(local_path, exist_ok=True)

            # Download the model
            StableDiffusionPipeline, _ = _get_diffusers_imports()
            temp_model = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                local_files_only=False
            )

            # Save the model locally
            temp_model.save_pretrained(local_path, safe_serialization=True)

            print(f"Model downloaded and cached at {local_path}")
            return local_path

        except Exception as e:
            print(f"Failed to download model: {e}")
            raise

    def load_model_async(self, model_path: str, quantization: str = "None", use_xformers: bool = True) -> ModelLoader:
        """Load a model asynchronously with progress reporting."""
        return ModelLoader(model_path, quantization, use_xformers)

    def _is_sdxl_model(self, model_path: str) -> bool:
        """Check if the model is SDXL based on model index, config files, or filename."""
        try:
            if os.path.isdir(model_path):
                # Check for SDXL-specific files
                model_index = os.path.join(model_path, "model_index.json")
                if os.path.exists(model_index):
                    import json
                    with open(model_index, 'r') as f:
                        config = json.load(f)
                        # SDXL has different architecture
                        class_name = config.get("_class_name", "").lower()
                        return "_xl" in class_name or "xl" in class_name or "stable_diffusion_xl" in class_name

                # Check for SDXL-specific config files
                unet_config = os.path.join(model_path, "unet", "config.json")
                if os.path.exists(unet_config):
                    with open(unet_config, 'r') as f:
                        config = json.load(f)
                        # SDXL UNet has different structure
                        return "cross_attention_dim" in config and config.get("cross_attention_dim") == 2048

            elif os.path.isfile(model_path):
                # For single files, check filename patterns that often indicate SDXL
                filename = os.path.basename(model_path).lower()
                if any(keyword in filename for keyword in ['xl', 'sdxl', 'stable-diffusion-xl']):
                    print(f"Detected potential SDXL model based on filename: {filename}")
                    return True

                # Try to load a small portion of the safetensors file to check metadata
                if model_path.endswith('.safetensors'):
                    try:
                        from safetensors import safe_open
                        with safe_open(model_path, framework="pt", device="cpu") as f:
                            # Check if any tensor names contain SDXL indicators
                            tensor_names = list(f.keys())[:10]  # Check first 10 tensors
                            for name in tensor_names:
                                if 'xl' in name.lower() or 'stable_diffusion_xl' in name.lower():
                                    print(f"Detected SDXL model based on tensor names: {name}")
                                    return True
                    except Exception as e:
                        print(f"Could not check safetensors metadata: {e}")

            # If we can't determine, assume SD 1.5
            return False

        except Exception as e:
            print(f"Error in SDXL detection: {e}")
            # If we can't determine, assume SD 1.5
            return False

    def load_model(self, model_path: str = None) -> bool:
        """Load the Stable Diffusion or SDXL model."""
        try:
            if model_path:
                # Detect if this is an SDXL model
                is_sdxl = self._is_sdxl_model(model_path)
                StableDiffusionPipeline, StableDiffusionXLPipeline = _get_diffusers_imports()
                pipeline_class = StableDiffusionXLPipeline if is_sdxl else StableDiffusionPipeline

                print(f"Loading {'SDXL' if is_sdxl else 'SD 1.5'} model from {model_path}")

                # Load custom model from specified path
                if os.path.exists(model_path):
                    if os.path.isdir(model_path):
                        # Load from directory (diffusers format)
                        self.model = pipeline_class.from_pretrained(
                            model_path,
                            torch_dtype=torch.float16,
                            local_files_only=True
                        )
                    elif os.path.isfile(model_path) and (model_path.endswith('.safetensors') or model_path.endswith('.ckpt')):
                        # Load from single file (safetensors or ckpt format)
                        if is_sdxl:
                            # SDXL single-file loading - updated to match sample code
                            self.model = pipeline_class.from_single_file(
                                model_path,
                                torch_dtype=torch.float16,  # Explicit torch.float16
                                variant="fp16",             # Required for SDXL
                                use_safetensors=True,       # Ensure safetensors format
                                local_files_only=True
                            )
                            # Move to GPU immediately after loading (matching sample)
                            if torch.cuda.is_available():
                                self.model = self.model.to("cuda")
                        else:
                            # SD 1.5 single-file loading - updated for consistency
                            self.model = StableDiffusionPipeline.from_single_file(
                                model_path,
                                torch_dtype=torch.float16,  # Explicit torch.float16
                                use_safetensors=True,       # Ensure safetensors format
                                local_files_only=True
                            )
                            # Re-enable safety checker for single-file models
                            if hasattr(self.model, 'safety_checker') and self.model.safety_checker is None:
                                from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
                                from transformers import CLIPFeatureExtractor
                                try:
                                    # Try to load safety checker components
                                    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                                        "CompVis/stable-diffusion-safety-checker",
                                        torch_dtype=torch.float16
                                    )
                                    feature_extractor = CLIPFeatureExtractor.from_pretrained(
                                        "openai/clip-vit-base-patch32"
                                    )
                                    self.model.safety_checker = safety_checker
                                    self.model.feature_extractor = feature_extractor
                                except Exception as e:
                                    print(f"Warning: Could not load safety checker: {e}")
                    else:
                        raise ValueError(f"Unsupported model format: {model_path}")
                else:
                    raise FileNotFoundError(f"Model path does not exist: {model_path}")
            else:
                # Load default model - try local cache first, then download
                default_model_name = "stable-diffusion-v1-4"
                local_path = self._get_local_model_path(default_model_name)

                if os.path.exists(local_path):
                    # Load from local cache
                    StableDiffusionPipeline, _ = _get_diffusers_imports()
                    self.model = StableDiffusionPipeline.from_pretrained(
                        local_path,
                        torch_dtype=torch.float16,
                        local_files_only=True
                    )
                else:
                    # Download and cache locally
                    local_path = self._download_model_locally("CompVis/stable-diffusion-v1-4", default_model_name)

                    # Reload from local cache to ensure consistency
                    StableDiffusionPipeline, _ = _get_diffusers_imports()
                    self.model = StableDiffusionPipeline.from_pretrained(
                        local_path,
                        torch_dtype=torch.float16,
                        local_files_only=True
                    )

            # Enable optimizations
            self.model.enable_attention_slicing()

            # Move to GPU if available
            if torch.cuda.is_available():
                self.model.to("cuda")

            # Store the current model path
            self._current_model_path = model_path or local_path

            return True

        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def generate_image(self, prompt: str, params: GenerationParams) -> ImageGenerator:
        """Create image generator thread."""
        if not self.model:
            raise RuntimeError("Model not loaded")

        return ImageGenerator(self.model, prompt, params)

    def unload_model(self):
        """Unload the model to free memory."""
        if self.model:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
