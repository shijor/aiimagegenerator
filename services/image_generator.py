"""
Image generation service.
"""
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from PIL import Image
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generation_params import GenerationParams

# Lazy imports to avoid PyTorch compatibility issues at module level
def _get_diffusers_imports():
    """Lazy import of diffusers to avoid compatibility issues."""
    try:
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
        # IP-Adapter is loaded as an adapter, not a separate pipeline class in newer diffusers versions
        return StableDiffusionPipeline, StableDiffusionXLPipeline, None
    except ImportError as e:
        raise ImportError(f"Failed to import diffusers: {e}")

def _get_lora_imports():
    """Lazy import of LoRA-related modules to avoid compatibility issues."""
    try:
        from peft import LoraConfig, PeftModel
        return LoraConfig, PeftModel
    except ImportError as e:
        raise ImportError(f"Failed to import PEFT for LoRA support: {e}. Install with: pip install peft")

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

    def __init__(self, model_path: str, quantization: str = "None", use_xformers: bool = True, cpu_offload: bool = False):
        super().__init__()
        self.model_path = model_path
        self.quantization = quantization
        self.use_xformers = use_xformers
        self.cpu_offload = cpu_offload

    def _is_diffusers_model_directory(self, dir_path: str) -> bool:
        """Check if a directory contains a valid diffusers model."""
        try:
            print(f"VALIDATING DIFFUSERS DIR: Checking directory: {dir_path}")

            # Check if model_index.json exists (main indicator of diffusers model)
            model_index_path = os.path.join(dir_path, "model_index.json")
            if not os.path.exists(model_index_path):
                print(f"VALIDATING DIFFUSERS DIR: model_index.json not found at: {model_index_path}")
                return False

            print(f"VALIDATING DIFFUSERS DIR: Found model_index.json at: {model_index_path}")

            # Validate that model_index.json is a valid JSON file
            with open(model_index_path, 'r', encoding='utf-8') as f:
                model_config = json.load(f)

            print(f"VALIDATING DIFFUSERS DIR: Loaded JSON with keys: {list(model_config.keys())}")

            # Check for required diffusers model components
            required_components = ['_class_name']  # At minimum, should have a class name
            if not all(key in model_config for key in required_components):
                print(f"VALIDATING DIFFUSERS DIR: Missing required components: {required_components}")
                return False

            print(f"VALIDATING DIFFUSERS DIR: Has required _class_name: {model_config.get('_class_name')}")

            # Check for typical diffusers subdirectories (be more lenient)
            expected_dirs = ['scheduler', 'text_encoder', 'tokenizer', 'unet', 'vae']
            existing_dirs = [d for d in expected_dirs if os.path.exists(os.path.join(dir_path, d))]
            print(f"VALIDATING DIFFUSERS DIR: Found subdirectories: {existing_dirs}")

            # Be more lenient - just require model_index.json with _class_name
            # The actual component validation can be done during loading
            has_components = len(existing_dirs) > 0
            print(f"VALIDATING DIFFUSERS DIR: Has components check: {has_components}")

            # Additional validation: check if it has model files
            has_model_files = False
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(('.safetensors', '.bin', '.pt', '.pth')):
                        has_model_files = True
                        print(f"VALIDATING DIFFUSERS DIR: Found model file: {os.path.join(root, file)}")
                        break
                if has_model_files:
                    break

            print(f"VALIDATING DIFFUSERS DIR: Has model files: {has_model_files}")

            # For now, be more lenient and just require model_index.json with _class_name
            # This allows for different diffusers model structures
            result = '_class_name' in model_config
            print(f"VALIDATING DIFFUSERS DIR: Final validation result: {result}")
            return result

        except (json.JSONDecodeError, IOError, OSError) as e:
            print(f"VALIDATING DIFFUSERS DIR: Error validating directory {dir_path}: {str(e)}")
            return False

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
            print("🔍 DEBUG: Starting model loading thread")
            print(f"🔍 DEBUG: Model path: {self.model_path}")
            print(f"🔍 DEBUG: Quantization: {self.quantization}")
            print(f"🔍 DEBUG: xFormers: {self.use_xformers}")
            print(f"🔍 DEBUG: CPU offload: {self.cpu_offload}")

            # Check system resources
            import torch
            if torch.cuda.is_available():
                print(f"🔍 DEBUG: CUDA available - {torch.cuda.get_device_name(0)}")
                print(f"🔍 DEBUG: CUDA memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB")
            else:
                print("🔍 DEBUG: CUDA not available, using CPU")

            self.progress.emit("Initializing model loading...", 0)

            # Detect if this is an SDXL model
            is_sdxl = self._is_sdxl_model(self.model_path)
            StableDiffusionPipeline, StableDiffusionXLPipeline, _ = _get_diffusers_imports()
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

            # Determine if this is a diffusers directory or single file
            is_diffusers_directory = os.path.isdir(self.model_path) and self._is_diffusers_model_directory(self.model_path)

            if is_diffusers_directory:
                print(f"🔍 DEBUG: Loading as diffusers directory: {self.model_path}")
                # Load from directory (diffusers format)
                model = pipeline_class.from_pretrained(
                    self.model_path,
                    torch_dtype=torch_dtype,
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit,
                    local_files_only=True
                )
            elif os.path.isfile(self.model_path) and (self.model_path.endswith('.safetensors') or self.model_path.endswith('.ckpt')):
                print(f"🔍 DEBUG: Loading as single file: {self.model_path}")
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
            else:
                raise ValueError(f"Unsupported model format or path: {self.model_path}")

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

            self.progress.emit("Model loaded successfully!", 100)
            # Note: Optimizations like attention slicing will be enabled after IP-Adapters are loaded

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

            # Enable CPU offload if requested (saves VRAM but slower)
            if self.cpu_offload:
                self.progress.emit("Enabling CPU offload...", 85)
                try:
                    model.enable_sequential_cpu_offload()
                    print("✅ Sequential CPU offload enabled - model components will be moved to CPU when not needed")
                    print("   This reduces VRAM usage but may slow down generation")
                except Exception as e:
                    print(f"⚠️  CPU offload not available: {e}")
                    print("   Continuing without CPU offload")

            # Move to GPU if available (only if not using CPU offload)
            if torch.cuda.is_available() and not self.cpu_offload:
                self.progress.emit("Moving model to GPU...", 90)
                model.to("cuda")

            self.progress.emit("Model loaded successfully!", 100)
            self.finished.emit(model)

        except Exception as e:
            print(f"🔍 DEBUG: Model loading failed with exception: {str(e)}")
            import traceback
            traceback.print_exc()
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

    def _inject_lora_trigger_words(self, prompt: str) -> str:
        """Inject LoRA trigger words into the prompt."""
        # Access the loaded LoRAs from the ImageGenerationService
        # We need to get this from the service that created this generator
        # For now, we'll assume the model has access to loaded LoRAs through some mechanism

        # Check if the model has loaded LoRAs (this would be set by the service)
        if not hasattr(self.model, '_loaded_loras') or not self.model._loaded_loras:
            return prompt

        trigger_words = []
        for lora_name, (lora_path, scaling) in self.model._loaded_loras.items():
            # We need to get the LoRA info to access trigger words
            # This requires access to the LoRA database or cached info
            # For now, we'll use a placeholder - in practice, this would be
            # populated by the ImageGenerationService when creating the generator

            # Try to get trigger words from model attributes (set by service)
            if hasattr(self.model, f'_lora_trigger_words_{lora_name}'):
                words = getattr(self.model, f'_lora_trigger_words_{lora_name}')
                if words:
                    trigger_words.extend(words)

        if not trigger_words:
            return prompt

        # Inject trigger words at the beginning of the prompt
        enhanced_prompt = ", ".join(trigger_words) + ", " + prompt
        return enhanced_prompt

    def _truncate_prompt(self, prompt: str, max_tokens: int = 75) -> str:
        """Truncate prompt to fit within token limit."""
        if not prompt:
            return prompt

        # SDXL can handle more tokens than SD 1.5
        _, StableDiffusionXLPipeline, _ = _get_diffusers_imports()
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
            # Check for interruption before starting
            if self.isInterruptionRequested():
                print("Image generation cancelled before starting")
                return

            # Debug: Show PyTorch version being used
            import torch
            print("🔍 DEBUG: Starting image generation thread")
            print(f"🔍 DEBUG: PyTorch version: {torch.__version__}")
            print(f"🔍 DEBUG: CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"🔍 DEBUG: CUDA version: {torch.version.cuda}")
                print(f"🔍 DEBUG: GPU: {torch.cuda.get_device_name(0)}")

            # Get pipeline classes for type checking
            _, StableDiffusionXLPipeline, _ = _get_diffusers_imports()
            model_type = 'SDXL' if isinstance(self.model, StableDiffusionXLPipeline) else 'SD 1.5'
            print(f"🔍 DEBUG: Model type detected: {model_type}")
            print(f"🔍 DEBUG: Prompt length: {len(self.prompt)} characters")
            print(f"🔍 DEBUG: Generation parameters:")
            print(f"   - Steps: {self.params.steps}")
            print(f"🔍 DEBUG: Guidance Scale: {self.params.guidance_scale}")
            print(f"🔍 DEBUG: Dimensions: {self.params.width}x{self.params.height}")
            print(f"🔍 DEBUG: Scheduler: {self.params.scheduler}")
            print(f"🔍 DEBUG: Seed: {self.params.seed}")
            print(f"🔍 DEBUG: VAE Tiling: {self.params.vae_tiling}")
            print(f"🔍 DEBUG: NSFW Filter: {self.params.enable_nsfw_filter}")

            # Handle NSFW filter setting
            original_safety_checker = None
            if not self.params.enable_nsfw_filter:
                # Temporarily disable safety checker
                original_safety_checker = self.model.safety_checker
                self.model.safety_checker = None
                print("🔍 DEBUG: NSFW filter disabled - safety checker temporarily removed")

            # Check for interruption after initial setup
            if self.isInterruptionRequested():
                print("Image generation cancelled during setup")
                return

            # Check memory usage before generation
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                print(f"🔍 DEBUG: GPU memory before generation: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

            # Inject LoRA trigger words into prompt if available
            original_prompt = self.prompt
            self.prompt = self._inject_lora_trigger_words(self.prompt)

            if self.prompt != original_prompt:
                print(f"LoRA trigger words injected into prompt")
                print(f"Original: '{original_prompt}'")
                print(f"Enhanced: '{self.prompt}'")

            # Truncate prompt if necessary
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

            # Enable VAE tiling for large images or SDXL models
            enable_vae_tiling = False

            # Check if user explicitly requested VAE tiling
            if self.params.vae_tiling:
                enable_vae_tiling = True
            # For SDXL models, enable VAE tiling for 1024x1024 and larger images
            elif isinstance(self.model, StableDiffusionXLPipeline):
                if self.params.width >= 1024 or self.params.height >= 1024:
                    enable_vae_tiling = True
                    print(f"SDXL detected - enabling VAE tiling for {self.params.width}x{self.params.height} image")

            if enable_vae_tiling:
                try:
                    self.model.enable_vae_tiling()
                    print(f"VAE tiling enabled for {'SDXL' if isinstance(self.model, StableDiffusionXLPipeline) else 'SD'} image: {self.params.width}x{self.params.height}")
                except Exception as e:
                    print(f"VAE tiling not available: {e}")

            # Check for interruption before starting inference
            if self.isInterruptionRequested():
                print("Image generation cancelled before inference")
                return

            # Debug: Show active LoRAs during generation
            if hasattr(self.model, 'active_adapters') and self.model.active_adapters:
                print(f"🔍 DEBUG: Active LoRA adapters during generation: {self.model.active_adapters}")
            elif hasattr(self.model, '_active_adapters') and self.model._active_adapters:
                print(f"🔍 DEBUG: Active LoRA adapters during generation: {self.model._active_adapters}")
            else:
                print("🔍 DEBUG: No active LoRA adapters detected during generation")

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

            # Check for interruption after inference
            if self.isInterruptionRequested():
                print("Image generation cancelled after inference")
                return

            print("Inference completed, processing result...")
            image = result.images[0]
            print(f"Generated image: {image.size} pixels")
            self.finished.emit(image)

        except Exception as e:
            print(f"Image generation failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.error.emit(f"Image generation failed: {str(e)}")
        finally:
            # Restore the original safety checker if it was temporarily disabled
            if original_safety_checker is not None:
                self.model.safety_checker = original_safety_checker
                print("🔍 DEBUG: NSFW filter restored - safety checker re-enabled")


class IPAdapterImageGenerator(QThread):
    """Thread for generating images using IP-Adapter conditioning."""

    finished = pyqtSignal(Image.Image)
    error = pyqtSignal(str)

    def __init__(self, model, prompt: str, params: GenerationParams, reference_image: Image.Image, ip_adapters: dict):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.params = params
        self.reference_image = reference_image
        self.ip_adapters = ip_adapters

    def run(self):
        """Generate image with IP-Adapter conditioning in background thread."""
        try:
            # Check for interruption before starting
            if self.isInterruptionRequested():
                print("IP-Adapter image generation cancelled before starting")
                return

            import torch
            print("🔍 DEBUG: Starting IP-Adapter image generation thread")
            print(f"🔍 DEBUG: PyTorch version: {torch.__version__}")
            print(f"🔍 DEBUG: CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"🔍 DEBUG: CUDA version: {torch.version.cuda}")
                print(f"🔍 DEBUG: GPU: {torch.cuda.get_device_name(0)}")

            print(f"🔍 DEBUG: Prompt length: {len(self.prompt)} characters")
            print(f"🔍 DEBUG: Generation parameters:")
            print(f"   - Steps: {self.params.steps}")
            print(f"🔍 DEBUG: Guidance Scale: {self.params.guidance_scale}")
            print(f"🔍 DEBUG: Dimensions: {self.params.width}x{self.params.height}")
            print(f"🔍 DEBUG: Scheduler: {self.params.scheduler}")
            print(f"🔍 DEBUG: Seed: {self.params.seed}")
            print(f"🔍 DEBUG: Reference image size: {self.reference_image.size}")
            print(f"🔍 DEBUG: IP-Adapters loaded: {list(self.ip_adapters.keys())}")

            # Handle NSFW filter setting
            original_safety_checker = None
            if not self.params.enable_nsfw_filter:
                # Temporarily disable safety checker
                original_safety_checker = self.model.safety_checker
                self.model.safety_checker = None
                print("🔍 DEBUG: NSFW filter disabled - safety checker temporarily removed")

            # Check for interruption after initial setup
            if self.isInterruptionRequested():
                print("IP-Adapter image generation cancelled during setup")
                return

            # Check memory usage before generation
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                print(f"🔍 DEBUG: GPU memory before generation: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

            # Inject LoRA trigger words into prompt if available (same as regular generation)
            original_prompt = self.prompt
            self.prompt = self._inject_lora_trigger_words(self.prompt)

            if self.prompt != original_prompt:
                print(f"LoRA trigger words injected into IP-Adapter prompt")
                print(f"Original: '{original_prompt}'")
                print(f"Enhanced: '{self.prompt}'")

            # Truncate prompt if necessary
            self.prompt = self._truncate_prompt(self.prompt)

            if self.prompt != original_prompt:
                print(f"IP-Adapter prompt was truncated for generation to fit token limit")
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
                scheduler_class = scheduler_config['class']
                scheduler_kwargs = scheduler_config.get('kwargs', {})

                # Create and set scheduler
                scheduler = scheduler_class.from_config(self.model.scheduler.config, **scheduler_kwargs)
                self.model.scheduler = scheduler
                print(f"Using scheduler: {self.params.scheduler}")

            # Enable VAE tiling for large images
            enable_vae_tiling = False
            if self.params.vae_tiling or self.params.width >= 1024 or self.params.height >= 1024:
                enable_vae_tiling = True
                print(f"Enabling VAE tiling for {self.params.width}x{self.params.height} image")

            if enable_vae_tiling:
                try:
                    self.model.enable_vae_tiling()
                    print("VAE tiling enabled for IP-Adapter image")
                except Exception as e:
                    print(f"VAE tiling not available: {e}")

            # Check for interruption before starting inference
            if self.isInterruptionRequested():
                print("IP-Adapter image generation cancelled before inference")
                return

            print("Starting IP-Adapter inference...")

            # Prepare IP-Adapter parameters
            ip_adapter_scales = []
            if self.ip_adapters:
                # Create a list of scales from all loaded IP-Adapters
                # The order of items in the dictionary is preserved (Python 3.7+)
                for adapter_name, (adapter_path, scale) in self.ip_adapters.items():
                    ip_adapter_scales.append(scale)
                print(f"Using IP-Adapter scales: {ip_adapter_scales}")

            # Generate image with IP-Adapter conditioning
            result = self.model(
                prompt=self.prompt,
                negative_prompt=self.params.negative_prompt,
                ip_adapter_image=self.reference_image,
                num_inference_steps=self.params.steps,
                guidance_scale=self.params.guidance_scale,
                width=self.params.width,
                height=self.params.height,
                generator=generator,
                ip_adapter_scale=ip_adapter_scales,  # Apply the list of scales
            )

            # Check for interruption after inference
            if self.isInterruptionRequested():
                print("IP-Adapter image generation cancelled after inference")
                return

            print("IP-Adapter inference completed, processing result...")
            image = result.images[0]
            print(f"Generated IP-Adapter image: {image.size} pixels")
            self.finished.emit(image)

        except Exception as e:
            print(f"IP-Adapter image generation failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.error.emit(f"IP-Adapter image generation failed: {str(e)}")
        finally:
            # Restore the original safety checker if it was temporarily disabled
            if original_safety_checker is not None:
                self.model.safety_checker = original_safety_checker
                print("🔍 DEBUG: NSFW filter restored - safety checker re-enabled")

    def _get_scheduler_config(self, scheduler_name: str) -> dict:
        """Get scheduler configuration for the specified scheduler."""
        try:
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

    def _inject_lora_trigger_words(self, prompt: str) -> str:
        """Inject LoRA trigger words into the prompt."""
        # For now, return the prompt unchanged since trigger word injection
        # requires access to LoRA metadata which isn't available here
        # TODO: Implement proper trigger word injection when LoRA metadata is accessible
        return prompt

    def _truncate_prompt(self, prompt: str, max_tokens: int = 75) -> str:
        """Truncate prompt to fit within token limit."""
        if not prompt:
            return prompt

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


class ImageGenerationService:
    """Service for managing image generation operations."""

    MODELS_DIR = "models"

    def __init__(self):
        self.model = None  # Can be StableDiffusionPipeline or StableDiffusionXLPipeline
        self._current_model_path: str = None
        self._loaded_loras = {}  # Dict of loaded LoRA adapters: {name: (path, scaling, adapter_name)}
        self._loaded_ip_adapters = {}  # Dict of loaded IP-Adapters: {name: (path, scale)}
        self._current_ip_adapter_image = None  # Current reference image for IP-Adapter
        self._original_model_state = None  # Store original model state before LoRA loading
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
            StableDiffusionPipeline, _, _ = _get_diffusers_imports()
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

    def load_model_async(self, model_path: str, quantization: str = "None", use_xformers: bool = True, cpu_offload: bool = False) -> ModelLoader:
        """Load a model asynchronously with progress reporting."""
        return ModelLoader(model_path, quantization, use_xformers, cpu_offload)

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
                StableDiffusionPipeline, StableDiffusionXLPipeline, _ = _get_diffusers_imports()
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
                    StableDiffusionPipeline, _, _ = _get_diffusers_imports()
                    self.model = StableDiffusionPipeline.from_pretrained(
                        local_path,
                        torch_dtype=torch.float16,
                        local_files_only=True
                    )
                else:
                    # Download and cache locally
                    local_path = self._download_model_locally("CompVis/stable-diffusion-v1-4", default_model_name)

                    # Reload from local cache to ensure consistency
                    StableDiffusionPipeline, _, _ = _get_diffusers_imports()
                    self.model = StableDiffusionPipeline.from_pretrained(
                        local_path,
                        torch_dtype=torch.float16,
                        local_files_only=True
                    )

            # Move to GPU if available first
            if torch.cuda.is_available():
                self.model.to("cuda")

            # Enable optimizations AFTER moving to GPU and AFTER any adapters are loaded
            # Note: Attention slicing will be enabled after IP-Adapters are loaded to avoid conflicts

            # Store the current model path
            self._current_model_path = model_path or local_path

            return True

        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def load_lora(self, lora_path: str, lora_name: str, scaling: float = 1.0, save_original_state: bool = True) -> bool:
        """Load a LoRA adapter into the current model."""
        try:
            if not self.model:
                raise RuntimeError("No base model loaded. Load a model first before applying LoRA.")

            print(f"Loading LoRA adapter: {lora_name} from {lora_path} with scaling {scaling}")

            # Check if LoRA file exists
            if not os.path.exists(lora_path):
                raise FileNotFoundError(f"LoRA file not found: {lora_path}")

            # Create a valid adapter name from the LoRA name
            # Adapter names must be valid Python identifiers (no dots, no special chars)
            import re
            adapter_name = re.sub(r'[^\w]', '_', lora_name)  # Replace non-word chars with underscores
            if not adapter_name or not adapter_name[0].isalpha():
                adapter_name = f"lora_{adapter_name}"  # Ensure it starts with a letter

            print(f"Using adapter name: '{adapter_name}' for LoRA '{lora_name}'")

            # Save original model state before loading first LoRA (only if not already saved)
            if self._original_model_state is None:
                print("Saving original model state before first LoRA loading")
                try:
                    # Try to save the model's state_dict before any LoRA modifications
                    if hasattr(self.model, 'state_dict') and callable(getattr(self.model, 'state_dict')):
                        self._original_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                        print(f"Saved original model state with {len(self._original_model_state)} parameters")
                    else:
                        print("Warning: Model does not have state_dict method - skipping original state saving")
                        print("LoRA unloading may not work properly without original state")
                except Exception as e:
                    print(f"Warning: Could not save original model state: {e}")
                    print("LoRA unloading may not work properly without original state")

            # Get LoRA imports
            LoraConfig, PeftModel = _get_lora_imports()

            # Load LoRA configuration and weights
            # For diffusers-compatible LoRA, we use the load_lora_weights method
            if hasattr(self.model, 'load_lora_weights'):
                # Use the built-in diffusers LoRA loading method
                try:
                    self.model.load_lora_weights(lora_path, adapter_name=adapter_name)
                    print(f"LoRA '{lora_name}' loaded using diffusers method")
                except IndexError as index_error:
                    print(f"❌ LoRA file format error: {index_error}")
                    print(f"   This LoRA file appears to be corrupted or incompatible with the current diffusers version")
                    print(f"   Try converting the LoRA to a different format or using a different LoRA file")
                    raise RuntimeError(f"LoRA file format incompatible: {index_error}")
                except Exception as diffusers_error:
                    print(f"❌ Diffusers LoRA loading failed: {diffusers_error}")
                    print(f"   This may be due to LoRA format incompatibility or corruption")
                    raise RuntimeError(f"Failed to load LoRA with diffusers: {diffusers_error}")
            else:
                # Fallback to PEFT method if available
                try:
                    # Create LoRA config
                    lora_config = LoraConfig(
                        r=16,  # Default rank
                        lora_alpha=scaling * 16,  # Scale alpha with scaling factor
                        target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # Attention layers
                        lora_dropout=0.0,
                        bias="none",
                    )

                    # Load LoRA model
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        lora_path,
                        config=lora_config,
                        adapter_name=adapter_name
                    )
                    print(f"LoRA '{lora_name}' loaded using PEFT method")
                except Exception as peft_error:
                    print(f"PEFT LoRA loading failed: {peft_error}")
                    raise RuntimeError(f"Failed to load LoRA using available methods: {peft_error}")

            # Store loaded LoRA info using the original name as key, but adapter_name for internal use
            self._loaded_loras[lora_name] = (lora_path, scaling, adapter_name)

            # Set the LoRA adapter as active if it's the only one
            if len(self._loaded_loras) == 1:
                self.model.set_adapters([adapter_name])

            # Debug: Show current active adapters
            if hasattr(self.model, 'active_adapters'):
                print(f"🔍 DEBUG: Active LoRA adapters after loading '{lora_name}': {self.model.active_adapters}")
            elif hasattr(self.model, '_active_adapters'):
                print(f"🔍 DEBUG: Active LoRA adapters after loading '{lora_name}': {self.model._active_adapters}")
            else:
                print(f"🔍 DEBUG: LoRA '{lora_name}' loaded but cannot determine active adapters")

            print(f"✅ LoRA '{lora_name}' loaded successfully with scaling {scaling}")
            return True

        except Exception as e:
            print(f"❌ Failed to load LoRA '{lora_name}': {str(e)}")
            return False

    def unload_lora(self, lora_name: str) -> bool:
        """Unload a specific LoRA adapter by restoring original model state."""
        try:
            if lora_name not in self._loaded_loras:
                print(f"LoRA '{lora_name}' is not currently loaded")
                return False

            if not self.model:
                print("No model loaded")
                return False

            print(f"Unloading LoRA adapter: {lora_name}")

            # Remove from loaded LoRAs dict first
            del self._loaded_loras[lora_name]

            # If we have an original model state saved, restore it completely
            if self._original_model_state is not None:
                print("Restoring original model state to remove all LoRA modifications")
                try:
                    # Restore the original model state
                    self.model.load_state_dict(self._original_model_state, strict=False)
                    print(f"✅ Restored original model state with {len(self._original_model_state)} parameters")

                    # Clear any LoRA-related attributes that might remain
                    if hasattr(self.model, '_loras'):
                        self.model._loras.clear()
                    if hasattr(self.model, '_active_adapters'):
                        self.model._active_adapters.clear()

                    # Re-load remaining LoRAs if any
                    if self._loaded_loras:
                        print(f"Re-loading {len(self._loaded_loras)} remaining LoRAs")
                        loaded_adapter_names = []
                        for name, (path, scaling, _) in self._loaded_loras.items():
                            if self.load_lora(path, name, scaling, save_original_state=False):
                                # Get the adapter name that was stored
                                _, _, adapter_name = self._loaded_loras[name]
                                loaded_adapter_names.append(adapter_name)
                            else:
                                print(f"Warning: Failed to re-load LoRA '{name}' after restoration")

                        # Set active adapters
                        if loaded_adapter_names and hasattr(self.model, 'set_adapters'):
                            self.model.set_adapters(loaded_adapter_names)
                            print(f"Active LoRA adapters after restoration: {loaded_adapter_names}")
                            # Debug: Confirm active adapters
                            if hasattr(self.model, 'active_adapters'):
                                print(f"🔍 DEBUG: Confirmed active adapters: {self.model.active_adapters}")
                            elif hasattr(self.model, '_active_adapters'):
                                print(f"🔍 DEBUG: Confirmed active adapters: {self.model._active_adapters}")
                    else:
                        print("No remaining LoRAs to load - model restored to original state")

                except Exception as e:
                    print(f"Warning: Could not restore original model state: {e}")
                    print("Attempting alternative unloading method...")

                    # Fallback: Try the old method if restoration fails
                    if hasattr(self.model, 'unload_lora_weights'):
                        try:
                            self.model.unload_lora_weights()
                            print("Used fallback unloading method")
                        except Exception as fallback_error:
                            print(f"Fallback unloading also failed: {fallback_error}")
                            return False
            else:
                print("No original model state saved - using standard unloading")
                # Fallback to standard unloading if no original state was saved
                if hasattr(self.model, 'unload_lora_weights'):
                    try:
                        self.model.unload_lora_weights()
                        print("Used standard diffusers unloading")
                    except Exception as e:
                        print(f"Standard unloading failed: {e}")
                        return False

            # After unloading, ensure no adapters are active
            if hasattr(self.model, 'set_adapters'):
                try:
                    self.model.set_adapters([])  # Clear all active adapters
                    print(f"Cleared all active adapters after unloading '{lora_name}'")
                except Exception as e:
                    print(f"Warning: Could not clear active adapters: {e}")

            print(f"✅ LoRA '{lora_name}' unloaded successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to unload LoRA '{lora_name}': {str(e)}")
            return False

    def unload_all_loras(self) -> bool:
        """Unload all LoRA adapters."""
        try:
            if not self._loaded_loras:
                return True

            print(f"Unloading all {len(self._loaded_loras)} LoRA adapters")

            # Clear the loaded LoRAs dict first
            self._loaded_loras.clear()

            # If we have an original model state saved, restore it completely
            if self._original_model_state is not None:
                print("Restoring original model state to remove all LoRA modifications")
                try:
                    # Restore the original model state
                    self.model.load_state_dict(self._original_model_state, strict=False)
                    print(f"✅ Restored original model state with {len(self._original_model_state)} parameters")

                    # Clear any LoRA-related attributes that might remain
                    if hasattr(self.model, '_loras'):
                        self.model._loras.clear()
                    if hasattr(self.model, '_active_adapters'):
                        self.model._active_adapters.clear()

                    print("No LoRAs remaining - model restored to original state")

                except Exception as e:
                    print(f"Warning: Could not restore original model state: {e}")
                    print("Attempting alternative unloading method...")

                    # Fallback: Try the old method if restoration fails
                    if hasattr(self.model, 'unload_lora_weights'):
                        try:
                            self.model.unload_lora_weights()
                            print("Used fallback unloading method")
                        except Exception as fallback_error:
                            print(f"Fallback unloading also failed: {fallback_error}")
                            return False
            else:
                print("No original model state saved - using standard unloading")
                # Fallback to standard unloading if no original state was saved
                if hasattr(self.model, 'unload_lora_weights'):
                    try:
                        self.model.unload_lora_weights()
                        print("Used standard diffusers unloading")
                    except Exception as e:
                        print(f"Standard unloading failed: {e}")
                        return False

            print("✅ All LoRA adapters unloaded successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to unload all LoRAs: {str(e)}")
            return False

    def get_loaded_loras(self) -> dict:
        """Get information about currently loaded LoRA adapters."""
        return self._loaded_loras.copy()

    def set_lora_scaling(self, lora_name: str, scaling: float) -> bool:
        """Set the scaling factor for a loaded LoRA adapter."""
        try:
            if lora_name not in self._loaded_loras:
                print(f"LoRA '{lora_name}' is not currently loaded")
                return False

            if not self.model:
                print("No model loaded")
                return False

            print(f"Setting LoRA '{lora_name}' scaling to {scaling}")

            # Update scaling in our tracking dict
            path, _, adapter_name = self._loaded_loras[lora_name]
            self._loaded_loras[lora_name] = (path, scaling, adapter_name)

            # If using PEFT, update the scaling
            if hasattr(self.model, 'set_adapter_lora_scaling'):
                try:
                    self.model.set_adapter_lora_scaling(adapter_name, scaling)
                    print(f"LoRA scaling updated in PEFT model")
                except Exception as e:
                    print(f"Warning: Could not update PEFT scaling: {e}")

            print(f"✅ LoRA '{lora_name}' scaling set to {scaling}")
            return True

        except Exception as e:
            print(f"❌ Failed to set LoRA scaling: {str(e)}")
            return False

    def apply_lora_adapters(self, lora_configs: list) -> bool:
        """Apply multiple LoRA adapters with their configurations."""
        try:
            if not self.model:
                raise RuntimeError("No base model loaded")

            if not lora_configs:
                # No LoRAs to apply, unload all
                return self.unload_all_loras()

            print(f"Applying {len(lora_configs)} LoRA adapters")

            # Unload all existing LoRAs first
            self.unload_all_loras()

            # Load new LoRAs
            loaded_adapter_names = []  # Store the actual adapter names used by diffusers
            for config in lora_configs:
                lora_name = config.get('name')
                lora_path = config.get('path')
                scaling = config.get('scaling', 1.0)

                # Validate and clamp scaling to reasonable range
                scaling = max(-5.0, min(5.0, scaling))  # Clamp to -5.0 to +5.0 range

                if not lora_name or not lora_path:
                    print(f"Skipping invalid LoRA config: {config}")
                    continue

                if self.load_lora(lora_path, lora_name, scaling):
                    # Use the actual adapter name that was stored (sanitized version)
                    _, _, adapter_name = self._loaded_loras[lora_name]
                    loaded_adapter_names.append(adapter_name)
                else:
                    print(f"Failed to load LoRA: {lora_name}")

            # Set active adapters using the correct sanitized adapter names
            if loaded_adapter_names and hasattr(self.model, 'set_adapters'):
                self.model.set_adapters(loaded_adapter_names)
                print(f"Active LoRA adapters: {loaded_adapter_names}")

            print(f"✅ Applied {len(loaded_adapter_names)} LoRA adapters successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to apply LoRA adapters: {str(e)}")
            return False

    def generate_image(self, prompt: str, params: GenerationParams) -> ImageGenerator:
        """Create image generator thread with LoRA and IP-Adapter support."""
        print(f"🔍 DEBUG: generate_image called with prompt: '{prompt[:50]}...'")
        print(f"🔍 DEBUG: Service has {len(self._loaded_loras)} loaded LoRAs: {list(self._loaded_loras.keys())}")

        if not self.model:
            raise RuntimeError("Model not loaded")

        # Apply LoRA adapters from generation parameters if specified
        if hasattr(params, 'lora_adapters') and params.lora_adapters:
            print(f"🔍 DEBUG: Applying LoRA adapters from params: {params.lora_adapters}")
            self.apply_lora_adapters(params.lora_adapters)
        # Ensure any already-loaded LoRAs are active if no specific LoRAs requested
        elif self._loaded_loras:
            print(f"🔍 DEBUG: Activating existing LoRAs for generation: {list(self._loaded_loras.keys())}")
            loaded_adapter_names = [adapter_name for _, _, adapter_name in self._loaded_loras.values()]
            if loaded_adapter_names and hasattr(self.model, 'set_adapters'):
                self.model.set_adapters(loaded_adapter_names)
                print(f"🔍 DEBUG: Activated existing LoRA adapters: {loaded_adapter_names}")
            else:
                print(f"🔍 DEBUG: No adapter names found or set_adapters not available")
        else:
            print(f"🔍 DEBUG: No LoRAs to activate - none loaded in service")

        # Check for IP-Adapter parameters
        if hasattr(params, 'ip_adapters') and params.ip_adapters and hasattr(params, 'reference_image_path') and params.reference_image_path:
            print("IP-Adapter parameters found, preparing for IP-Adapter generation.")

            # Load reference image
            from PIL import Image
            try:
                reference_image = Image.open(params.reference_image_path).convert("RGB")
            except Exception as e:
                raise RuntimeError(f"Failed to load reference image: {e}")

            # Unload all previously loaded IP-Adapters to ensure a clean state
            self.unload_all_ip_adapters()

            # Apply mode-specific conditioning adjustments
            mode_adjusted_adapters = self._apply_ip_adapter_mode(params.ip_adapters, params.ip_adapter_mode)

            # Load the IP-Adapters specified in the parameters with mode adjustments
            for adapter_config in mode_adjusted_adapters:
                self.load_ip_adapter(adapter_config['path'], adapter_config['name'], adapter_config['scale'])

            # Use the dedicated IP-Adapter generator
            return IPAdapterImageGenerator(self.model, prompt, params, reference_image, self._loaded_ip_adapters)

        # Fallback to regular generation
        return ImageGenerator(self.model, prompt, params)

    def load_ip_adapter(self, adapter_path: str, adapter_name: str, scale: float = 1.0) -> bool:
        """Load an IP-Adapter into the current model."""
        try:
            if not self.model:
                raise RuntimeError("No base model loaded. Load a model first before applying IP-Adapter.")

            print(f"Loading IP-Adapter: {adapter_name} from {adapter_path} with scale {scale}")

            # Check if IP-Adapter file exists
            if not os.path.exists(adapter_path):
                raise FileNotFoundError(f"IP-Adapter file not found: {adapter_path}")

            # Load IP-Adapter from single safetensors file
            try:
                from safetensors import safe_open

                # Load the state dict from safetensors file
                state_dict = {"image_proj": {}, "ip_adapter": {}}
                with safe_open(adapter_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("image_proj."):
                            state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                        elif key.startswith("ip_adapter."):
                            state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)

                # Verify we have the required keys
                if not state_dict["image_proj"] or not state_dict["ip_adapter"]:
                    raise ValueError(f"IP-Adapter file {adapter_path} does not contain required 'image_proj' and 'ip_adapter' keys")

                # Temporarily disable attention slicing if enabled, as it can interfere with IP-Adapter loading
                attention_slicing_was_enabled = hasattr(self.model, '_attention_slicing_enabled') and self.model._attention_slicing_enabled
                if attention_slicing_was_enabled:
                    print("Temporarily disabling attention slicing for IP-Adapter loading")
                    self.model.disable_attention_slicing()

                # Load IP-Adapter using state dict
                # Note: load_ip_adapter expects (pretrained_model_name_or_path_or_dict, subfolder, weight_name, ...)
                # When passing a state_dict, we need image_encoder_folder=None to avoid loading attempts
                self.model.load_ip_adapter(state_dict, None, None, image_encoder_folder=None)
                print(f"IP-Adapter '{adapter_name}' loaded successfully from state dict")

                # Load image encoder manually if it's missing (required for IP-Adapter generation)
                if hasattr(self.model, 'image_encoder') and getattr(self.model, 'image_encoder', None) is None:
                    print("Loading CLIP image encoder for IP-Adapter generation...")
                    try:
                        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

                        # Load CLIP Vision model (CLIP-ViT-H/14 for 1024-dim embeddings to match IP-Adapter expectations)
                        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                            torch_dtype=self.model.dtype,
                            low_cpu_mem_usage=True
                        ).to(self.model.device)

                        # Load feature extractor
                        feature_extractor = CLIPImageProcessor(size=224, crop_size=224)

                        # Register components to the pipeline
                        self.model.register_modules(
                            image_encoder=image_encoder,
                            feature_extractor=feature_extractor
                        )
                        print("CLIP image encoder loaded successfully")
                    except Exception as e:
                        print(f"Warning: Failed to load CLIP image encoder: {e}")
                        print("IP-Adapter generation may fail without image encoder")

                # Re-enable attention slicing if it was previously enabled
                if attention_slicing_was_enabled:
                    print("Re-enabling attention slicing after IP-Adapter loading")
                    self.model.enable_attention_slicing()

            except Exception as e:
                print(f"Failed to load IP-Adapter weights: {e}")
                return False

            # Store loaded IP-Adapter info
            self._loaded_ip_adapters[adapter_name] = (adapter_path, scale)

            print(f"SUCCESS: IP-Adapter '{adapter_name}' loaded successfully with scale {scale}")
            return True

        except Exception as e:
            print(f"FAILED: Failed to load IP-Adapter '{adapter_name}': {str(e)}")
            return False

    def unload_ip_adapter(self, adapter_name: str) -> bool:
        """Unload a specific IP-Adapter."""
        try:
            if adapter_name not in self._loaded_ip_adapters:
                print(f"IP-Adapter '{adapter_name}' is not currently loaded")
                return False

            if not self.model:
                print("No model loaded")
                return False

            print(f"Unloading IP-Adapter: {adapter_name}")

            # Remove from loaded IP-Adapters dict
            del self._loaded_ip_adapters[adapter_name]

            # If no IP-Adapters remain, we could potentially convert back to regular pipeline
            # But for now, we'll keep the IP-Adapter pipeline

            print(f"✅ IP-Adapter '{adapter_name}' unloaded successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to unload IP-Adapter '{adapter_name}': {str(e)}")
            return False

    def unload_all_ip_adapters(self) -> bool:
        """Unload all IP-Adapters."""
        try:
            if not self._loaded_ip_adapters:
                return True

            print(f"Unloading all {len(self._loaded_ip_adapters)} IP-Adapters")

            # Unload all IP-Adapters
            for adapter_name in list(self._loaded_ip_adapters.keys()):
                self.unload_ip_adapter(adapter_name)

            # Clear the loaded IP-Adapters dict
            self._loaded_ip_adapters.clear()

            print("✅ All IP-Adapters unloaded successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to unload all IP-Adapters: {str(e)}")
            return False

    def get_loaded_ip_adapters(self) -> dict:
        """Get information about currently loaded IP-Adapters."""
        return self._loaded_ip_adapters.copy()

    def set_ip_adapter_scale(self, adapter_name: str, scale: float) -> bool:
        """Set the scaling factor for a loaded IP-Adapter."""
        try:
            if adapter_name not in self._loaded_ip_adapters:
                print(f"IP-Adapter '{adapter_name}' is not currently loaded")
                return False

            print(f"Setting IP-Adapter '{adapter_name}' scale to {scale}")

            # Update scale in our tracking dict
            path, _ = self._loaded_ip_adapters[adapter_name]
            self._loaded_ip_adapters[adapter_name] = (path, scale)

            print(f"✅ IP-Adapter '{adapter_name}' scale set to {scale}")
            return True

        except Exception as e:
            print(f"❌ Failed to set IP-Adapter scale: {str(e)}")
            return False

    def set_reference_image(self, image_path: str = None, pil_image: Image.Image = None) -> bool:
        """Set the reference image for IP-Adapter conditioning."""
        try:
            if image_path:
                # Load image from file path
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Reference image not found: {image_path}")

                self._current_ip_adapter_image = Image.open(image_path).convert("RGB")
                print(f"Reference image loaded from: {image_path}")
            elif pil_image:
                # Use provided PIL image
                self._current_ip_adapter_image = pil_image.convert("RGB") if pil_image.mode != "RGB" else pil_image
                print("Reference image set from PIL image")
            else:
                # Clear reference image
                self._current_ip_adapter_image = None
                print("Reference image cleared")

            return True

        except Exception as e:
            print(f"❌ Failed to set reference image: {str(e)}")
            return False

    def _apply_ip_adapter_mode(self, ip_adapters: list, mode: str) -> list:
        """Apply mode-specific adjustments to IP-Adapter configurations.

        Args:
            ip_adapters: List of IP-Adapter configs with {'name', 'path', 'scale'}
            mode: Conditioning mode - 'style', 'composition', 'style_and_composition'

        Returns:
            Modified list of IP-Adapter configs with adjusted scales
        """
        if not ip_adapters:
            return ip_adapters

        print(f"Applying IP-Adapter mode: {mode}")

        # Create a copy to avoid modifying the original
        adjusted_adapters = []

        for adapter in ip_adapters:
            adapter_copy = adapter.copy()

            # Apply mode-specific scale adjustments
            base_scale = adapter_copy.get('scale', 1.0)

            if mode == 'style':
                # Style mode: Emphasize artistic elements, stronger style influence
                # Use higher scale for better style transfer
                adapter_copy['scale'] = base_scale * 1.5
                print(f"  Style mode: {adapter_copy['name']} scale {base_scale} -> {adapter_copy['scale']}")

            elif mode == 'composition':
                # Composition mode: Emphasize layout/positioning, weaker composition influence
                # Use lower scale for more controlled composition transfer
                adapter_copy['scale'] = base_scale * 0.5
                print(f"  Composition mode: {adapter_copy['name']} scale {base_scale} -> {adapter_copy['scale']}")

            elif mode == 'style_and_composition':
                # Balanced mode: Both style and composition influence
                # Use default scale for balanced conditioning
                adapter_copy['scale'] = base_scale
                print(f"  Style+Composition mode: {adapter_copy['name']} scale {base_scale} (unchanged)")

            elif mode == 'prompt_priority':
                # Prompt Priority mode: Weak image conditioning, strong prompt influence
                # Use very low scale to minimize IP-Adapter influence
                adapter_copy['scale'] = base_scale * 0.1
                print(f"  Prompt Priority mode: {adapter_copy['name']} scale {base_scale} -> {adapter_copy['scale']}")

            else:
                # Default/fallback: use original scale
                print(f"  Unknown mode '{mode}', using default scale: {adapter_copy['name']} scale {base_scale}")

            adjusted_adapters.append(adapter_copy)

        return adjusted_adapters

    def get_reference_image(self) -> Image.Image:
        """Get the current reference image."""
        return self._current_ip_adapter_image

    def unload_model(self):
        """Unload the model and all adapters to free memory."""
        if self.model:
            # Unload all LoRAs first
            self.unload_all_loras()

            # Unload all IP-Adapters
            self.unload_all_ip_adapters()

            # Clear reference image
            self._current_ip_adapter_image = None

            # Clear original model state
            self._original_model_state = None

            # Unload the base model
            del self.model
            self.model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
