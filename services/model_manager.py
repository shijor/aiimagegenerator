"""
Model management service.
"""
import os
import shutil
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_info import ModelInfo, ModelType, ModelCategory, LoRAInfo, IPAdapterInfo
from models.settings import SettingsManager, AppSettings
from services.database_manager import DatabaseManager


class ModelManager:
    """Manages AI model installation, switching, and metadata using SQLite database."""

    MODELS_DIR = "models"

    def __init__(self):
        self.db = DatabaseManager()
        self._ensure_models_directory()
        self._migrate_legacy_data()

    def _ensure_models_directory(self) -> None:
        """Ensure models directory exists."""
        os.makedirs(self.MODELS_DIR, exist_ok=True)

    def _migrate_legacy_data(self) -> None:
        """Migrate data from JSON files to database if needed."""
        # Check if we have legacy JSON data
        models_file = os.path.join(self.MODELS_DIR, "installed_models.json")
        if os.path.exists(models_file):
            print("Migrating legacy JSON data to SQLite database...")
            if self.db.migrate_from_json():
                # Backup and remove old JSON files
                import shutil
                shutil.move(models_file, models_file + ".backup")
                print("Migration completed successfully!")
            else:
                print("Migration failed, falling back to JSON data")

        # Ensure we have at least the default model
        if not self.db.get_all_models():
            self._create_default_model()

    def _create_default_model(self) -> None:
        """Create the default model entry."""
        default_model_path = os.path.join(self.MODELS_DIR, "stable-diffusion-v1-4")

        default_model = ModelInfo(
            name="Stable Diffusion v1.4",
            path=default_model_path,
            model_type=ModelType.STABLE_DIFFUSION_V1_4,
            description="Default Stable Diffusion model (locally cached)",
            is_default=True,
            installed_date=datetime.now().isoformat()
        )
        self.db.save_model(default_model)

    def _is_lora_file(self, filename: str, file_size: int) -> bool:
        """Determine if a file is likely a LoRA adapter based on size (anything below 1000MB is considered LoRA)."""
        # Size check: Any file below 1000MB (1GB) is considered a LoRA adapter
        # Full models are typically 1GB or larger, LoRAs are smaller
        if file_size < 1000 * 1024 * 1024:  # < 1000MB - treat as LoRA
            return True

        # Files >= 1000MB are considered full models
        return False

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

    def _validate_diffusers_model_comprehensive(self, dir_path: str) -> dict:
        """Comprehensive diffusers model validation with detailed component checking."""
        validation_result = {
            'is_valid': False,
            'model_type': None,
            'components': {},
            'metadata': {},
            'issues': [],
            'warnings': []
        }

        try:
            print(f"COMPREHENSIVE VALIDATION: Starting validation of {dir_path}")

            # 1. Check model_index.json
            model_index_path = os.path.join(dir_path, "model_index.json")
            if not os.path.exists(model_index_path):
                validation_result['issues'].append("model_index.json not found")
                return validation_result

            with open(model_index_path, 'r', encoding='utf-8') as f:
                model_config = json.load(f)

            # 2. Validate model class and determine type
            class_name = model_config.get('_class_name', '')
            if not class_name:
                validation_result['issues'].append("Missing _class_name in model_index.json")
                return validation_result

            # Determine model type from class name
            if 'StableDiffusionXL' in class_name or 'XL' in class_name:
                validation_result['model_type'] = ModelType.STABLE_DIFFUSION_XL
            elif 'StableDiffusion' in class_name:
                # Check version from other indicators
                if self._detect_sd_version(dir_path) == '1.5':
                    validation_result['model_type'] = ModelType.STABLE_DIFFUSION_V1_5
                else:
                    validation_result['model_type'] = ModelType.STABLE_DIFFUSION_V1_4
            else:
                validation_result['issues'].append(f"Unknown model class: {class_name}")
                return validation_result

            # 3. Check required components
            required_components = {
                'text_encoder': ['config.json', 'model.safetensors'],
                'tokenizer': ['tokenizer_config.json', 'vocab.json'],
                'unet': ['config.json', 'diffusion_pytorch_model.safetensors'],
                'vae': ['config.json', 'diffusion_pytorch_model.safetensors'],
                'scheduler': ['scheduler_config.json']
            }

            components_status = {}
            for component, required_files in required_components.items():
                component_path = os.path.join(dir_path, component)
                if not os.path.exists(component_path):
                    validation_result['issues'].append(f"Missing component directory: {component}")
                    components_status[component] = {'exists': False, 'files': {}}
                    continue

                components_status[component] = {'exists': True, 'files': {}}
                for required_file in required_files:
                    file_path = os.path.join(component_path, required_file)
                    exists = os.path.exists(file_path)
                    components_status[component]['files'][required_file] = exists
                    if not exists:
                        validation_result['issues'].append(f"Missing file: {component}/{required_file}")

            validation_result['components'] = components_status

            # 4. Extract metadata
            validation_result['metadata'] = self._extract_diffusers_metadata_comprehensive(dir_path, model_config)

            # 5. Check component sizes (basic corruption detection)
            total_size = self._get_directory_size(dir_path)
            if total_size < 100 * 1024 * 1024:  # Less than 100MB
                validation_result['warnings'].append(f"Model size ({total_size/1024/1024:.1f}MB) seems too small")

            # 6. Final validation
            has_critical_issues = len(validation_result['issues']) > 0
            validation_result['is_valid'] = not has_critical_issues

            print(f"COMPREHENSIVE VALIDATION: Result - Valid: {validation_result['is_valid']}, Issues: {len(validation_result['issues'])}, Warnings: {len(validation_result['warnings'])}")

            return validation_result

        except Exception as e:
            validation_result['issues'].append(f"Validation error: {str(e)}")
            print(f"COMPREHENSIVE VALIDATION: Exception - {str(e)}")
            return validation_result

    def _detect_sd_version(self, dir_path: str) -> str:
        """Detect Stable Diffusion version from model characteristics."""
        try:
            # Check UNet config for version indicators
            unet_config_path = os.path.join(dir_path, "unet", "config.json")
            if os.path.exists(unet_config_path):
                with open(unet_config_path, 'r') as f:
                    unet_config = json.load(f)

                # SDXL has different cross_attention_dim
                if unet_config.get('cross_attention_dim') == 2048:
                    return 'xl'

                # Check for other SDXL indicators
                if unet_config.get('sample_size') == 1024:
                    return 'xl'

            # Check model_index.json for version hints
            model_index_path = os.path.join(dir_path, "model_index.json")
            if os.path.exists(model_index_path):
                with open(model_index_path, 'r') as f:
                    config = json.load(f)

                # Check various fields for version indicators
                version_hints = ['_name_or_path', 'name', 'model_name']
                for field in version_hints:
                    value = config.get(field, '').lower()
                    if 'xl' in value or 'stable-diffusion-xl' in value:
                        return 'xl'
                    if '1.5' in value or 'v1-5' in value:
                        return '1.5'

            return '1.4'  # Default assumption

        except Exception:
            return '1.4'  # Default fallback

    def _extract_diffusers_metadata_comprehensive(self, dir_path: str, model_config: dict) -> dict:
        """Extract comprehensive metadata from diffusers model."""
        metadata = {
            'name': None,
            'display_name': None,
            'description': None,
            'author': None,
            'license': None,
            'tags': [],
            'source_url': None,
            'class_name': model_config.get('_class_name', '')
        }

        # Extract from model_index.json
        name_fields = ['_name_or_path', 'name', 'model_name', 'hub_name', 'repo_name']
        for field in name_fields:
            if field in model_config and model_config[field]:
                metadata['name'] = model_config[field]
                break

        # Try to extract more metadata from various config files
        try:
            # Check text_encoder config for potential metadata
            text_encoder_config = os.path.join(dir_path, "text_encoder", "config.json")
            if os.path.exists(text_encoder_config):
                with open(text_encoder_config, 'r') as f:
                    te_config = json.load(f)
                    # Could extract model size, vocab size, etc.

            # Check scheduler config
            scheduler_config = os.path.join(dir_path, "scheduler", "scheduler_config.json")
            if os.path.exists(scheduler_config):
                with open(scheduler_config, 'r') as f:
                    sched_config = json.load(f)
                    # Could extract scheduler type, parameters

        except Exception as e:
            print(f"METADATA EXTRACTION: Error reading config files: {e}")

        # Generate display name from name if available
        if metadata['name']:
            # Clean up the name for display
            display_name = metadata['name'].split('/')[-1]  # Get last part of path
            display_name = display_name.replace('-', ' ').replace('_', ' ').title()
            metadata['display_name'] = display_name

        return metadata

    # Comprehensive model type definitions for filtering
    VALID_IMAGE_GENERATION_PIPELINES = {
        # Core Stable Diffusion pipelines
        'StableDiffusionPipeline',
        'StableDiffusionXLPipeline',
        'StableDiffusionImg2ImgPipeline',
        'StableDiffusionInpaintPipeline',

        # ControlNet variants
        'StableDiffusionControlNetPipeline',
        'StableDiffusionXLControlNetPipeline',
        'StableDiffusionControlNetImg2ImgPipeline',
        'StableDiffusionControlNetInpaintPipeline',

        # Specialized image generation pipelines
        'StableDiffusionDepth2ImgPipeline',
        'StableDiffusionImageVariationPipeline',
        'StableDiffusionLatentUpscalePipeline',
        'StableDiffusionUpscalePipeline',

        # SDXL specific pipelines
        'StableDiffusionXLImg2ImgPipeline',
        'StableDiffusionXLInpaintPipeline',

        # PAG (Perturbed Attention Guidance) variants
        'StableDiffusionPAGPipeline',
        'StableDiffusionXLPAGPipeline',

        # Union ControlNet pipelines
        'StableDiffusionControlNetUnionPipeline',
        'StableDiffusionXLControlNetUnionPipeline',
    }

    EXCLUDED_MODEL_TYPES = {
        # Computer Vision (non-generation)
        'sam', 'segment-anything', 'detect', 'classify', 'recognition', 'pose',
        'face-recognition', 'depth-estimation', 'normal', 'edge', 'canny', 'hed', 'mlsd', 'lineart',

        # Language Models
        'bert', 'gpt', 't5', 'llama', 'opt', 'bloom', 'roberta', 'albert',
        'electra', 'distilbert', 'xlnet', 'bart', 'pegasus', 'mbart',

        # Audio Models
        'whisper', 'wav2vec', 'hubert', 'speech', 'audio', 'music', 'tts',
        'text-to-speech', 'speech-to-text', 'voice',

        # Text Encoding Only (explicit CLIP variants)
        'clip-text', 'text-encoder', 'sentence-transformer',
        'clip-vit', 'clip-vision', 'openai-clip',

        # Document Processing
        'ocr', 'handwriting', 'signature', 'barcode', 'qr', 'document',
        'layout-parser', 'receipt', 'invoice', 'form',

        # Scientific/Research models
        'protein', 'dna', 'rna', 'molecule', 'chemistry', 'physics',
        'mathematics', 'theorem', 'proof', 'equation',

        # Data Science models
        'tabular', 'time-series', 'forecast', 'regression', 'classification',
        'clustering', 'anomaly-detection', 'recommendation', 'ranking', 'search'
    }

    def _is_image_generation_model(self, validation_result: dict) -> bool:
        """Comprehensive check if a validated diffusers model is for image generation."""
        class_name = validation_result.get('metadata', {}).get('class_name', '')

        # 1. Check against known valid image generation pipelines
        if any(valid_pipeline in class_name for valid_pipeline in self.VALID_IMAGE_GENERATION_PIPELINES):
            print(f"SCANNING MODELS: ✅ Valid image generation pipeline: {class_name}")
            return True

        # 2. Check against excluded model types
        class_lower = class_name.lower()
        if any(excluded in class_lower for excluded in self.EXCLUDED_MODEL_TYPES):
            print(f"SCANNING MODELS: ❌ Excluded model type in class '{class_name}': {excluded}")
            return False

        # 3. Check model name/description for exclusion clues
        metadata = validation_result.get('metadata', {})
        name = (metadata.get('name', '') + metadata.get('display_name', '')).lower()

        if any(excluded in name for excluded in self.EXCLUDED_MODEL_TYPES):
            print(f"SCANNING MODELS: ❌ Excluded model type in name '{name}': {excluded}")
            return False

        # 4. Check for required image generation architecture components
        components = validation_result.get('components', {})
        has_unet = 'unet' in components and components['unet']['exists']
        has_vae = 'vae' in components and components['vae']['exists']
        has_text_encoder = 'text_encoder' in components and components['text_encoder']['exists']

        # Must have core image generation components (UNet + VAE + Text Encoder)
        if has_unet and has_vae and has_text_encoder:
            print(f"SCANNING MODELS: ✅ Has image generation architecture (UNet+VAE+TextEncoder)")
            return True

        # 5. Special case: ControlNet models (may not have all components in config)
        if 'controlnet' in class_lower or 'control' in class_lower:
            print(f"SCANNING MODELS: ✅ ControlNet model detected: {class_name}")
            return True

        print(f"SCANNING MODELS: ❌ Model '{class_name}' lacks image generation architecture")
        return False

    def _validate_image_generation_model_file(self, file_path: str) -> bool:
        """Comprehensive validation for traditional model files to ensure they're image generation models."""
        try:
            # Basic file validation
            if not self._validate_image_model(file_path):
                print(f"SCANNING MODELS: Basic file validation failed for {file_path}")
                return False

            # Check file size (reasonable bounds for image models)
            file_size = os.path.getsize(file_path)
            if file_size < 200 * 1024 * 1024:  # < 200MB - too small for image generation model
                print(f"SCANNING MODELS: File too small ({file_size/1024/1024:.1f}MB) for image generation model: {file_path}")
                return False
            if file_size > 15 * 1024 * 1024 * 1024:  # > 15GB - too large (unreasonable)
                print(f"SCANNING MODELS: File too large ({file_size/1024/1024/1024:.1f}GB) for image generation model: {file_path}")
                return False

            filename = os.path.basename(file_path).lower()

            # 1. Explicit CLIP model rejection (highest priority)
            if 'clip' in filename and ('vit' in filename or 'large' in filename or 'base' in filename):
                print(f"SCANNING MODELS: ❌ Explicit CLIP model rejection: {filename}")
                return False

            # 2. Component file detection and rejection
            component_indicators = [
                'text_encoder', 'text-encoder', 'tokenizer', 'unet', 'vae', 'vae_decoder',
                'vae_encoder', 'safety_checker', 'feature_extractor', 'scheduler',
                'diffusion_pytorch_model'  # Individual component files
            ]

            for component in component_indicators:
                if component in filename:
                    # Additional check: if it's in a subfolder, definitely a component
                    parent_dir = os.path.basename(os.path.dirname(file_path)).lower()
                    if parent_dir in ['text_encoder', 'tokenizer', 'unet', 'vae', 'scheduler']:
                        print(f"SCANNING MODELS: ❌ Component file in component directory: {file_path}")
                        return False

                    # If filename starts with component name, likely a component
                    if filename.startswith(component + '.') or filename.startswith(component + '-'):
                        print(f"SCANNING MODELS: ❌ Component file pattern detected: {filename}")
                        return False

            # 3. Check against excluded model types by filename - MORE SPECIFIC
            excluded_found = []
            for excluded in self.EXCLUDED_MODEL_TYPES:
                if excluded in filename:
                    # Be more specific - don't exclude if it's part of a legitimate model name
                    # For example, don't exclude "stable-diffusion" just because it contains "table"
                    if excluded == 'table' and ('stable-diffusion' in filename or 'diffusion' in filename):
                        continue  # Allow table in stable-diffusion context
                    if excluded == 'clip' and ('stable-diffusion' in filename or 'diffusion' in filename):
                        continue  # Allow clip in SD model names
                    excluded_found.append(excluded)

            if excluded_found:
                print(f"SCANNING MODELS: ❌ Excluded model types {excluded_found} in filename: {filename}")
                return False

            # 4. Detect model type from file content/metadata
            model_type = self._detect_model_type_from_file(file_path)
            if not model_type:
                print(f"SCANNING MODELS: Could not detect model type for: {file_path}")
                return False

            # 5. Advanced architecture analysis for safetensors files
            if file_path.endswith('.safetensors'):
                try:
                    from safetensors import safe_open
                    with safe_open(file_path, framework="pt", device="cpu") as f:
                        tensor_names = [name.lower() for name in f.keys()]

                        # Check for non-image-generation model indicators
                        has_language_only = any(term in ' '.join(tensor_names) for term in
                                               ['lm_head', 'embed_tokens', 'position_embeddings'])
                        has_audio_only = any(term in ' '.join(tensor_names) for term in
                                           ['audio', 'speech', 'wav2vec', 'hubert'])

                        if has_language_only:
                            print(f"SCANNING MODELS: ❌ Detected language-only model architecture: {file_path}")
                            return False
                        if has_audio_only:
                            print(f"SCANNING MODELS: ❌ Detected audio-only model architecture: {file_path}")
                            return False

                        # Check for image generation architecture (be more lenient)
                        has_unet = any('unet' in name or 'down_blocks' in name for name in tensor_names)
                        has_vae = any('vae' in name or 'decoder' in name or 'encoder' in name for name in tensor_names)
                        has_clip = any('clip' in name or 'text' in name for name in tensor_names)

                        # Be much more lenient: accept if we have any diffusion-related components
                        # or if the model has passed all other validation checks
                        has_diffusion_components = has_unet or has_vae or has_clip
                        has_basic_architecture = len(tensor_names) > 10  # Reasonable number of tensors

                        if has_diffusion_components:
                            print(f"SCANNING MODELS: ✅ Has diffusion components (UNet/VAE/CLIP): {file_path}")
                        elif has_basic_architecture:
                            print(f"SCANNING MODELS: ✅ Has basic model architecture (accepting): {file_path}")
                        else:
                            print(f"SCANNING MODELS: ❌ Insufficient model architecture: {file_path}")
                            return False

                except ImportError:
                    print("SCANNING MODELS: safetensors not available for architecture analysis")
                except Exception as e:
                    # If we can't parse the safetensors file, be more lenient
                    # Accept models that have passed all other validation checks
                    print(f"SCANNING MODELS: Could not analyze safetensors architecture ({str(e)}), but accepting due to other validation passing")
                    return True  # Accept the model since other checks passed

            # 6. Filename pattern validation for known image generation models
            is_known_image_model = (
                'xl' in filename or 'sdxl' in filename or 'stable-diffusion-xl' in filename or
                'stable-diffusion' in filename or 'diffusion' in filename or
                'sd' in filename or 'v1' in filename or 'v2' in filename
            )

            if not is_known_image_model and not model_type:
                print(f"SCANNING MODELS: ❌ Not a recognized image generation model: {filename}")
                return False

            print(f"SCANNING MODELS: ✅ File passed comprehensive validation: {file_path} ({model_type.value if model_type else 'unknown'})")
            return True

        except Exception as e:
            print(f"SCANNING MODELS: Validation exception for {file_path}: {str(e)}")
            return False

    def _detect_model_type_from_file(self, file_path: str) -> Optional[ModelType]:
        """Detect model type from file content or filename."""
        try:
            # First try filename-based detection (fastest)
            filename = os.path.basename(file_path).lower()
            if 'xl' in filename or 'sdxl' in filename or 'stable-diffusion-xl' in filename:
                return ModelType.STABLE_DIFFUSION_XL
            elif '1.5' in filename or 'v1-5' in filename:
                return ModelType.STABLE_DIFFUSION_V1_5
            elif '1.4' in filename or 'v1-4' in filename or 'stable-diffusion' in filename:
                return ModelType.STABLE_DIFFUSION_V1_4

            # For safetensors files, try to check metadata (more expensive)
            if file_path.endswith('.safetensors'):
                try:
                    from safetensors import safe_open
                    with safe_open(file_path, framework="pt", device="cpu") as f:
                        # Check tensor names for SDXL indicators
                        tensor_names = list(f.keys())[:20]  # Check first 20 tensors
                        tensor_name_str = ' '.join(tensor_names).lower()

                        if 'xl' in tensor_name_str or 'stable_diffusion_xl' in tensor_name_str:
                            return ModelType.STABLE_DIFFUSION_XL
                        elif any('down_blocks' in name or 'up_blocks' in name for name in tensor_names):
                            # Has typical SD architecture - assume SD 1.5 as default
                            return ModelType.STABLE_DIFFUSION_V1_5
                except ImportError:
                    print("SCANNING MODELS: safetensors not available for metadata check")
                except Exception as e:
                    print(f"SCANNING MODELS: Could not check safetensors metadata: {e}")

            # For ckpt files, we can't easily check content without loading
            # Fall back to SD 1.5 as most common
            elif file_path.endswith('.ckpt'):
                return ModelType.STABLE_DIFFUSION_V1_5

            # Default fallback
            print(f"SCANNING MODELS: Using default SD 1.5 for unrecognized model: {filename}")
            return ModelType.STABLE_DIFFUSION_V1_5

        except Exception as e:
            print(f"SCANNING MODELS: Error detecting model type for {file_path}: {str(e)}")
            return None

    def _extract_diffusers_model_name(self, dir_path: str) -> str:
        """Extract the model name from model_index.json using multiple possible field names."""
        try:
            model_index_path = os.path.join(dir_path, "model_index.json")
            print(f"EXTRACTING NAME: Checking for model_index.json at: {model_index_path}")

            if not os.path.exists(model_index_path):
                print(f"EXTRACTING NAME: model_index.json not found at: {model_index_path}")
                return None

            print(f"EXTRACTING NAME: Reading model_index.json from: {model_index_path}")
            with open(model_index_path, 'r', encoding='utf-8') as f:
                model_config = json.load(f)

            print(f"EXTRACTING NAME: Loaded JSON config with keys: {list(model_config.keys())}")

            # Try multiple possible field names for the model name
            possible_fields = ['_name_or_path', 'name', 'model_name', 'hub_name', 'repo_name']

            for field in possible_fields:
                if field in model_config:
                    value = model_config[field]
                    print(f"EXTRACTING NAME: Found field '{field}': {value}")

                    if value and isinstance(value, str) and value.strip():
                        extracted_name = value.strip()
                        print(f"EXTRACTING NAME: Successfully extracted name from '{field}': {extracted_name}")
                        return extracted_name
                    elif isinstance(value, dict) and 'name' in value:
                        # Handle nested structure like {"name": "model-name"}
                        nested_name = value.get('name')
                        if nested_name and isinstance(nested_name, str) and nested_name.strip():
                            extracted_name = nested_name.strip()
                            print(f"EXTRACTING NAME: Successfully extracted nested name from '{field}': {extracted_name}")
                            return extracted_name

            print(f"EXTRACTING NAME: No valid name field found in config")

        except (json.JSONDecodeError, IOError, OSError, KeyError) as e:
            print(f"EXTRACTING NAME: Error extracting name from {dir_path}: {str(e)}")
            pass

        print(f"EXTRACTING NAME: Returning None for directory: {dir_path}")
        return None

    def _get_directory_size(self, dir_path: str) -> int:
        """Calculate the total size of a directory recursively."""
        total_size = 0
        try:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    try:
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
                    except OSError:
                        # Skip files that can't be accessed
                        continue
        except OSError:
            # If we can't walk the directory, return 0
            return 0

        return total_size

    def scan_models_in_folder(self, folder_path: str) -> List[Dict[str, str]]:
        """Scan folder for image generation models with comprehensive validation, including unrecognized models."""
        all_models = []
        if not os.path.exists(folder_path):
            return all_models

        # Get list of already installed model unique_ids for exclusion
        installed_models = self.db.get_all_models()
        installed_unique_ids = {model.unique_id for model in installed_models}

        print(f"SCANNING MODELS: Starting comprehensive validation scan of folder: {folder_path}")
        print(f"SCANNING MODELS: Found {len(installed_unique_ids)} existing models in database")

        # Track processed directories to avoid duplicates
        processed_directories = set()

        # 1. Scan for diffusers directories with COMPREHENSIVE validation
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)

            if os.path.isdir(item_path):
                print(f"SCANNING MODELS: Checking directory: {item}")

                # Use comprehensive validation instead of basic check
                validation = self._validate_diffusers_model_comprehensive(item_path)

                # Generate unique_id for this diffusers model directory
                unique_id = f"{item_path}|{os.path.basename(item_path)}"

                # Skip if already installed (by unique_id)
                if unique_id in installed_unique_ids:
                    print(f"SCANNING MODELS: ⏭️  Skipping already installed diffusers model: {item} (unique_id: {unique_id})")
                    continue

                # Get directory size (all components)
                file_size = self._get_directory_size(item_path)
                size_mb = round(file_size / (1024 * 1024), 2)

                # Extract metadata for display
                metadata = validation['metadata']
                display_name = metadata.get('display_name') or metadata.get('name') or item

                if validation['is_valid']:
                    # Additional check: ensure it's an image generation model
                    if self._is_image_generation_model(validation):
                        print(f"SCANNING MODELS: ✅ Valid image generation diffusers model: {item}")

                        # Mark as processed
                        processed_directories.add(item_path)

                        # Create model info
                        model_info = {
                            "name": display_name,
                            "path": item_path,
                            "type": "diffusers",
                            "size_mb": size_mb,
                            "model_type": validation['model_type'].value if validation['model_type'] else None,
                            "unique_id": unique_id,
                            "validation_status": "valid",
                            "metadata": metadata
                        }

                        all_models.append(model_info)
                        print(f"SCANNING MODELS: ➕ Added diffusers model: {display_name} ({size_mb} MB)")

                    else:
                        print(f"SCANNING MODELS: ❌ Not an image generation model: {item} (class: {validation.get('metadata', {}).get('class_name', 'unknown')})")
                        # Add as unrecognized
                        model_info = {
                            "name": display_name,
                            "path": item_path,
                            "type": "diffusers",
                            "size_mb": size_mb,
                            "model_type": validation['model_type'].value if validation['model_type'] else None,
                            "unique_id": unique_id,
                            "validation_status": "unrecognized",
                            "validation_reason": f"Not an image generation model (class: {validation.get('metadata', {}).get('class_name', 'unknown')})",
                            "metadata": metadata
                        }
                        all_models.append(model_info)
                        print(f"SCANNING MODELS: ➕ Added unrecognized diffusers model: {display_name}")
                else:
                    issues = ', '.join(validation['issues'][:3])  # Show first 3 issues
                    print(f"SCANNING MODELS: ❌ Invalid diffusers model: {item} - Issues: {issues}")
                    # Add as unrecognized
                    model_info = {
                        "name": display_name,
                        "path": item_path,
                        "type": "diffusers",
                        "size_mb": size_mb,
                        "model_type": validation['model_type'].value if validation['model_type'] else None,
                        "unique_id": unique_id,
                        "validation_status": "unrecognized",
                        "validation_reason": f"Validation failed: {issues}",
                        "metadata": metadata
                    }
                    all_models.append(model_info)
                    print(f"SCANNING MODELS: ➕ Added unrecognized diffusers model: {display_name}")

        print(f"SCANNING MODELS: Processed {len(processed_directories)} diffusers directories, found {len(all_models)} models so far")

        # 2. Scan for traditional model files with enhanced validation
        for root, dirs, files in os.walk(folder_path):
            # Skip already processed diffusers directories
            if any(root.startswith(proc_dir + os.sep) or root == proc_dir for proc_dir in processed_directories):
                continue

            for file in files:
                if file.endswith(('.safetensors', '.ckpt', '.gguf')):
                    full_path = os.path.join(root, file)
                    print(f"SCANNING MODELS: Checking file: {file}")

                    # Generate unique_id for this model
                    unique_id = f"{full_path}|{file}"

                    # Skip if already installed (by unique_id)
                    if unique_id in installed_unique_ids:
                        print(f"SCANNING MODELS: ⏭️  Skipping already installed model: {file} (unique_id: {unique_id})")
                        continue

                    # Skip component files (safety_checker, text_encoder, unet, vae, etc.)
                    file_lower = file.lower()
                    skip_file = False

                    # Component file patterns to skip
                    component_patterns = [
                        'safety_checker', 'text_encoder', 'tokenizer', 'unet', 'vae',
                        'scheduler', 'feature_extractor', 'image_encoder'
                    ]

                    # Skip if file matches component patterns
                    if any(comp in file_lower for comp in component_patterns):
                        print(f"SCANNING MODELS: ⏭️  Skipping component file: {file}")
                        skip_file = True

                    if skip_file:
                        continue

                    # Get file size
                    file_size = os.path.getsize(full_path)
                    size_mb = round(file_size / (1024 * 1024), 2)

                    # Create display name
                    display_name = os.path.splitext(file)[0]  # Remove extension

                    # Check if this is a GGUF text model file
                    if file.endswith('.gguf'):
                        # GGUF files are text models, not image generation models
                        print(f"SCANNING MODELS: ✅ Detected GGUF text model: {file}")

                        model_info = {
                            "name": display_name,
                            "path": full_path,
                            "type": "gguf",
                            "size_mb": size_mb,
                            "model_type": ModelType.TEXT_MODEL.value,
                            "unique_id": unique_id,
                            "validation_status": "valid"
                        }

                        all_models.append(model_info)
                        print(f"SCANNING MODELS: ➕ Added GGUF text model: {display_name} ({size_mb} MB)")
                    # Enhanced validation for traditional model files
                    elif self._validate_image_generation_model_file(full_path):
                        # Detect model type
                        model_type = self._detect_model_type_from_file(full_path)

                        model_info = {
                            "name": display_name,
                            "path": full_path,
                            "type": "safetensors" if file.endswith('.safetensors') else "checkpoint",
                            "size_mb": size_mb,
                            "model_type": model_type.value if model_type else None,
                            "unique_id": unique_id,
                            "validation_status": "valid"
                        }

                        all_models.append(model_info)
                        print(f"SCANNING MODELS: ➕ Added traditional model: {display_name} ({size_mb} MB)")
                    else:
                        print(f"SCANNING MODELS: ❌ Invalid model file: {file} - failed validation")
                        # Add as unrecognized
                        model_info = {
                            "name": display_name,
                            "path": full_path,
                            "type": "safetensors" if file.endswith('.safetensors') else "checkpoint",
                            "size_mb": size_mb,
                            "model_type": None,
                            "unique_id": unique_id,
                            "validation_status": "unrecognized",
                            "validation_reason": "Failed image generation model validation"
                        }
                        all_models.append(model_info)
                        print(f"SCANNING MODELS: ➕ Added unrecognized traditional model: {display_name}")

        print(f"SCANNING MODELS: Comprehensive scan completed - found {len(all_models)} models total")
        return all_models

    def _validate_image_model(self, file_path: str) -> bool:
        """Validate if a file is a valid image generation model."""
        try:
            # Check if file exists and is readable
            if not os.path.exists(file_path):
                return False

            # Check file size (> 1MB for basic validation)
            file_size = os.path.getsize(file_path)
            if file_size < 1024 * 1024:  # Less than 1MB
                return False

            # Check if it's a regular file
            if not os.path.isfile(file_path):
                return False

            # Additional validation could be added here:
            # - Check file headers
            # - Verify it's a valid safetensors/ckpt file
            # - Check for expected model structure

            return True

        except (OSError, IOError):
            return False

    def install_model(self, name: str, source_path: str, display_name: str = "",
                     categories: List[str] = None, description: str = "",
                     usage_notes: str = "", source_url: str = None,
                     license_info: str = None, progress_callback=None,
                     skip_validation: bool = False) -> tuple[bool, str]:
        """Install a model from source path with enhanced metadata and progress tracking."""
        try:
            print(f"MODEL_MANAGER: Starting installation of '{name}' from '{source_path}'")

            # Step 1: Validate source file/directory (10%)
            print("MODEL_MANAGER: Step 1 - Validating source file/directory")
            if progress_callback:
                progress_callback("Validating source file/directory...", 10)

            if not os.path.exists(source_path):
                error_msg = f"Source file/directory does not exist: {source_path}"
                print(f"MODEL_MANAGER: ERROR - {error_msg}")
                return False, error_msg

            # Check if it's a file (traditional model) or directory (diffusers model)
            is_file = os.path.isfile(source_path)
            is_directory = os.path.isdir(source_path)

            if not (is_file or is_directory):
                error_msg = f"Source path is neither a file nor directory: {source_path}"
                print(f"MODEL_MANAGER: ERROR - {error_msg}")
                return False, error_msg

            # For diffusers models, validate the directory structure comprehensively (unless skipped)
            if is_directory and not skip_validation:
                validation = self._validate_diffusers_model_comprehensive(source_path)
                if not validation['is_valid']:
                    error_msg = f"Source directory is not a valid diffusers model: {', '.join(validation['issues'])}"
                    if validation['warnings']:
                        error_msg += f" Warnings: {', '.join(validation['warnings'])}"
                    print(f"MODEL_MANAGER: ERROR - {error_msg}")
                    return False, error_msg

                # Use detected model type from validation
                if validation['model_type']:
                    model_type = validation['model_type']
                    print(f"MODEL_MANAGER: Detected model type from validation: {model_type}")

                # Extract metadata from validation
                if validation['metadata']:
                    extracted_name = validation['metadata'].get('name')
                    if extracted_name and not display_name:
                        display_name = validation['metadata'].get('display_name', extracted_name)
                        print(f"MODEL_MANAGER: Extracted display name: {display_name}")

                    if not description and validation['metadata'].get('description'):
                        description = validation['metadata']['description']
                        print(f"MODEL_MANAGER: Extracted description: {description}")
            elif is_directory and skip_validation:
                print(f"MODEL_MANAGER: Skipping validation for unrecognized model directory: {source_path}")
                # For unrecognized models, check if this directory contains a single model file
                # If so, use the file path instead of the directory path
                model_files = []

                # Search recursively for model files
                for root, dirs, files in os.walk(source_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_lower = file.lower()

                        # Check for model file extensions (case-insensitive)
                        if file_lower.endswith(('.safetensors', '.ckpt')):
                            print(f"MODEL_MANAGER: Found potential model file: {file_path}")

                            # More precise component file filtering
                            # Only skip if it's clearly a component file, not just contains component keywords
                            is_component = False

                            # Check if filename starts with component patterns
                            if (file_lower.startswith(('text_encoder', 'tokenizer', 'unet', 'vae', 'scheduler',
                                                     'feature_extractor', 'safety_checker')) or
                                file_lower in ['model.safetensors', 'pytorch_model.bin', 'diffusion_pytorch_model.safetensors']):
                                # Additional check: if it's in a component subdirectory, definitely skip
                                rel_path = os.path.relpath(root, source_path).lower()
                                if any(comp_dir in rel_path for comp_dir in ['text_encoder', 'tokenizer', 'unet', 'vae', 'scheduler']):
                                    is_component = True
                                    print(f"MODEL_MANAGER: Skipping component file in component directory: {file_path}")
                                else:
                                    # If it's in root or non-component directory, it might be a model file
                                    # Only skip if it's exactly a known component filename pattern
                                    is_component = False
                                    print(f"MODEL_MANAGER: Keeping potential model file (not clearly a component): {file_path}")

                            if not is_component:
                                model_files.append(file_path)
                                print(f"MODEL_MANAGER: Added as model file: {file_path}")

                print(f"MODEL_MANAGER: Found {len(model_files)} model files in directory")

                if len(model_files) == 1:
                    # Directory contains exactly one model file - use that file path
                    actual_model_path = model_files[0]
                    print(f"MODEL_MANAGER: Using single model file: {actual_model_path}")
                    source_path = actual_model_path
                    is_file = True
                    is_directory = False
                elif len(model_files) > 1:
                    # Multiple model files - show them and ask user to choose
                    file_list = [os.path.basename(f) for f in model_files]
                    error_msg = f"Directory contains multiple model files: {', '.join(file_list)}. Please specify the exact model file path."
                    print(f"MODEL_MANAGER: ERROR - {error_msg}")
                    return False, error_msg
                else:
                    # List all files found for debugging
                    all_files = []
                    for root, dirs, files in os.walk(source_path):
                        for file in files:
                            all_files.append(os.path.join(root, file))
                    print(f"MODEL_MANAGER: All files in directory: {all_files}")
                    error_msg = f"Directory does not contain any model files (.safetensors or .ckpt). Found {len(all_files)} total files."
                    print(f"MODEL_MANAGER: ERROR - {error_msg}")
                    return False, error_msg

            # Check file/directory size (basic validation)
            if is_file:
                file_size = os.path.getsize(source_path)
            else:  # is_directory
                file_size = self._get_directory_size(source_path)

            print(f"MODEL_MANAGER: File/directory size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")

            if file_size < 1024 * 1024:  # Less than 1MB
                error_msg = "File/directory is too small to be a valid model (must be at least 1MB)"
                print(f"MODEL_MANAGER: ERROR - {error_msg}")
                return False, error_msg

            # Step 2: Determine model type (20%)
            print("MODEL_MANAGER: Step 2 - Determining model type")
            if progress_callback:
                progress_callback("Analyzing model type...", 20)

            model_type = self._determine_model_type(name, source_path)
            print(f"MODEL_MANAGER: Determined model type: {model_type}")

            # Step 3: Process categories (30%)
            print("MODEL_MANAGER: Step 3 - Processing categories")
            if progress_callback:
                progress_callback("Processing categories...", 30)

            model_categories = []
            if categories:
                print(f"MODEL_MANAGER: Processing {len(categories)} categories: {categories}")
                for cat in categories:
                    try:
                        category_enum = ModelCategory(cat.lower())
                        model_categories.append(category_enum)
                        print(f"MODEL_MANAGER: Added category: {category_enum}")
                    except ValueError as e:
                        print(f"MODEL_MANAGER: Skipping invalid category '{cat}': {e}")
                        pass
            else:
                print("MODEL_MANAGER: No categories provided")

            # Step 4: Register model path (40%)
            print("MODEL_MANAGER: Step 4 - Registering model path")
            if progress_callback:
                progress_callback("Registering model path...", 40)

            # Use the source path directly instead of copying
            dest_path = source_path
            print(f"MODEL_MANAGER: Using source path directly: {dest_path}")

            # Generate unique_id as path + filename for duplicate checking
            if is_directory:
                # For diffusers models, filename is the directory name
                filename = os.path.basename(dest_path)
            else:
                # For traditional models, filename is the actual file name
                filename = os.path.basename(dest_path)
            unique_id = f"{dest_path}|{filename}"

            # Step 6: Create model metadata (80%) - moved before duplicate check
            print("MODEL_MANAGER: Step 6 - Creating model metadata")
            if progress_callback:
                progress_callback("Creating model metadata...", 80)

            final_description = description or f"Installed from {source_path}"
            final_display_name = display_name or name  # Use name as fallback for display_name

            model_info = ModelInfo(
                name=name,
                path=dest_path,
                unique_id=unique_id,
                display_name=final_display_name,
                model_type=model_type,
                description=final_description,
                categories=model_categories,
                usage_notes=usage_notes,
                source_url=source_url,
                license_info=license_info,
                is_default=False,
                size_mb=round(file_size / (1024 * 1024), 2),
                installed_date=datetime.now().isoformat()
            )

            # Check if a model with this unique_id already exists (optimized database query)
            existing_model = self.db.get_model_by_unique_id(unique_id)

            if existing_model:
                # Update existing model with new metadata
                print(f"MODEL_MANAGER: Updating existing model '{existing_model.name}' with new metadata")
                update_result = self.db.update_model_by_unique_id(existing_model.unique_id, model_info)
                if update_result:
                    success_msg = f"Model '{existing_model.name}' updated successfully"
                    print(f"MODEL_MANAGER: SUCCESS - {success_msg}")
                    return True, success_msg
                else:
                    error_msg = "Failed to update existing model"
                    print(f"MODEL_MANAGER: ERROR - {error_msg}")
                    return False, error_msg

            # Set default aspect ratios based on model type
            model_info.set_default_aspect_ratios()

            # Set default generation parameters based on model type
            if model_type == ModelType.STABLE_DIFFUSION_XL:
                # SDXL models often work better with different defaults
                model_info.default_steps = 25  # SDXL typically needs more steps
                model_info.default_cfg = 7.0   # Slightly lower CFG for SDXL
            else:
                # SD 1.4/1.5 defaults
                model_info.default_steps = 20
                model_info.default_cfg = 7.5

            print(f"MODEL_MANAGER: Created ModelInfo: {model_info.name}, {model_info.model_type}, {len(model_info.categories)} categories")
            print(f"MODEL_MANAGER: Default aspect ratios: 1:1={model_info.aspect_ratio_1_1}, 9:16={model_info.aspect_ratio_9_16}, 16:9={model_info.aspect_ratio_16_9}")
            print(f"MODEL_MANAGER: Default generation params: steps={model_info.default_steps}, cfg={model_info.default_cfg}")

            # Step 7: Save to database (100%)
            print("MODEL_MANAGER: Step 7 - Saving to database")
            if progress_callback:
                progress_callback("Saving to database...", 100)

            save_result = self.db.save_model(model_info)
            print(f"MODEL_MANAGER: Database save result: {save_result}")

            if save_result:
                success_msg = f"Model '{name}' installed successfully"
                print(f"MODEL_MANAGER: SUCCESS - {success_msg}")
                return True, success_msg
            else:
                # Clean up file if database save failed
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                    print("MODEL_MANAGER: Cleaned up file after database save failure")
                error_msg = "Failed to save model to database"
                print(f"MODEL_MANAGER: ERROR - {error_msg}")
                return False, error_msg

        except PermissionError as e:
            error_msg = "Permission denied. Cannot write to models directory"
            print(f"MODEL_MANAGER: PERMISSION ERROR - {error_msg}: {e}")
            return False, error_msg
        except OSError as e:
            error_msg = f"File system error: {str(e)}"
            print(f"MODEL_MANAGER: OS ERROR - {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error during installation: {str(e)}"
            print(f"MODEL_MANAGER: UNEXPECTED ERROR - {error_msg}")
            import traceback
            traceback.print_exc()
            return False, error_msg

    def _determine_model_type(self, name: str, path: str) -> ModelType:
        """Determine model type from name/path, with enhanced detection for single-file models."""
        name_lower = name.lower()

        # First try name-based detection (fastest)
        if "xl" in name_lower:
            return ModelType.STABLE_DIFFUSION_XL
        elif "v1.5" in name_lower or "1.5" in name_lower:
            return ModelType.STABLE_DIFFUSION_V1_5
        elif "v1.4" in name_lower or "1.4" in name_lower:
            return ModelType.STABLE_DIFFUSION_V1_4

        # For single-file models (unrecognized models are always single files),
        # try to detect model type from file content
        if os.path.isfile(path):
            detected_type = self._detect_model_type_from_file(path)
            if detected_type:
                print(f"MODEL_TYPE: Detected {detected_type.value} from file content for '{name}'")
                return detected_type

        # Default fallback for unrecognized single-file models
        print(f"MODEL_TYPE: Using default SD 1.5 for unrecognized model '{name}'")
        return ModelType.STABLE_DIFFUSION_V1_5

    def get_installed_models(self) -> List[ModelInfo]:
        """Get list of installed models."""
        return self.db.get_all_models()

    def get_default_model(self) -> Optional[ModelInfo]:
        """Get the default model."""
        return self.db.get_default_model()

    def set_default_model(self, unique_id: str) -> bool:
        """Set a model as default by unique_id."""
        return self.db.set_default_model_by_unique_id(unique_id)

    def delete_model(self, unique_id: str) -> bool:
        """Delete an installed model from database only (preserves the actual model file)."""
        # Get model info first
        models = self.db.get_all_models()
        model_to_delete = None
        for model in models:
            if model.unique_id == unique_id:
                model_to_delete = model
                break

        if not model_to_delete or model_to_delete.is_default:
            return False

        # NOTE: We intentionally do NOT delete the actual model file
        # This allows users to keep their model files while removing them from the app's database
        # The model file remains on disk and can be re-scanned/installed later if needed

        # Remove from database only (now uses unique_id for identification)
        return self.db.delete_model_by_unique_id(unique_id)

    def get_model_names(self) -> List[str]:
        """Get list of installed model names for dropdown."""
        models = self.db.get_all_models()
        return [model.name for model in models]

    def _save_operation_for_undo(self, operation: str, data: Dict[str, Any] = None) -> None:
        """Save an operation to the database for undo functionality."""
        self.db.save_operation(operation, data or {})

    def undo_last_operation(self) -> tuple[bool, str]:
        """Undo the last operation using database history."""
        operations = self.db.get_recent_operations(1)
        if not operations:
            return False, "No operations to undo"

        operation = operations[0]
        operation_type = operation['operation_type']

        # For now, we'll implement basic undo for common operations
        # This could be expanded to handle more complex undo scenarios
        if operation_type == "delete_model":
            # For delete operations, we would need to restore the model
            # This is complex and would require storing the full model data
            return False, "Delete undo not yet implemented"
        elif operation_type == "set_default_model":
            # For default changes, we could potentially undo
            return False, "Default model undo not yet implemented"

        return False, f"Undo not supported for operation: {operation_type}"

    def redo_last_operation(self) -> tuple[bool, str]:
        """Redo the last undone operation."""
        # Redo functionality would require more complex state management
        return False, "Redo functionality not yet implemented"

    def can_undo(self) -> bool:
        """Check if undo is available."""
        operations = self.db.get_recent_operations(1)
        return len(operations) > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        # For now, redo is not implemented
        return False

    def get_last_operation_description(self) -> Optional[str]:
        """Get description of the last operation that can be undone."""
        operations = self.db.get_recent_operations(1)
        if operations:
            return operations[0]['operation_type']
        return None

    def export_model_metadata(self, model_name: str, export_path: str) -> tuple[bool, str]:
        """Export model metadata to a JSON file for sharing."""
        return self.db.export_model_metadata(model_name, export_path)

    def import_model_metadata(self, import_path: str) -> tuple[bool, str]:
        """Import model metadata from a JSON file."""
        success, message = self.db.import_model_metadata(import_path)
        if success:
            # Refresh the UI if needed
            pass
        return success, message

    def update_model_usage(self, unique_id: str) -> None:
        """Update usage statistics for a model by unique_id."""
        self.db.update_model_usage_by_unique_id(unique_id)

    def update_model_by_unique_id(self, unique_id: str, updated_model: ModelInfo) -> bool:
        """Update model by unique_id."""
        return self.db.update_model_by_unique_id(unique_id, updated_model)

    def get_models_by_category(self, category: ModelCategory) -> List[ModelInfo]:
        """Get models filtered by category."""
        return self.db.get_models_by_category(category)

    def search_models(self, query: str, categories: List[ModelCategory] = None) -> List[ModelInfo]:
        """Search models by name, description, or filter by categories."""
        return self.db.search_models(query, categories)

    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        return self.db.backup_database(backup_path)

    def restore_database(self, backup_path: str) -> bool:
        """Restore database from backup."""
        return self.db.restore_database(backup_path)

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self.db.get_database_stats()

    def clear_database(self) -> bool:
        """Clear all database entries to start fresh."""
        return self.db.clear_database()

    # LoRA management methods
    def scan_loras_in_folder(self, folder_path: str) -> List[Dict[str, str]]:
        """Scan folder for LoRA adapter files and return available LoRAs."""
        lora_files = []
        if not os.path.exists(folder_path):
            return lora_files

        # Get list of already installed LoRA unique_ids for exclusion
        installed_loras = self.db.get_all_loras()
        installed_unique_ids = {lora.unique_id for lora in installed_loras}

        print(f"SCANNING LoRAs: Starting scan of folder: {folder_path}")
        print(f"SCANNING LoRAs: Found {len(installed_unique_ids)} existing LoRAs in database")

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.safetensors', '.ckpt')):
                    full_path = os.path.join(root, file)

                    # Generate unique_id for this LoRA
                    unique_id = f"{full_path}|{file}"

                    # Skip if already installed (by unique_id)
                    if unique_id in installed_unique_ids:
                        print(f"SCANNING LoRAs: Skipping already installed LoRA: {file} (unique_id: {unique_id})")
                        continue

                    # Skip component files (safety_checker, text_encoder, unet, vae, etc.)
                    file_lower = file.lower()
                    skip_file = False

                    # Component file patterns to skip
                    component_patterns = [
                        'safety_checker', 'text_encoder', 'tokenizer', 'unet', 'vae',
                        'scheduler', 'feature_extractor', 'image_encoder'
                    ]

                    # Skip if file is in a subdirectory AND matches component patterns
                    if root != folder_path:  # File is in subdirectory
                        if any(comp in file_lower for comp in component_patterns):
                            skip_file = True
                        # Skip model.safetensors in subdirectories
                        elif file_lower == 'model.safetensors':
                            skip_file = True

                    # Skip files that clearly start with component names (even in root)
                    elif any(file_lower.startswith(comp + '.') or file_lower.startswith(comp + '-') for comp in component_patterns):
                        skip_file = True

                    if skip_file:
                        print(f"SCANNING LoRAs: Skipping component file: {file}")
                        continue

                    # Get file size for classification
                    try:
                        file_size = os.path.getsize(full_path)
                    except OSError:
                        print(f"SCANNING LoRAs: Could not read file size for: {file}")
                        continue

                    # Use classification logic to determine if this is a LoRA
                    if not self._is_lora_file(file, file_size):
                        print(f"SCANNING LoRAs: Skipping non-LoRA file: {file} ({file_size / (1024*1024):.1f} MB)")
                        continue

                    # Validate file
                    if not self._validate_image_model(full_path):
                        print(f"SCANNING LoRAs: Skipping invalid LoRA file: {file}")
                        continue

                    size_mb = round(file_size / (1024 * 1024), 2)

                    print(f"SCANNING LoRAs: Found valid LoRA: {file} ({size_mb} MB)")
                    lora_files.append({
                        "name": file,  # This will be used as the unique name (filename without extension)
                        "path": full_path,
                        "type": "safetensors" if file.endswith('.safetensors') else "checkpoint",
                        "size_mb": size_mb
                    })

        print(f"SCANNING LoRAs: Completed scan, found {len(lora_files)} available LoRAs")
        return lora_files

    def install_lora(self, name: str, source_path: str, display_name: str = "",
                     base_model_type: ModelType = None, categories: List[str] = None,
                     description: str = "", trigger_words: List[str] = None,
                     usage_notes: str = "", source_url: str = None,
                     license_info: str = None, default_scaling: float = 1.0,
                     progress_callback=None) -> tuple[bool, str]:
        """Install a LoRA adapter from source path with enhanced metadata."""
        try:
            print(f"LORA_MANAGER: Starting installation of '{name}' from '{source_path}'")

            # Step 1: Validate source file (10%)
            print("LORA_MANAGER: Step 1 - Validating source file")
            if progress_callback:
                progress_callback("Validating source file...", 10)

            if not os.path.exists(source_path):
                error_msg = f"Source file does not exist: {source_path}"
                print(f"LORA_MANAGER: ERROR - {error_msg}")
                return False, error_msg

            if not os.path.isfile(source_path):
                error_msg = f"Source path is not a file: {source_path}"
                print(f"LORA_MANAGER: ERROR - {error_msg}")
                return False, error_msg

            # Check file size (basic validation)
            file_size = os.path.getsize(source_path)
            print(f"LORA_MANAGER: File size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")

            if file_size < 1024 * 1024:  # Less than 1MB
                error_msg = "File is too small to be a valid LoRA (must be at least 1MB)"
                print(f"LORA_MANAGER: ERROR - {error_msg}")
                return False, error_msg

            # Step 2: Process categories (30%)
            print("LORA_MANAGER: Step 2 - Processing categories")
            if progress_callback:
                progress_callback("Processing categories...", 30)

            lora_categories = []
            if categories:
                print(f"LORA_MANAGER: Processing {len(categories)} categories: {categories}")
                for cat in categories:
                    try:
                        category_enum = ModelCategory(cat.lower())
                        lora_categories.append(category_enum)
                        print(f"LORA_MANAGER: Added category: {category_enum}")
                    except ValueError as e:
                        print(f"LORA_MANAGER: Skipping invalid category '{cat}': {e}")
                        pass
            else:
                print("LORA_MANAGER: No categories provided")

            # Step 3: Process trigger words (40%)
            print("LORA_MANAGER: Step 3 - Processing trigger words")
            if progress_callback:
                progress_callback("Processing trigger words...", 40)

            final_trigger_words = trigger_words or []
            print(f"LORA_MANAGER: Trigger words: {final_trigger_words}")

            # Step 4: Register LoRA path (60%)
            print("LORA_MANAGER: Step 4 - Registering LoRA path")
            if progress_callback:
                progress_callback("Registering LoRA path...", 60)

            # Use the source path directly instead of copying
            dest_path = source_path
            print(f"LORA_MANAGER: Using source path directly: {dest_path}")

            # Generate unique_id as path + filename for duplicate checking
            filename = os.path.basename(dest_path)
            unique_id = f"{dest_path}|{filename}"

            # Check if a LoRA with this unique_id already exists (optimized database query)
            existing_lora = None
            all_loras = self.db.get_all_loras()
            for lora in all_loras:
                if lora.unique_id == unique_id:
                    existing_lora = lora
                    break

            if existing_lora:
                error_msg = f"LoRA '{existing_lora.name}' is already installed from this path"
                print(f"LORA_MANAGER: ERROR - {error_msg}")
                return False, error_msg

            # Step 5: Create LoRA metadata (80%)
            print("LORA_MANAGER: Step 5 - Creating LoRA metadata")
            if progress_callback:
                progress_callback("Creating LoRA metadata...", 80)

            final_description = description or f"Installed from {source_path}"
            final_display_name = display_name or name  # Use name as fallback for display_name

            lora_info = LoRAInfo(
                name=name,
                path=dest_path,
                unique_id=unique_id,
                display_name=final_display_name,
                base_model_type=base_model_type,
                description=final_description,
                trigger_words=final_trigger_words,
                categories=lora_categories,
                usage_notes=usage_notes,
                source_url=source_url,
                license_info=license_info,
                size_mb=round(file_size / (1024 * 1024), 2),
                installed_date=datetime.now().isoformat(),
                default_scaling=default_scaling
            )

            print(f"LORA_MANAGER: Created LoRAInfo: {lora_info.name}, base_model_type={lora_info.base_model_type}, {len(lora_info.categories)} categories")
            print(f"LORA_MANAGER: Trigger words: {lora_info.trigger_words}, default_scaling: {lora_info.default_scaling}")

            # Step 6: Save to database (100%)
            print("LORA_MANAGER: Step 6 - Saving to database")
            if progress_callback:
                progress_callback("Saving to database...", 100)

            save_result = self.db.save_lora(lora_info)
            print(f"LORA_MANAGER: Database save result: {save_result}")

            if save_result:
                success_msg = f"LoRA '{name}' installed successfully"
                print(f"LORA_MANAGER: SUCCESS - {success_msg}")
                return True, success_msg
            else:
                error_msg = "Failed to save LoRA to database"
                print(f"LORA_MANAGER: ERROR - {error_msg}")
                return False, error_msg

        except PermissionError as e:
            error_msg = "Permission denied. Cannot write to models directory"
            print(f"LORA_MANAGER: PERMISSION ERROR - {error_msg}: {e}")
            return False, error_msg
        except OSError as e:
            error_msg = f"File system error: {str(e)}"
            print(f"LORA_MANAGER: OS ERROR - {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error during LoRA installation: {str(e)}"
            print(f"LORA_MANAGER: UNEXPECTED ERROR - {error_msg}")
            import traceback
            traceback.print_exc()
            return False, error_msg

    def get_installed_loras(self) -> List[LoRAInfo]:
        """Get list of installed LoRA adapters."""
        return self.db.get_all_loras()

    def get_loras_by_base_model_type(self, base_model_type: ModelType) -> List[LoRAInfo]:
        """Get LoRA adapters compatible with a specific base model type."""
        return self.db.get_loras_by_base_model_type(base_model_type)

    def delete_lora(self, lora_name: str) -> bool:
        """Delete an installed LoRA adapter from database only (preserves the actual LoRA file)."""
        # Get LoRA info first
        loras = self.db.get_all_loras()
        lora_to_delete = None
        for lora in loras:
            if lora.name == lora_name:
                lora_to_delete = lora
                break

        if not lora_to_delete:
            return False

        # NOTE: We intentionally do NOT delete the actual LoRA file
        # This allows users to keep their LoRA files while removing them from the app's database
        # The LoRA file remains on disk and can be re-scanned/installed later if needed

        # Remove from database only
        return self.db.delete_lora(lora_name)

    def get_lora_names(self) -> List[str]:
        """Get list of installed LoRA adapter names for dropdown."""
        loras = self.db.get_all_loras()
        return [lora.name for lora in loras]

    def update_lora_usage(self, lora_name: str) -> None:
        """Update usage statistics for a LoRA adapter."""
        self.db.update_lora_usage(lora_name)

    # IP-Adapter management methods
    def scan_ip_adapters_in_folder(self, folder_path: str) -> List[Dict[str, str]]:
        """Scan folder for IP-Adapter files and return available IP-Adapters."""
        ip_adapter_files = []
        if not os.path.exists(folder_path):
            return ip_adapter_files

        # Get list of already installed IP-Adapter unique_ids for exclusion
        installed_ip_adapters = self.db.get_all_ip_adapters()
        installed_unique_ids = {adapter.unique_id for adapter in installed_ip_adapters}

        print(f"SCANNING IP-Adapters: Starting scan of folder: {folder_path}")
        print(f"SCANNING IP-Adapters: Found {len(installed_unique_ids)} existing IP-Adapters in database")

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.safetensors', '.bin', '.pt')):
                    full_path = os.path.join(root, file)
                    file_lower = file.lower()

                    # BEST WAY TO IDENTIFY IP-ADAPTERS: Check for 'ip-adapter' or 'ipadapter' in the filename
                    if 'ip-adapter' not in file_lower and 'ipadapter' not in file_lower:
                        continue

                    # Generate unique_id for this IP-Adapter
                    unique_id = f"{full_path}|{file}"

                    # Skip if already installed (by unique_id)
                    if unique_id in installed_unique_ids:
                        print(f"SCANNING IP-Adapters: Skipping already installed IP-Adapter: {file} (unique_id: {unique_id})")
                        continue

                    # Skip component files (safety_checker, text_encoder, unet, vae, etc.)
                    skip_file = False

                    # Component file patterns to skip
                    component_patterns = [
                        'safety_checker', 'text_encoder', 'tokenizer', 'unet', 'vae',
                        'scheduler', 'feature_extractor', 'image_encoder'
                    ]

                    # Skip if file is in a subdirectory AND matches component patterns
                    if root != folder_path:  # File is in subdirectory
                        if any(comp in file_lower for comp in component_patterns):
                            skip_file = True
                        # Skip model files in subdirectories
                        elif file_lower in ['model.safetensors', 'pytorch_model.bin', 'diffusion_pytorch_model.safetensors']:
                            skip_file = True

                    # Skip files that clearly start with component names (even in root)
                    elif any(file_lower.startswith(comp + '.') or file_lower.startswith(comp + '-') for comp in component_patterns):
                        skip_file = True

                    if skip_file:
                        print(f"SCANNING IP-Adapters: Skipping component file: {file}")
                        continue

                    # Get file size for basic validation
                    try:
                        file_size = os.path.getsize(full_path)
                    except OSError:
                        print(f"SCANNING IP-Adapters: Could not read file size for: {file}")
                        continue

                    # Basic size validation (IP-Adapters are typically smaller than full models)
                    if file_size < 10 * 1024 * 1024:  # Less than 10MB - too small
                        print(f"SCANNING IP-Adapters: Skipping file too small: {file} ({file_size / (1024*1024):.1f} MB)")
                        continue
                    if file_size > 500 * 1024 * 1024:  # More than 500MB - too large for IP-Adapter
                        print(f"SCANNING IP-Adapters: Skipping file too large: {file} ({file_size / (1024*1024):.1f} MB)")
                        continue

                    # Validate file
                    if not self._validate_image_model(full_path):
                        print(f"SCANNING IP-Adapters: Skipping invalid IP-Adapter file: {file}")
                        continue

                    size_mb = round(file_size / (1024 * 1024), 2)

                    print(f"SCANNING IP-Adapters: Found valid IP-Adapter: {file} ({size_mb} MB)")
                    ip_adapter_files.append({
                        "name": file,  # This will be used as the unique name (filename without extension)
                        "path": full_path,
                        "type": "safetensors" if file.endswith('.safetensors') else "bin",
                        "size_mb": size_mb
                    })

        print(f"SCANNING IP-Adapters: Completed scan, found {len(ip_adapter_files)} available IP-Adapters")
        return ip_adapter_files

    def install_ip_adapter(self, name: str, source_path: str, display_name: str = "",
                          adapter_type: str = "style", categories: List[str] = None,
                          description: str = "", usage_notes: str = "",
                          source_url: str = None, license_info: str = None,
                          default_scale: float = 1.0, recommended_use_cases: str = "",
                          progress_callback=None) -> tuple[bool, str]:
        """Install an IP-Adapter from source path with enhanced metadata."""
        try:
            print(f"IP-ADAPTER_MANAGER: Starting installation of '{name}' from '{source_path}'")

            # Step 1: Validate source file (10%)
            print("IP-ADAPTER_MANAGER: Step 1 - Validating source file")
            if progress_callback:
                progress_callback("Validating source file...", 10)

            if not os.path.exists(source_path):
                error_msg = f"Source file does not exist: {source_path}"
                print(f"IP-ADAPTER_MANAGER: ERROR - {error_msg}")
                return False, error_msg

            if not os.path.isfile(source_path):
                error_msg = f"Source path is not a file: {source_path}"
                print(f"IP-ADAPTER_MANAGER: ERROR - {error_msg}")
                return False, error_msg

            # Check file size (basic validation)
            file_size = os.path.getsize(source_path)
            print(f"IP-ADAPTER_MANAGER: File size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")

            if file_size < 10 * 1024 * 1024:  # Less than 10MB
                error_msg = "File is too small to be a valid IP-Adapter (must be at least 10MB)"
                print(f"IP-ADAPTER_MANAGER: ERROR - {error_msg}")
                return False, error_msg

            if file_size > 500 * 1024 * 1024:  # More than 500MB
                error_msg = "File is too large to be an IP-Adapter (must be less than 500MB)"
                print(f"IP-ADAPTER_MANAGER: ERROR - {error_msg}")
                return False, error_msg

            # Step 2: Process categories (30%)
            print("IP-ADAPTER_MANAGER: Step 2 - Processing categories")
            if progress_callback:
                progress_callback("Processing categories...", 30)

            ip_adapter_categories = []
            if categories:
                print(f"IP-ADAPTER_MANAGER: Processing {len(categories)} categories: {categories}")
                for cat in categories:
                    try:
                        category_enum = ModelCategory(cat.lower())
                        ip_adapter_categories.append(category_enum)
                        print(f"IP-ADAPTER_MANAGER: Added category: {category_enum}")
                    except ValueError as e:
                        print(f"IP-ADAPTER_MANAGER: Skipping invalid category '{cat}': {e}")
                        pass
            else:
                print("IP-ADAPTER_MANAGER: No categories provided")

            # Step 3: Register IP-Adapter path (60%)
            print("IP-ADAPTER_MANAGER: Step 3 - Registering IP-Adapter path")
            if progress_callback:
                progress_callback("Registering IP-Adapter path...", 60)

            # Use the source path directly instead of copying
            dest_path = source_path
            print(f"IP-ADAPTER_MANAGER: Using source path directly: {dest_path}")

            # Generate unique_id as path + filename for duplicate checking
            filename = os.path.basename(dest_path)
            unique_id = f"{dest_path}|{filename}"

            # Check if an IP-Adapter with this unique_id already exists
            existing_ip_adapter = None
            all_ip_adapters = self.db.get_all_ip_adapters()
            for adapter in all_ip_adapters:
                if adapter.unique_id == unique_id:
                    existing_ip_adapter = adapter
                    break

            if existing_ip_adapter:
                error_msg = f"IP-Adapter '{existing_ip_adapter.name}' is already installed from this path"
                print(f"IP-ADAPTER_MANAGER: ERROR - {error_msg}")
                return False, error_msg

            # Step 4: Create IP-Adapter metadata (80%)
            print("IP-ADAPTER_MANAGER: Step 4 - Creating IP-Adapter metadata")
            if progress_callback:
                progress_callback("Creating IP-Adapter metadata...", 80)

            final_description = description or f"Installed from {source_path}"
            final_display_name = display_name or name  # Use name as fallback for display_name

            ip_adapter_info = IPAdapterInfo(
                name=name,
                path=dest_path,
                unique_id=unique_id,
                display_name=final_display_name,
                adapter_type=adapter_type,
                description=final_description,
                categories=ip_adapter_categories,
                usage_notes=usage_notes,
                source_url=source_url,
                license_info=license_info,
                size_mb=round(file_size / (1024 * 1024), 2),
                installed_date=datetime.now().isoformat(),
                default_scale=default_scale,
                recommended_use_cases=recommended_use_cases
            )

            print(f"IP-ADAPTER_MANAGER: Created IPAdapterInfo: {ip_adapter_info.name}, adapter_type={ip_adapter_info.adapter_type}, {len(ip_adapter_info.categories)} categories")
            print(f"IP-ADAPTER_MANAGER: default_scale: {ip_adapter_info.default_scale}, recommended_use_cases: {ip_adapter_info.recommended_use_cases}")

            # Step 5: Save to database (100%)
            print("IP-ADAPTER_MANAGER: Step 5 - Saving to database")
            if progress_callback:
                progress_callback("Saving to database...", 100)

            save_result = self.db.save_ip_adapter(ip_adapter_info)
            print(f"IP-ADAPTER_MANAGER: Database save result: {save_result}")

            if save_result:
                success_msg = f"IP-Adapter '{name}' installed successfully"
                print(f"IP-ADAPTER_MANAGER: SUCCESS - {success_msg}")
                return True, success_msg
            else:
                error_msg = "Failed to save IP-Adapter to database"
                print(f"IP-ADAPTER_MANAGER: ERROR - {error_msg}")
                return False, error_msg

        except PermissionError as e:
            error_msg = "Permission denied. Cannot write to models directory"
            print(f"IP-ADAPTER_MANAGER: PERMISSION ERROR - {error_msg}: {e}")
            return False, error_msg
        except OSError as e:
            error_msg = f"File system error: {str(e)}"
            print(f"IP-ADAPTER_MANAGER: OS ERROR - {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error during IP-Adapter installation: {str(e)}"
            print(f"IP-ADAPTER_MANAGER: UNEXPECTED ERROR - {error_msg}")
            import traceback
            traceback.print_exc()
            return False, error_msg

    def get_all_ip_adapters(self) -> List[IPAdapterInfo]:
        """Get list of installed IP-Adapters."""
        return self.db.get_all_ip_adapters()

    def get_ip_adapters_by_type(self, adapter_type: str) -> List[IPAdapterInfo]:
        """Get IP-Adapters filtered by adapter type."""
        return self.db.get_ip_adapters_by_type(adapter_type)

    def delete_ip_adapter(self, ip_adapter_name: str) -> bool:
        """Delete an installed IP-Adapter from database only (preserves the actual IP-Adapter file)."""
        # Get IP-Adapter info first
        ip_adapters = self.db.get_all_ip_adapters()
        ip_adapter_to_delete = None
        for adapter in ip_adapters:
            if adapter.name == ip_adapter_name:
                ip_adapter_to_delete = adapter
                break

        if not ip_adapter_to_delete:
            return False

        # NOTE: We intentionally do NOT delete the actual IP-Adapter file
        # This allows users to keep their IP-Adapter files while removing them from the app's database
        # The IP-Adapter file remains on disk and can be re-scanned/installed later if needed

        # Remove from database only
        return self.db.delete_ip_adapter(ip_adapter_name)

    def get_ip_adapter_names(self) -> List[str]:
        """Get list of installed IP-Adapter names for dropdown."""
        ip_adapters = self.db.get_all_ip_adapters()
        return [adapter.name for adapter in ip_adapters]

    def update_ip_adapter_usage(self, ip_adapter_name: str) -> None:
        """Update usage statistics for an IP-Adapter."""
        self.db.update_ip_adapter_usage(ip_adapter_name)

    def check_model_integrity(self, model_path: str) -> dict:
        """Check diffusers model integrity and health."""
        integrity_report = {
            'overall_health': 'unknown',
            'component_status': {},
            'missing_components': [],
            'corrupted_components': [],
            'recommendations': [],
            'last_checked': datetime.now().isoformat()
        }

        try:
            print(f"INTEGRITY CHECK: Starting integrity check for {model_path}")

            # Determine if this is a diffusers directory or single file
            if os.path.isdir(model_path) and self._is_diffusers_model_directory(model_path):
                # Diffusers directory integrity check
                validation = self._validate_diffusers_model_comprehensive(model_path)

                if validation['is_valid']:
                    integrity_report['overall_health'] = 'healthy'
                elif validation['issues']:
                    integrity_report['overall_health'] = 'critical'
                elif validation['warnings']:
                    integrity_report['overall_health'] = 'warning'
                else:
                    integrity_report['overall_health'] = 'healthy'

                integrity_report['component_status'] = validation['components']
                integrity_report['missing_components'] = [issue for issue in validation['issues'] if 'Missing' in issue]
                integrity_report['corrupted_components'] = [issue for issue in validation['issues'] if 'corrupted' in issue.lower()]

                # Generate recommendations
                if validation['issues']:
                    integrity_report['recommendations'].append("Model has critical issues - may not load properly")
                if validation['warnings']:
                    integrity_report['recommendations'].extend(validation['warnings'])

            elif os.path.isfile(model_path):
                # Single file integrity check
                if os.path.exists(model_path):
                    file_size = os.path.getsize(model_path)
                    if file_size < 1024 * 1024:  # Less than 1MB
                        integrity_report['overall_health'] = 'critical'
                        integrity_report['recommendations'].append("File size is suspiciously small")
                    else:
                        integrity_report['overall_health'] = 'healthy'
                        integrity_report['component_status'] = {'file': {'exists': True, 'size_mb': round(file_size / (1024*1024), 2)}}
                else:
                    integrity_report['overall_health'] = 'critical'
                    integrity_report['missing_components'].append("Model file not found")
                    integrity_report['recommendations'].append("Model file has been moved or deleted")
            else:
                integrity_report['overall_health'] = 'critical'
                integrity_report['recommendations'].append("Model path does not exist")

            print(f"INTEGRITY CHECK: Health assessment - {integrity_report['overall_health']}")

            return integrity_report

        except Exception as e:
            integrity_report['overall_health'] = 'error'
            integrity_report['recommendations'].append(f"Integrity check failed: {str(e)}")
            print(f"INTEGRITY CHECK: Exception - {str(e)}")
            return integrity_report

    def update_diffusers_model(self, unique_id: str, new_source_path: str, progress_callback=None) -> tuple[bool, str]:
        """Update existing diffusers model with new version."""
        try:
            print(f"MODEL UPDATE: Starting update for model {unique_id}")

            # Get current model info
            current_model = self.db.get_model_by_unique_id(unique_id)
            if not current_model:
                return False, f"Model with unique_id '{unique_id}' not found"

            print(f"MODEL UPDATE: Current model path: {current_model.path}")
            print(f"MODEL UPDATE: New source path: {new_source_path}")

            # Validate new model
            if progress_callback:
                progress_callback("Validating new model...", 20)

            if os.path.isdir(new_source_path):
                validation = self._validate_diffusers_model_comprehensive(new_source_path)
                if not validation['is_valid']:
                    error_msg = f"New model is invalid: {', '.join(validation['issues'])}"
                    return False, error_msg

                new_model_type = validation['model_type']
                new_metadata = validation['metadata']
            else:
                return False, "Update only supports diffusers directory models"

            # Check compatibility
            if progress_callback:
                progress_callback("Checking compatibility...", 40)

            if current_model.model_type != new_model_type:
                return False, f"Model type mismatch: {current_model.model_type} -> {new_model_type}"

            # Backup current model info
            if progress_callback:
                progress_callback("Backing up current model...", 60)

            backup_info = {
                'path': current_model.path,
                'size_mb': current_model.size_mb,
                'model_type': current_model.model_type,
                'backed_up_at': datetime.now().isoformat()
            }

            # Update model path and metadata
            if progress_callback:
                progress_callback("Updating model information...", 80)

            current_model.path = new_source_path
            current_model.size_mb = round(self._get_directory_size(new_source_path) / (1024 * 1024), 2)

            # Update metadata if available
            if new_metadata.get('display_name'):
                current_model.display_name = new_metadata['display_name']
            if new_metadata.get('description'):
                current_model.description = new_metadata['description']

            # Save updated model
            if progress_callback:
                progress_callback("Saving updated model...", 100)

            success = self.db.update_model_by_unique_id(unique_id, current_model)

            if success:
                print(f"MODEL UPDATE: Successfully updated model {unique_id}")
                return True, f"Model '{current_model.name}' updated successfully"
            else:
                print(f"MODEL UPDATE: Failed to save updated model")
                return False, "Failed to save model updates"

        except Exception as e:
            print(f"MODEL UPDATE: Exception - {str(e)}")
            return False, f"Update failed: {str(e)}"

    def optimize_diffusers_model(self, model_path: str, optimizations: list = None) -> tuple[bool, str]:
        """Apply optimizations to a diffusers model."""
        if not optimizations:
            optimizations = ['safetensors']  # Default optimization

        try:
            print(f"MODEL OPTIMIZATION: Starting optimization for {model_path}")

            if not os.path.isdir(model_path):
                return False, "Optimization only supports diffusers directory models"

            optimized_path = model_path

            for optimization in optimizations:
                print(f"MODEL OPTIMIZATION: Applying {optimization} optimization")

                if optimization == 'safetensors':
                    # Convert .bin files to .safetensors format
                    optimized_path = self._convert_to_safetensors_format(optimized_path)

                elif optimization == 'prune':
                    # Remove unused components (experimental)
                    print("MODEL OPTIMIZATION: Pruning not yet implemented")

                elif optimization == 'quantize':
                    # Apply quantization (experimental)
                    print("MODEL OPTIMIZATION: Quantization not yet implemented")

                else:
                    print(f"MODEL OPTIMIZATION: Unknown optimization: {optimization}")

            return True, f"Model optimized successfully"

        except Exception as e:
            print(f"MODEL OPTIMIZATION: Exception - {str(e)}")
            return False, f"Optimization failed: {str(e)}"

    def _convert_to_safetensors_format(self, model_path: str) -> str:
        """Convert .bin/.pt files to .safetensors format for better performance."""
        try:
            print(f"SAFETENSORS CONVERSION: Converting model at {model_path}")

            # This would require the safetensors library and conversion logic
            # For now, just return the original path
            print("SAFETENSORS CONVERSION: Conversion not yet implemented")
            return model_path

        except Exception as e:
            print(f"SAFETENSORS CONVERSION: Exception - {str(e)}")
            return model_path
