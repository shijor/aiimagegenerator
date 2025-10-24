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

from models.model_info import ModelInfo, ModelType, ModelCategory
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

    def scan_models_in_folder(self, folder_path: str) -> List[Dict[str, str]]:
        """Scan folder for compatible model files and return available models."""
        model_files = []
        if not os.path.exists(folder_path):
            return model_files

        # Get list of already installed model names for exclusion
        installed_models = self.db.get_all_models()
        installed_names = {model.name for model in installed_models}

        print(f"SCANNING: Starting scan of folder: {folder_path}")
        print(f"SCANNING: Found {len(installed_names)} existing models in database")

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.safetensors', '.ckpt')):
                    full_path = os.path.join(root, file)

                    # Skip if already installed (by name)
                    if file in installed_names:
                        print(f"SCANNING: Skipping already installed model: {file}")
                        continue

                    # Skip component files (safety_checker, text_encoder, unet, vae, etc.)
                    # Be more permissive - only skip files that are clearly components
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
                        print(f"SCANNING: Skipping component file: {file}")
                        continue

                    # Validate file
                    if not self._validate_image_model(full_path):
                        print(f"SCANNING: Skipping invalid model file: {file}")
                        continue

                    # Get file size
                    try:
                        file_size = os.path.getsize(full_path)
                        size_mb = round(file_size / (1024 * 1024), 2)
                    except OSError:
                        print(f"SCANNING: Could not read file size for: {file}")
                        continue

                    # Determine model type
                    model_type = self._determine_model_type(file, full_path)

                    print(f"SCANNING: Found valid model: {file}")
                    model_files.append({
                        "name": file,  # This will be used as the unique name (filename without extension)
                        "path": full_path,
                        "type": "safetensors" if file.endswith('.safetensors') else "checkpoint",
                        "size_mb": size_mb,
                        "model_type": model_type.value if model_type else None
                    })

        print(f"SCANNING: Completed scan, found {len(model_files)} available models")
        return model_files

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
                     license_info: str = None, progress_callback=None) -> tuple[bool, str]:
        """Install a model from source path with enhanced metadata and progress tracking."""
        try:
            print(f"MODEL_MANAGER: Starting installation of '{name}' from '{source_path}'")

            # Step 1: Validate source file (10%)
            print("MODEL_MANAGER: Step 1 - Validating source file")
            if progress_callback:
                progress_callback("Validating source file...", 10)

            if not os.path.exists(source_path):
                error_msg = f"Source file does not exist: {source_path}"
                print(f"MODEL_MANAGER: ERROR - {error_msg}")
                return False, error_msg

            if not os.path.isfile(source_path):
                error_msg = f"Source path is not a file: {source_path}"
                print(f"MODEL_MANAGER: ERROR - {error_msg}")
                return False, error_msg

            # Check file size (basic validation)
            file_size = os.path.getsize(source_path)
            print(f"MODEL_MANAGER: File size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")

            if file_size < 1024 * 1024:  # Less than 1MB
                error_msg = "File is too small to be a valid model (must be at least 1MB)"
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

            # Check if a model with this path already exists
            existing_models = self.get_installed_models()
            for existing_model in existing_models:
                if existing_model.path == dest_path:
                    error_msg = f"Model at path '{dest_path}' is already registered"
                    print(f"MODEL_MANAGER: ERROR - {error_msg}")
                    return False, error_msg

            # Step 6: Create model metadata (80%)
            print("MODEL_MANAGER: Step 6 - Creating model metadata")
            if progress_callback:
                progress_callback("Creating model metadata...", 80)

            final_description = description or f"Installed from {source_path}"
            final_display_name = display_name or name  # Use name as fallback for display_name

            model_info = ModelInfo(
                name=name,
                path=dest_path,
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

            print(f"MODEL_MANAGER: Created ModelInfo: {model_info.name}, {model_info.model_type}, {len(model_info.categories)} categories")

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
        """Determine model type from name/path."""
        name_lower = name.lower()
        if "xl" in name_lower:
            return ModelType.STABLE_DIFFUSION_XL
        elif "v1.5" in name_lower or "1.5" in name_lower:
            return ModelType.STABLE_DIFFUSION_V1_5
        else:
            return ModelType.STABLE_DIFFUSION_V1_4

    def get_installed_models(self) -> List[ModelInfo]:
        """Get list of installed models."""
        return self.db.get_all_models()

    def get_default_model(self) -> Optional[ModelInfo]:
        """Get the default model."""
        return self.db.get_default_model()

    def set_default_model(self, model_name: str) -> bool:
        """Set a model as default."""
        return self.db.set_default_model(model_name)

    def delete_model(self, model_name: str) -> bool:
        """Delete an installed model."""
        # Get model info first
        models = self.db.get_all_models()
        model_to_delete = None
        for model in models:
            if model.name == model_name:
                model_to_delete = model
                break

        if not model_to_delete or model_to_delete.is_default:
            return False

        # Remove file
        try:
            if os.path.exists(model_to_delete.path):
                os.remove(model_to_delete.path)
        except OSError:
            pass  # File might not exist

        # Remove from database
        return self.db.delete_model(model_name)

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

    def update_model_usage(self, model_name: str) -> None:
        """Update usage statistics for a model."""
        self.db.update_model_usage(model_name)

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
