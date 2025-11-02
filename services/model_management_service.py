"""
Model management service - handles business logic for model management operations.
"""
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.model_manager import ModelManager
from models.model_info import ModelInfo, LoRAInfo, ModelCategory, ModelType


class ModelManagementService:
    """Service class handling all model management business logic."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def scan_models_in_folder(self, folder_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Scan folder for models and LoRAs, returning separate lists."""
        return self.model_manager.scan_models_in_folder(folder_path)

    def install_model(self, name: str, path: str, display_name: str = "",
                     categories: List[ModelCategory] = None, description: str = "",
                     usage_notes: str = "", source_url: str = "", license_info: str = "",
                     progress_callback: callable = None, skip_validation: bool = False) -> Tuple[bool, str]:
        """Install a model with metadata."""
        return self.model_manager.install_model(
            name, path, display_name, categories, description,
            usage_notes, source_url, license_info, progress_callback, skip_validation
        )

    def install_lora(self, name: str, path: str, display_name: str = "",
                    base_model_type: ModelType = None, categories: List[ModelCategory] = None,
                    description: str = "", trigger_words: List[str] = None,
                    usage_notes: str = "", source_url: str = "", license_info: str = "",
                    default_scaling: float = 1.0, progress_callback: callable = None) -> Tuple[bool, str]:
        """Install a LoRA adapter with metadata."""
        return self.model_manager.install_lora(
            name, path, display_name, base_model_type, categories, description,
            trigger_words, usage_notes, source_url, license_info, default_scaling, progress_callback
        )

    def get_installed_models(self) -> List[ModelInfo]:
        """Get all installed models."""
        return self.model_manager.get_installed_models()

    def get_installed_loras(self) -> List[LoRAInfo]:
        """Get all installed LoRAs."""
        return self.model_manager.get_installed_loras()

    def delete_model(self, unique_id: str) -> bool:
        """Delete a model by unique_id."""
        return self.model_manager.delete_model(unique_id)

    def delete_lora(self, lora_name: str) -> bool:
        """Delete a LoRA by name."""
        return self.model_manager.delete_lora(lora_name)

    def update_model_by_unique_id(self, unique_id: str, updated_model: ModelInfo) -> bool:
        """Update model by unique_id."""
        return self.model_manager.update_model_by_unique_id(unique_id, updated_model)

    def update_lora(self, old_name: str, lora: LoRAInfo) -> bool:
        """Update LoRA."""
        return self.model_manager.update_lora(old_name, lora)

    def export_model_metadata(self, model_name: str, export_path: str) -> Tuple[bool, str]:
        """Export model metadata."""
        return self.model_manager.export_model_metadata(model_name, export_path)

    def import_model_metadata(self, import_path: str) -> Tuple[bool, str]:
        """Import model metadata."""
        return self.model_manager.import_model_metadata(import_path)

    def clear_database(self) -> bool:
        """Clear all database entries."""
        return self.model_manager.clear_database()

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self.model_manager.can_undo()

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self.model_manager.can_redo()

    def undo_last_operation(self) -> Tuple[bool, str]:
        """Undo the last operation."""
        return self.model_manager.undo_last_operation()

    def redo_last_operation(self) -> Tuple[bool, str]:
        """Redo the last undone operation."""
        return self.model_manager.redo_last_operation()

    def get_last_operation_description(self) -> Optional[str]:
        """Get description of the last operation for UI display."""
        return self.model_manager.get_last_operation_description()

    def save_default_folder(self, folder_path: str) -> bool:
        """Save default model folder to settings."""
        return self.model_manager.db.save_setting("default_model_folder", folder_path)

    def get_default_folder(self) -> Optional[str]:
        """Get default model folder from settings."""
        return self.model_manager.db.get_setting("default_model_folder")
