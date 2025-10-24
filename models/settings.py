"""
Settings data model and persistence using SQLite database.
"""
import json
import os
from typing import Optional, Dict, Any
from pathlib import Path

# Import database manager
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.database_manager import DatabaseManager


class AppSettings:
    """Application settings data model."""

    def __init__(self):
        self.default_model_folder: Optional[str] = None
        self.theme: str = "default"
        self.performance_mode: str = "balanced"
        self.output_resolution: str = "512x512"
        self.cache_location: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "default_model_folder": self.default_model_folder,
            "theme": self.theme,
            "performance_mode": self.performance_mode,
            "output_resolution": self.output_resolution,
            "cache_location": self.cache_location,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppSettings':
        """Create settings from dictionary."""
        settings = cls()
        settings.default_model_folder = data.get("default_model_folder")
        settings.theme = data.get("theme", "default")
        settings.performance_mode = data.get("performance_mode", "balanced")
        settings.output_resolution = data.get("output_resolution", "512x512")
        settings.cache_location = data.get("cache_location", "default")
        return settings


class SettingsManager:
    """Manages application settings persistence using SQLite database."""

    def __init__(self):
        self.db = DatabaseManager()
        self._migration_done = False

    def _migrate_legacy_settings(self) -> None:
        """Migrate settings from JSON file to database if needed."""
        settings_file = "settings.json"
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r') as f:
                    settings_data = json.load(f)

                # Migrate to database
                for key, value in settings_data.items():
                    self.db.save_setting(key, value)

                # Backup and remove old file
                import shutil
                shutil.move(settings_file, settings_file + ".backup")
                print("Settings migrated to database successfully!")

            except Exception as e:
                print(f"Settings migration failed: {e}")

    def load_settings(self) -> AppSettings:
        """Load settings from database."""
        # Do migration lazily on first access
        if not self._migration_done:
            self._migrate_legacy_settings()
            self._migration_done = True

        try:
            settings_data = self.db.get_all_settings()
            return AppSettings.from_dict(settings_data)
        except Exception:
            return AppSettings()

    def save_settings(self, settings: AppSettings) -> None:
        """Save settings to database."""
        try:
            settings_dict = settings.to_dict()
            for key, value in settings_dict.items():
                self.db.save_setting(key, value)
        except Exception as e:
            print(f"Failed to save settings: {e}")

    # Remove class methods to avoid recursion issues
    # Use instance methods directly
