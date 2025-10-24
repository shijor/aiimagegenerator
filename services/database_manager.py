"""
SQLite database manager for AI Image Generator.
"""
import sqlite3
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_info import ModelInfo, ModelCategory, ModelType


class DatabaseManager:
    """SQLite database manager for the application."""

    DB_FILE = "aiimg.db"

    def __init__(self, db_path: str = None):
        """Initialize database connection."""
        if db_path is None:
            # Default to models directory
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            os.makedirs(models_dir, exist_ok=True)
            db_path = os.path.join(models_dir, self.DB_FILE)

        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database and create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create models table
            cursor.execute('''CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                path TEXT NOT NULL,
                display_name TEXT,
                model_type TEXT, -- Optional model type
                description TEXT,
                categories TEXT, -- JSON array of categories
                usage_notes TEXT,
                source_url TEXT,
                license_info TEXT,
                is_default BOOLEAN DEFAULT 0,
                size_mb REAL,
                installed_date TEXT,
                last_used TEXT,
                usage_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )''')

            # Run migrations for existing databases
            self._run_database_migrations(cursor)

            # Create settings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create operations_history table for undo/redo
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS operations_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_type TEXT NOT NULL,
                    operation_data TEXT, -- JSON data
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_models_name ON models(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_models_type ON models(model_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_models_default ON models(is_default)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_settings_key ON settings(key)')

            conn.commit()

    def _run_database_migrations(self, cursor) -> None:
        """Run database migrations for existing databases."""
        # Add display_name column if it doesn't exist
        try:
            cursor.execute("SELECT display_name FROM models LIMIT 1")
        except sqlite3.OperationalError:
            print("DATABASE MIGRATION: Adding display_name column to models table...")
            cursor.execute("ALTER TABLE models ADD COLUMN display_name TEXT")
            print("DATABASE MIGRATION: display_name column added successfully!")

        # Add aspect ratio columns if they don't exist
        aspect_ratio_columns = [
            ("aspect_ratio_1_1", "TEXT"),
            ("aspect_ratio_9_16", "TEXT"),
            ("aspect_ratio_16_9", "TEXT")
        ]

        for column_name, column_type in aspect_ratio_columns:
            try:
                cursor.execute(f"SELECT {column_name} FROM models LIMIT 1")
            except sqlite3.OperationalError:
                print(f"DATABASE MIGRATION: Adding {column_name} column to models table...")
                cursor.execute(f"ALTER TABLE models ADD COLUMN {column_name} {column_type}")
                print(f"DATABASE MIGRATION: {column_name} column added successfully!")

        # Add default generation parameter columns if they don't exist
        generation_param_columns = [
            ("default_steps", "INTEGER DEFAULT 20"),
            ("default_cfg", "REAL DEFAULT 7.5")
        ]

        for column_name, column_type in generation_param_columns:
            try:
                cursor.execute(f"SELECT {column_name} FROM models LIMIT 1")
            except sqlite3.OperationalError:
                print(f"DATABASE MIGRATION: Adding {column_name} column to models table...")
                cursor.execute(f"ALTER TABLE models ADD COLUMN {column_name} {column_type}")
                print(f"DATABASE MIGRATION: {column_name} column added successfully!")

    def migrate_from_json(self) -> bool:
        """Migrate data from JSON files to SQLite database."""
        try:
            models_dir = os.path.dirname(self.db_path)
            json_file = os.path.join(models_dir, "installed_models.json")
            settings_file = os.path.join(os.path.dirname(models_dir), "settings.json")

            # Migrate models data
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    models_data = json.load(f)

                for model_data in models_data:
                    # Convert old format to new format
                    categories = model_data.get("categories", [])
                    if isinstance(categories, list) and categories:
                        # Convert ModelCategory enums to strings if needed
                        categories = [cat if isinstance(cat, str) else str(cat) for cat in categories]

                    self.save_model(ModelInfo(
                        name=model_data["name"],
                        path=model_data["path"],
                        model_type=model_data["model_type"],
                        description=model_data.get("description", ""),
                        categories=categories,
                        usage_notes=model_data.get("usage_notes", ""),
                        source_url=model_data.get("source_url"),
                        license_info=model_data.get("license_info"),
                        is_default=model_data.get("is_default", False),
                        size_mb=model_data.get("size_mb"),
                        installed_date=model_data.get("installed_date"),
                        last_used=model_data.get("last_used"),
                        usage_count=model_data.get("usage_count", 0)
                    ))

            # Migrate settings data
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    settings_data = json.load(f)

                for key, value in settings_data.items():
                    self.save_setting(key, value)

            return True

        except Exception as e:
            print(f"Migration failed: {e}")
            return False

    # Model operations
    def save_model(self, model: ModelInfo) -> bool:
        """Save or update a model in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Convert categories to JSON string
                categories_json = json.dumps([str(cat) for cat in model.categories]) if model.categories else "[]"

                cursor.execute('''
                    INSERT OR REPLACE INTO models
                    (name, path, display_name, model_type, description, categories, usage_notes,
                     source_url, license_info, is_default, size_mb, installed_date,
                     last_used, usage_count, aspect_ratio_1_1, aspect_ratio_9_16, aspect_ratio_16_9,
                     default_steps, default_cfg, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model.name, model.path, model.display_name,
                    model.model_type.value if model.model_type else None,
                    model.description, categories_json, model.usage_notes,
                    model.source_url, model.license_info, model.is_default,
                    model.size_mb, model.installed_date, model.last_used,
                    model.usage_count, model.aspect_ratio_1_1, model.aspect_ratio_9_16, model.aspect_ratio_16_9,
                    model.default_steps, model.default_cfg, datetime.now().isoformat()
                ))

                conn.commit()
                return True

        except Exception as e:
            print(f"Failed to save model: {e}")
            return False

    def update_model(self, old_name: str, model: ModelInfo) -> bool:
        """Update an existing model, handling name changes safely."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Convert categories to JSON string
                categories_json = json.dumps([str(cat) for cat in model.categories]) if model.categories else "[]"

                if model.name == old_name:
                    # Name hasn't changed, use UPDATE
                    cursor.execute('''
                        UPDATE models SET
                            path = ?, display_name = ?, model_type = ?, description = ?,
                            categories = ?, usage_notes = ?, source_url = ?, license_info = ?,
                            is_default = ?, size_mb = ?, installed_date = ?, last_used = ?,
                            usage_count = ?, aspect_ratio_1_1 = ?, aspect_ratio_9_16 = ?, aspect_ratio_16_9 = ?,
                            default_steps = ?, default_cfg = ?, updated_at = ?
                        WHERE name = ?
                    ''', (
                        model.path, model.display_name,
                        model.model_type.value if model.model_type else None,
                        model.description, categories_json, model.usage_notes,
                        model.source_url, model.license_info, model.is_default,
                        model.size_mb, model.installed_date, model.last_used,
                        model.usage_count, model.aspect_ratio_1_1, model.aspect_ratio_9_16, model.aspect_ratio_16_9,
                        model.default_steps, model.default_cfg, datetime.now().isoformat(),
                        old_name
                    ))
                else:
                    # Name changed, insert new row then delete old
                    # First, insert the new model
                    cursor.execute('''
                        INSERT INTO models
                        (name, path, display_name, model_type, description, categories, usage_notes,
                         source_url, license_info, is_default, size_mb, installed_date,
                         last_used, usage_count, aspect_ratio_1_1, aspect_ratio_9_16, aspect_ratio_16_9,
                         default_steps, default_cfg, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        model.name, model.path, model.display_name,
                        model.model_type.value if model.model_type else None,
                        model.description, categories_json, model.usage_notes,
                        model.source_url, model.license_info, model.is_default,
                        model.size_mb, model.installed_date, model.last_used,
                        model.usage_count, model.aspect_ratio_1_1, model.aspect_ratio_9_16, model.aspect_ratio_16_9,
                        model.default_steps, model.default_cfg, datetime.now().isoformat()
                    ))

                    # Then delete the old model
                    cursor.execute('DELETE FROM models WHERE name = ?', (old_name,))

                conn.commit()
                return True

        except Exception as e:
            print(f"Failed to update model: {e}")
            return False

    def get_all_models(self) -> List[ModelInfo]:
        """Get all models from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM models ORDER BY name')

                models = []
                for row in cursor.fetchall():
                    # Parse categories from JSON (index 5)
                    categories_data = json.loads(row[5] or "[]")
                    categories = []
                    for cat in categories_data:
                        try:
                            categories.append(ModelCategory(cat))
                        except ValueError:
                            pass  # Skip invalid categories

                    # Handle optional model_type (index 3)
                    model_type_value = None
                    if row[3]:  # model_type column
                        try:
                            model_type_value = ModelType(row[3])
                        except ValueError:
                            model_type_value = None

                    model = ModelInfo(
                        name=row[1], path=row[2], display_name=row[16] or "",
                        model_type=model_type_value, description=row[4] or "",
                        categories=categories, usage_notes=row[6] or "",
                        source_url=row[7], license_info=row[8],
                        is_default=bool(row[9]), size_mb=row[10],
                        installed_date=row[11], last_used=row[12],
                        usage_count=row[13] or 0,
                        aspect_ratio_1_1=row[17] or "",
                        aspect_ratio_9_16=row[18] or "",
                        aspect_ratio_16_9=row[19] or "",
                        default_steps=row[20] or 20,
                        default_cfg=row[21] or 7.5
                    )
                    models.append(model)

                return models

        except Exception as e:
            print(f"Failed to get models: {e}")
            return []

    def delete_model(self, model_name: str) -> bool:
        """Delete a model from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM models WHERE name = ?', (model_name,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Failed to delete model: {e}")
            return False

    def get_default_model(self) -> Optional[ModelInfo]:
        """Get the default model."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM models WHERE is_default = 1 LIMIT 1')

                row = cursor.fetchone()
                if row:
                    categories_data = json.loads(row[5] or "[]")
                    categories = []
                    for cat in categories_data:
                        try:
                            categories.append(ModelCategory(cat))
                        except ValueError:
                            pass

                    # Handle optional model_type (index 3)
                    model_type_value = None
                    if row[3]:  # model_type column
                        try:
                            model_type_value = ModelType(row[3])
                        except ValueError:
                            model_type_value = None

                    return ModelInfo(
                        name=row[1], path=row[2], display_name=row[16] or "",
                        model_type=model_type_value, description=row[4] or "",
                        categories=categories, usage_notes=row[6] or "",
                        source_url=row[7], license_info=row[8],
                        is_default=bool(row[9]), size_mb=row[10],
                        installed_date=row[11], last_used=row[12],
                        usage_count=row[13] or 0,
                        aspect_ratio_1_1=row[17] or "",
                        aspect_ratio_9_16=row[18] or "",
                        aspect_ratio_16_9=row[19] or "",
                        default_steps=row[20] or 20,
                        default_cfg=row[21] or 7.5
                    )
        except Exception as e:
            print(f"Failed to get default model: {e}")

        return None

    def set_default_model(self, model_name: str) -> bool:
        """Set a model as default."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # First, unset all defaults
                cursor.execute('UPDATE models SET is_default = 0')

                # Then set the new default
                cursor.execute('UPDATE models SET is_default = 1 WHERE name = ?', (model_name,))

                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            print(f"Failed to set default model: {e}")
            return False

    def update_model_usage(self, model_name: str) -> None:
        """Update usage statistics for a model."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE models
                    SET usage_count = usage_count + 1,
                        last_used = ?,
                        updated_at = ?
                    WHERE name = ?
                ''', (datetime.now().isoformat(), datetime.now().isoformat(), model_name))
                conn.commit()
        except Exception as e:
            print(f"Failed to update model usage: {e}")

    # Settings operations
    def save_setting(self, key: str, value: Any) -> bool:
        """Save a setting to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Convert value to string if it's not already
                if not isinstance(value, str):
                    value = json.dumps(value)

                cursor.execute('''
                    INSERT OR REPLACE INTO settings (key, value, updated_at)
                    VALUES (?, ?, ?)
                ''', (key, value, datetime.now().isoformat()))

                conn.commit()
                return True

        except Exception as e:
            print(f"Failed to save setting: {e}")
            return False

    def get_setting(self, key: str, default_value: Any = None) -> Any:
        """Get a setting from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT value FROM settings WHERE key = ?', (key,))

                row = cursor.fetchone()
                if row:
                    value = row[0]
                    # Try to parse as JSON, fallback to string
                    try:
                        return json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        return value

        except Exception as e:
            print(f"Failed to get setting: {e}")

        return default_value

    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings as a dictionary."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT key, value FROM settings')

                settings = {}
                for row in cursor.fetchall():
                    key, value = row
                    try:
                        settings[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        settings[key] = value

                return settings

        except Exception as e:
            print(f"Failed to get all settings: {e}")
            return {}

    # Operations history for undo/redo
    def save_operation(self, operation_type: str, operation_data: Dict[str, Any]) -> int:
        """Save an operation to history for undo/redo."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO operations_history (operation_type, operation_data, timestamp)
                    VALUES (?, ?, ?)
                ''', (operation_type, json.dumps(operation_data), datetime.now().isoformat()))

                operation_id = cursor.lastrowid
                conn.commit()
                return operation_id

        except Exception as e:
            print(f"Failed to save operation: {e}")
            return -1

    def get_recent_operations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent operations for undo functionality."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, operation_type, operation_data, timestamp
                    FROM operations_history
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))

                operations = []
                for row in cursor.fetchall():
                    operations.append({
                        'id': row[0],
                        'operation_type': row[1],
                        'operation_data': json.loads(row[2]),
                        'timestamp': row[3]
                    })

                return operations

        except Exception as e:
            print(f"Failed to get operations: {e}")
            return []

    def clear_old_operations(self, keep_last: int = 100) -> None:
        """Clear old operations history, keeping only the most recent ones."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM operations_history
                    WHERE id NOT IN (
                        SELECT id FROM operations_history
                        ORDER BY timestamp DESC
                        LIMIT ?
                    )
                ''', (keep_last,))
                conn.commit()
        except Exception as e:
            print(f"Failed to clear old operations: {e}")

    # Backup and restore
    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        try:
            with sqlite3.connect(self.db_path) as source_conn:
                with sqlite3.connect(backup_path) as backup_conn:
                    source_conn.backup(backup_conn)
            return True
        except Exception as e:
            print(f"Failed to backup database: {e}")
            return False

    def restore_database(self, backup_path: str) -> bool:
        """Restore database from backup."""
        try:
            # Create backup of current database first
            current_backup = f"{self.db_path}.backup"
            if os.path.exists(self.db_path):
                self.backup_database(current_backup)

            # Restore from backup
            with sqlite3.connect(backup_path) as backup_conn:
                with sqlite3.connect(self.db_path) as target_conn:
                    backup_conn.backup(target_conn)
            return True
        except Exception as e:
            print(f"Failed to restore database: {e}")
            return False

    def clear_database(self) -> bool:
        """Clear all data from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Clear all tables
                cursor.execute("DELETE FROM models")
                cursor.execute("DELETE FROM settings")
                cursor.execute("DELETE FROM operations_history")

                # Reset auto-increment counters
                cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('models', 'settings', 'operations_history')")

                conn.commit()
                return True

        except Exception as e:
            print(f"Failed to clear database: {e}")
            return False

    def export_model_metadata(self, model_name: str, export_path: str) -> Tuple[bool, str]:
        """Export model metadata to a JSON file for sharing."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM models WHERE name = ?', (model_name,))

                row = cursor.fetchone()
                if not row:
                    return False, f"Model '{model_name}' not found in database"

                # Parse categories from JSON
                categories_data = json.loads(row[6] or "[]")
                categories = []
                for cat in categories_data:
                    try:
                        categories.append(ModelCategory(cat))
                    except ValueError:
                        pass

                # Handle optional model_type
                model_type_value = None
                if row[4]:  # model_type column
                    try:
                        model_type_value = ModelType(row[4])
                    except ValueError:
                        model_type_value = None

                # Create model info object
                model = ModelInfo(
                    name=row[1], path=row[2], display_name=row[3] or "",
                    model_type=model_type_value, description=row[5] or "",
                    categories=categories, usage_notes=row[7] or "",
                    source_url=row[8], license_info=row[9],
                    is_default=bool(row[10]), size_mb=row[11],
                    installed_date=row[12], last_used=row[13],
                    usage_count=row[14] or 0
                )

                # Convert to exportable dictionary
                export_data = {
                    "name": model.name,
                    "display_name": model.display_name,
                    "model_type": model.model_type.value if model.model_type else None,
                    "description": model.description,
                    "categories": [cat.value for cat in model.categories],
                    "usage_notes": model.usage_notes,
                    "source_url": model.source_url,
                    "license_info": model.license_info,
                    "size_mb": model.size_mb,
                    "installed_date": model.installed_date,
                    "exported_at": datetime.now().isoformat(),
                    "version": "1.0"
                }

                # Write to JSON file
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)

                return True, f"Model metadata exported successfully to {export_path}"

        except Exception as e:
            error_msg = f"Failed to export model metadata: {str(e)}"
            print(error_msg)
            return False, error_msg

    def import_model_metadata(self, import_path: str) -> Tuple[bool, str]:
        """Import model metadata from a JSON file."""
        try:
            # Read JSON file
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)

            # Validate required fields
            required_fields = ["name", "model_type"]
            for field in required_fields:
                if field not in import_data:
                    return False, f"Missing required field: {field}"

            # Convert categories
            categories = []
            if "categories" in import_data:
                for cat in import_data["categories"]:
                    try:
                        categories.append(ModelCategory(cat))
                    except ValueError:
                        pass  # Skip invalid categories

            # Convert model type
            model_type = None
            if import_data.get("model_type"):
                try:
                    model_type = ModelType(import_data["model_type"])
                except ValueError:
                    pass

            # Create model info (without path since we're importing metadata only)
            model = ModelInfo(
                name=import_data["name"],
                path="",  # Path will be empty for imported metadata
                display_name=import_data.get("display_name", ""),
                model_type=model_type,
                description=import_data.get("description", ""),
                categories=categories,
                usage_notes=import_data.get("usage_notes", ""),
                source_url=import_data.get("source_url"),
                license_info=import_data.get("license_info"),
                is_default=False,  # Don't set imported models as default
                size_mb=import_data.get("size_mb"),
                installed_date=import_data.get("installed_date", datetime.now().isoformat()),
                last_used=None,
                usage_count=0
            )

            # Save to database (this will be metadata-only, user will need to provide actual model file)
            if self.save_model(model):
                return True, f"Model metadata imported successfully from {import_path}"
            else:
                return False, "Failed to save imported model metadata to database"

        except json.JSONDecodeError as e:
            return False, f"Invalid JSON file: {str(e)}"
        except Exception as e:
            error_msg = f"Failed to import model metadata: {str(e)}"
            print(error_msg)
            return False, error_msg

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get table counts
                stats = {}

                cursor.execute("SELECT COUNT(*) FROM models")
                stats['total_models'] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM settings")
                stats['total_settings'] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM operations_history")
                stats['total_operations'] = cursor.fetchone()[0]

                # Get database file size
                if os.path.exists(self.db_path):
                    stats['database_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)

                return stats

        except Exception as e:
            print(f"Failed to get database stats: {e}")
            return {}
