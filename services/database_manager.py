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

from models.model_info import ModelInfo, ModelCategory, ModelType, LoRAInfo, IPAdapterInfo


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
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                unique_id TEXT NOT NULL UNIQUE, -- path + filename for uniqueness
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

            # Create loras table for LoRA adapters
            cursor.execute('''CREATE TABLE IF NOT EXISTS loras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                path TEXT NOT NULL,
                display_name TEXT,
                base_model_type TEXT, -- Compatible base model type
                description TEXT,
                trigger_words TEXT, -- JSON array of trigger words
                categories TEXT, -- JSON array of categories
                usage_notes TEXT,
                source_url TEXT,
                license_info TEXT,
                size_mb REAL,
                installed_date TEXT,
                last_used TEXT,
                usage_count INTEGER DEFAULT 0,
                default_scaling REAL DEFAULT 1.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )''')

            # Create ip_adapters table for IP-Adapter models
            cursor.execute('''CREATE TABLE IF NOT EXISTS ip_adapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                path TEXT NOT NULL,
                display_name TEXT,
                adapter_type TEXT, -- Type of IP-Adapter (style, composition, etc.)
                description TEXT,
                categories TEXT, -- JSON array of categories
                usage_notes TEXT,
                source_url TEXT,
                license_info TEXT,
                size_mb REAL,
                installed_date TEXT,
                last_used TEXT,
                usage_count INTEGER DEFAULT 0,
                default_scale REAL DEFAULT 1.0,
                recommended_use_cases TEXT, -- Recommended use cases for this adapter
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )''')

            # Create text_models table for OpenAI models
            cursor.execute('''CREATE TABLE IF NOT EXISTS text_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL UNIQUE,
                display_name TEXT,
                description TEXT,
                context_window INTEGER,
                input_pricing REAL,
                output_pricing REAL,
                is_active BOOLEAN DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )''')

            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_models_name ON models(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_models_type ON models(model_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_models_default ON models(is_default)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_loras_name ON loras(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_loras_base_type ON loras(base_model_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ip_adapters_name ON ip_adapters(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ip_adapters_type ON ip_adapters(adapter_type)')
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

        # Add unique_id column if it doesn't exist
        try:
            cursor.execute("SELECT unique_id FROM models LIMIT 1")
        except sqlite3.OperationalError:
            print("DATABASE MIGRATION: Adding unique_id column to models table...")
            cursor.execute("ALTER TABLE models ADD COLUMN unique_id TEXT")
            # Populate existing rows with path as unique_id temporarily
            cursor.execute("UPDATE models SET unique_id = path WHERE unique_id IS NULL")
            print("DATABASE MIGRATION: unique_id column added successfully!")

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

        # Add recommended_use_cases column to ip_adapters table if it doesn't exist
        try:
            cursor.execute("SELECT recommended_use_cases FROM ip_adapters LIMIT 1")
        except sqlite3.OperationalError:
            print("DATABASE MIGRATION: Adding recommended_use_cases column to ip_adapters table...")
            cursor.execute("ALTER TABLE ip_adapters ADD COLUMN recommended_use_cases TEXT")
            print("DATABASE MIGRATION: recommended_use_cases column added successfully!")

        # Migration: Remove UNIQUE constraint from name column if it exists
        # This allows multiple models with the same name but different unique_ids
        try:
            # Check if there's a UNIQUE constraint on name by trying to insert duplicate names
            # If it fails, we need to recreate the table without the constraint
            cursor.execute("SELECT COUNT(*) FROM models WHERE name = 'test_unique_constraint_check'")
            test_count = cursor.fetchone()[0]

            if test_count == 0:  # No existing test entry
                try:
                    # Try to insert two rows with same name - if it fails, constraint exists
                    cursor.execute("INSERT INTO models (name, path, unique_id, display_name, model_type, description, categories, usage_notes, source_url, license_info, is_default, size_mb, installed_date, last_used, usage_count, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                 ("test_unique_constraint_check", "/test/path1", "test_unique_1", "test", None, "test", "[]", "", "", "", False, 0, "2025-01-01", None, 0, "2025-01-01"))
                    cursor.execute("INSERT INTO models (name, path, unique_id, display_name, model_type, description, categories, usage_notes, source_url, license_info, is_default, size_mb, installed_date, last_used, usage_count, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                 ("test_unique_constraint_check", "/test/path2", "test_unique_2", "test", None, "test", "[]", "", "", "", False, 0, "2025-01-01", None, 0, "2025-01-01"))

                    # If we get here, no constraint exists - clean up test entries
                    cursor.execute("DELETE FROM models WHERE name = 'test_unique_constraint_check'")
                    print("DATABASE MIGRATION: Name column allows duplicates - no migration needed")

                except sqlite3.IntegrityError:
                    # UNIQUE constraint exists on name - need to recreate table without it
                    print("DATABASE MIGRATION: UNIQUE constraint detected on name column - recreating table without constraint...")

                    # Get all existing data
                    cursor.execute("SELECT id, name, path, unique_id, display_name, model_type, description, categories, usage_notes, source_url, license_info, is_default, size_mb, installed_date, last_used, usage_count, aspect_ratio_1_1, aspect_ratio_9_16, aspect_ratio_16_9, default_steps, default_cfg FROM models")
                    existing_data = cursor.fetchall()

                    # Drop and recreate table without UNIQUE on name
                    cursor.execute("DROP TABLE models")

                    cursor.execute('''CREATE TABLE models (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        path TEXT NOT NULL,
                        unique_id TEXT NOT NULL UNIQUE,
                        display_name TEXT,
                        model_type TEXT,
                        description TEXT,
                        categories TEXT,
                        usage_notes TEXT,
                        source_url TEXT,
                        license_info TEXT,
                        is_default BOOLEAN DEFAULT 0,
                        size_mb REAL,
                        installed_date TEXT,
                        last_used TEXT,
                        usage_count INTEGER DEFAULT 0,
                        aspect_ratio_1_1 TEXT,
                        aspect_ratio_9_16 TEXT,
                        aspect_ratio_16_9 TEXT,
                        default_steps INTEGER DEFAULT 20,
                        default_cfg REAL DEFAULT 7.5,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )''')

                    # Restore data
                    for row in existing_data:
                        cursor.execute('''INSERT INTO models
                            (id, name, path, unique_id, display_name, model_type, description, categories,
                             usage_notes, source_url, license_info, is_default, size_mb, installed_date,
                             last_used, usage_count, aspect_ratio_1_1, aspect_ratio_9_16, aspect_ratio_16_9,
                             default_steps, default_cfg)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', row)

                    print("DATABASE MIGRATION: Table recreated without UNIQUE constraint on name - data preserved")

        except Exception as e:
            print(f"DATABASE MIGRATION: Error during name constraint migration: {e}")
            # Continue with other migrations

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
        """Save or update a model in the database with transaction safety."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("BEGIN TRANSACTION")
            cursor = conn.cursor()

            # Debug logging
            print(f"DATABASE SAVE: Attempting to save model: {model.name}")
            print(f"DATABASE SAVE: unique_id: '{model.unique_id}'")
            print(f"DATABASE SAVE: path: '{model.path}'")

            # Validate required fields
            if not model.name or not model.unique_id:
                raise ValueError("Model name and unique_id are required")

            # Check if model with this unique_id already exists
            cursor.execute('SELECT id FROM models WHERE unique_id = ?', (model.unique_id,))
            existing = cursor.fetchone()

            # Convert categories to JSON string - handle both enum objects and strings
            if model.categories:
                categories_json = json.dumps([cat.value if hasattr(cat, 'value') else str(cat) for cat in model.categories])
            else:
                categories_json = "[]"

            # Prepare values
            values = (
                model.name, model.path, model.unique_id, model.display_name,
                model.model_type.value if model.model_type else None,
                model.description, categories_json, model.usage_notes,
                model.source_url, model.license_info, model.is_default,
                model.size_mb, model.installed_date, model.last_used,
                model.usage_count, getattr(model, 'aspect_ratio_1_1', ''),
                getattr(model, 'aspect_ratio_9_16', ''), getattr(model, 'aspect_ratio_16_9', ''),
                getattr(model, 'default_steps', 20), getattr(model, 'default_cfg', 7.5),
                datetime.now().isoformat()
            )

            print(f"DATABASE SAVE: Values to insert: {values}")

            if existing:
                # Update existing model
                print(f"DATABASE SAVE: Updating existing model with unique_id: {model.unique_id}")
                cursor.execute('''UPDATE models SET
                    name = ?, path = ?, display_name = ?, model_type = ?, description = ?,
                    categories = ?, usage_notes = ?, source_url = ?, license_info = ?,
                    is_default = ?, size_mb = ?, installed_date = ?, last_used = ?,
                    usage_count = ?, aspect_ratio_1_1 = ?, aspect_ratio_9_16 = ?, aspect_ratio_16_9 = ?,
                    default_steps = ?, default_cfg = ?, updated_at = ?
                    WHERE unique_id = ?''', values)
            else:
                # Insert new model
                print(f"DATABASE SAVE: Inserting new model with unique_id: {model.unique_id}")
                cursor.execute('''INSERT INTO models
                    (name, path, unique_id, display_name, model_type, description, categories, usage_notes,
                     source_url, license_info, is_default, size_mb, installed_date,
                     last_used, usage_count, aspect_ratio_1_1, aspect_ratio_9_16, aspect_ratio_16_9,
                     default_steps, default_cfg, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', values)

            conn.commit()
            print(f"DATABASE SAVE: Successfully saved model: {model.name}")
            return True

        except sqlite3.IntegrityError as e:
            print(f"DATABASE SAVE: Integrity error - possible duplicate unique_id: {e}")
            if conn:
                conn.rollback()
            return False
        except Exception as e:
            print(f"DATABASE SAVE: Failed to save model: {e}")
            import traceback
            traceback.print_exc()
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()

    def update_model(self, old_name: str, model: ModelInfo) -> bool:
        """Update an existing model, handling name changes safely."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Convert categories to JSON string
                categories_json = json.dumps([cat.value for cat in model.categories]) if model.categories else "[]"

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

    def get_model_by_unique_id(self, unique_id: str) -> Optional[ModelInfo]:
        """Get a model by unique_id."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''SELECT name, path, unique_id, display_name, model_type, description,
                                 categories, usage_notes, source_url, license_info, is_default, size_mb,
                                 installed_date, last_used, usage_count, aspect_ratio_1_1, aspect_ratio_9_16,
                                 aspect_ratio_16_9, default_steps, default_cfg FROM models
                                 WHERE unique_id = ? LIMIT 1''', (unique_id,))

                row = cursor.fetchone()
                if row:
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

                    return ModelInfo(
                        name=row[0], path=row[1], unique_id=row[2] or "",
                        display_name=row[3] or "",
                        model_type=model_type_value, description=row[5] or "",
                        categories=categories, usage_notes=row[7] or "",
                        source_url=row[8], license_info=row[9],
                        is_default=bool(row[10]), size_mb=row[11],
                        installed_date=row[12], last_used=row[13],
                        usage_count=row[14] or 0,
                        aspect_ratio_1_1=row[15] or "",
                        aspect_ratio_9_16=row[16] or "",
                        aspect_ratio_16_9=row[17] or "",
                        default_steps=row[18] or 20,
                        default_cfg=row[19] or 7.5
                    )
        except Exception as e:
            print(f"Failed to get model by unique_id: {e}")

        return None

    def update_model_by_unique_id(self, unique_id: str, updated_model: ModelInfo) -> bool:
        """Update model by unique_id with proper transaction safety."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("BEGIN TRANSACTION")
            cursor = conn.cursor()

            # First verify the model exists
            cursor.execute('SELECT id FROM models WHERE unique_id = ?', (unique_id,))
            existing = cursor.fetchone()
            if not existing:
                print(f"Model with unique_id '{unique_id}' not found")
                conn.rollback()
                return False

            # Ensure unique_id is preserved
            updated_model.unique_id = unique_id

            # Convert categories to JSON string
            categories_json = json.dumps([cat.value if hasattr(cat, 'value') else str(cat)
                                        for cat in updated_model.categories]) if updated_model.categories else "[]"

            # Perform explicit UPDATE (safer than INSERT OR REPLACE)
            cursor.execute('''UPDATE models SET
                name = ?, display_name = ?, model_type = ?, description = ?,
                categories = ?, usage_notes = ?, source_url = ?, license_info = ?,
                aspect_ratio_1_1 = ?, aspect_ratio_9_16 = ?, aspect_ratio_16_9 = ?,
                default_steps = ?, default_cfg = ?, updated_at = ?
                WHERE unique_id = ?''', (
                updated_model.name, updated_model.display_name,
                updated_model.model_type.value if updated_model.model_type else None,
                updated_model.description, categories_json,
                updated_model.usage_notes, updated_model.source_url,
                updated_model.license_info,
                getattr(updated_model, 'aspect_ratio_1_1', ''),
                getattr(updated_model, 'aspect_ratio_9_16', ''),
                getattr(updated_model, 'aspect_ratio_16_9', ''),
                getattr(updated_model, 'default_steps', 20),
                getattr(updated_model, 'default_cfg', 7.5),
                datetime.now().isoformat(),
                unique_id
            ))

            if cursor.rowcount == 0:
                print(f"No rows updated for unique_id '{unique_id}'")
                conn.rollback()
                return False

            conn.commit()
            print(f"Successfully updated model with unique_id '{unique_id}'")
            return True

        except sqlite3.IntegrityError as e:
            print(f"Database integrity error updating model: {e}")
            if conn:
                conn.rollback()
            return False
        except Exception as e:
            print(f"Error updating model by unique_id: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()

    def get_all_models(self) -> List[ModelInfo]:
        """Get all models from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''SELECT name, path, unique_id, display_name, model_type, description,
                                 categories, usage_notes, source_url, license_info, is_default, size_mb,
                                 installed_date, last_used, usage_count, aspect_ratio_1_1, aspect_ratio_9_16,
                                 aspect_ratio_16_9, default_steps, default_cfg FROM models ORDER BY name''')

                models = []
                for row in cursor.fetchall():
                    # Parse categories from JSON
                    categories_data = json.loads(row[6] or "[]")
                    categories = []
                    for cat in categories_data:
                        try:
                            categories.append(ModelCategory(cat))
                        except ValueError:
                            pass  # Skip invalid categories

                    # Handle optional model_type
                    model_type_value = None
                    if row[4]:  # model_type column
                        try:
                            model_type_value = ModelType(row[4])
                        except ValueError:
                            model_type_value = None

                    model = ModelInfo(
                        name=row[0], path=row[1], unique_id=row[2] or "",
                        display_name=row[3] or "",
                        model_type=model_type_value, description=row[5] or "",
                        categories=categories, usage_notes=row[7] or "",
                        source_url=row[8], license_info=row[9],
                        is_default=bool(row[10]), size_mb=row[11],
                        installed_date=row[12], last_used=row[13],
                        usage_count=row[14] or 0,
                        aspect_ratio_1_1=row[15] or "",
                        aspect_ratio_9_16=row[16] or "",
                        aspect_ratio_16_9=row[17] or "",
                        default_steps=row[18] or 20,
                        default_cfg=row[19] or 7.5
                    )
                    models.append(model)

                return models

        except Exception as e:
            print(f"Failed to get models: {e}")
            return []

    def delete_model(self, model_name: str) -> bool:
        """Delete a model from the database by name."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM models WHERE name = ?', (model_name,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Failed to delete model: {e}")
            return False

    def delete_model_by_unique_id(self, unique_id: str) -> bool:
        """Delete a model from the database by unique_id."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM models WHERE unique_id = ?', (unique_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Failed to delete model by unique_id: {e}")
            return False

    def get_default_model(self) -> Optional[ModelInfo]:
        """Get the default model."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''SELECT name, path, unique_id, display_name, model_type, description,
                                 categories, usage_notes, source_url, license_info, is_default, size_mb,
                                 installed_date, last_used, usage_count, aspect_ratio_1_1, aspect_ratio_9_16,
                                 aspect_ratio_16_9, default_steps, default_cfg FROM models
                                 WHERE is_default = 1 LIMIT 1''')

                row = cursor.fetchone()
                if row:
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

                    return ModelInfo(
                        name=row[0], path=row[1], unique_id=row[2] or "",
                        display_name=row[3] or "",
                        model_type=model_type_value, description=row[5] or "",
                        categories=categories, usage_notes=row[7] or "",
                        source_url=row[8], license_info=row[9],
                        is_default=bool(row[10]), size_mb=row[11],
                        installed_date=row[12], last_used=row[13],
                        usage_count=row[14] or 0,
                        aspect_ratio_1_1=row[15] or "",
                        aspect_ratio_9_16=row[16] or "",
                        aspect_ratio_16_9=row[17] or "",
                        default_steps=row[18] or 20,
                        default_cfg=row[19] or 7.5
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

    def set_default_model_by_unique_id(self, unique_id: str) -> bool:
        """Set a model as default by unique_id."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # First, unset all defaults
                cursor.execute('UPDATE models SET is_default = 0')

                # Then set the new default
                cursor.execute('UPDATE models SET is_default = 1 WHERE unique_id = ?', (unique_id,))

                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            print(f"Failed to set default model by unique_id: {e}")
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

    def update_model_usage_by_unique_id(self, unique_id: str) -> None:
        """Update usage statistics for a model by unique_id."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE models
                    SET usage_count = usage_count + 1,
                        last_used = ?,
                        updated_at = ?
                    WHERE unique_id = ?
                ''', (datetime.now().isoformat(), datetime.now().isoformat(), unique_id))
                conn.commit()
        except Exception as e:
            print(f"Failed to update model usage by unique_id: {e}")

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
                cursor.execute("DELETE FROM loras")
                cursor.execute("DELETE FROM ip_adapters")
                cursor.execute("DELETE FROM settings")
                cursor.execute("DELETE FROM operations_history")

                # Reset auto-increment counters
                cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('models', 'loras', 'ip_adapters', 'settings', 'operations_history')")

                conn.commit()
                return True

        except Exception as e:
            print(f"Failed to clear database: {e}")
            return False

    # LoRA operations
    def save_lora(self, lora: LoRAInfo) -> bool:
        """Save or update a LoRA adapter in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Convert trigger words and categories to JSON strings
                trigger_words_json = json.dumps(lora.trigger_words) if lora.trigger_words else "[]"
                categories_json = json.dumps([str(cat) for cat in lora.categories]) if lora.categories else "[]"

                cursor.execute('''INSERT OR REPLACE INTO loras
                    (name, path, display_name, base_model_type, description, trigger_words,
                     categories, usage_notes, source_url, license_info, size_mb,
                     installed_date, last_used, usage_count, default_scaling, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                    lora.name, lora.path, lora.display_name,
                    lora.base_model_type.value if lora.base_model_type else None,
                    lora.description, trigger_words_json, categories_json,
                    lora.usage_notes, lora.source_url, lora.license_info,
                    lora.size_mb, lora.installed_date, lora.last_used,
                    lora.usage_count, lora.default_scaling, datetime.now().isoformat()
                ))

                conn.commit()
                return True

        except Exception as e:
            print(f"Failed to save LoRA: {e}")
            return False

    def update_lora(self, old_name: str, lora: LoRAInfo) -> bool:
        """Update an existing LoRA adapter, handling name changes safely."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Convert trigger words and categories to JSON strings
                trigger_words_json = json.dumps(lora.trigger_words) if lora.trigger_words else "[]"
                categories_json = json.dumps([str(cat) for cat in lora.categories]) if lora.categories else "[]"

                if lora.name == old_name:
                    # Name hasn't changed, use UPDATE
                    cursor.execute('''UPDATE loras SET
                        path = ?, display_name = ?, base_model_type = ?, description = ?,
                        trigger_words = ?, categories = ?, usage_notes = ?, source_url = ?,
                        license_info = ?, size_mb = ?, installed_date = ?, last_used = ?,
                        usage_count = ?, default_scaling = ?, updated_at = ?
                        WHERE name = ?''', (
                        lora.path, lora.display_name,
                        lora.base_model_type.value if lora.base_model_type else None,
                        lora.description, trigger_words_json, categories_json,
                        lora.usage_notes, lora.source_url, lora.license_info,
                        lora.size_mb, lora.installed_date, lora.last_used,
                        lora.usage_count, lora.default_scaling, datetime.now().isoformat(),
                        old_name
                    ))
                else:
                    # Name changed, insert new row then delete old
                    cursor.execute('''INSERT INTO loras
                        (name, path, display_name, base_model_type, description, trigger_words,
                         categories, usage_notes, source_url, license_info, size_mb,
                         installed_date, last_used, usage_count, default_scaling, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                        lora.name, lora.path, lora.display_name,
                        lora.base_model_type.value if lora.base_model_type else None,
                        lora.description, trigger_words_json, categories_json,
                        lora.usage_notes, lora.source_url, lora.license_info,
                        lora.size_mb, lora.installed_date, lora.last_used,
                        lora.usage_count, lora.default_scaling, datetime.now().isoformat()
                    ))

                    # Then delete the old LoRA
                    cursor.execute('DELETE FROM loras WHERE name = ?', (old_name,))

                conn.commit()
                return True

        except Exception as e:
            print(f"Failed to update LoRA: {e}")
            return False

    def get_all_loras(self) -> List[LoRAInfo]:
        """Get all LoRA adapters from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM loras ORDER BY name')

                loras = []
                for row in cursor.fetchall():
                    # Parse trigger words and categories from JSON
                    trigger_words = json.loads(row[6] or "[]")
                    categories_data = json.loads(row[7] or "[]")
                    categories = []
                    for cat in categories_data:
                        try:
                            categories.append(ModelCategory(cat))
                        except ValueError:
                            pass

                    # Handle optional base_model_type
                    base_model_type = None
                    if row[4]:  # base_model_type column
                        try:
                            base_model_type = ModelType(row[4])
                        except ValueError:
                            base_model_type = None

                    lora = LoRAInfo(
                        name=row[1], path=row[2], display_name=row[3] or "",
                        base_model_type=base_model_type, description=row[5] or "",
                        trigger_words=trigger_words, categories=categories,
                        usage_notes=row[8] or "", source_url=row[9],
                        license_info=row[10], size_mb=row[11],
                        installed_date=row[12], last_used=row[13],
                        usage_count=row[14] or 0, default_scaling=row[15] or 1.0
                    )
                    loras.append(lora)

                return loras

        except Exception as e:
            print(f"Failed to get LoRAs: {e}")
            return []

    def get_loras_by_base_model_type(self, base_model_type: ModelType) -> List[LoRAInfo]:
        """Get LoRA adapters compatible with a specific base model type."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM loras WHERE base_model_type = ? ORDER BY name',
                             (base_model_type.value,))

                loras = []
                for row in cursor.fetchall():
                    # Parse trigger words and categories from JSON
                    trigger_words = json.loads(row[6] or "[]")
                    categories_data = json.loads(row[7] or "[]")
                    categories = []
                    for cat in categories_data:
                        try:
                            categories.append(ModelCategory(cat))
                        except ValueError:
                            pass

                    lora = LoRAInfo(
                        name=row[1], path=row[2], display_name=row[3] or "",
                        base_model_type=base_model_type, description=row[5] or "",
                        trigger_words=trigger_words, categories=categories,
                        usage_notes=row[8] or "", source_url=row[9],
                        license_info=row[10], size_mb=row[11],
                        installed_date=row[12], last_used=row[13],
                        usage_count=row[14] or 0, default_scaling=row[15] or 1.0
                    )
                    loras.append(lora)

                return loras

        except Exception as e:
            print(f"Failed to get LoRAs by base model type: {e}")
            return []

    def delete_lora(self, lora_name: str) -> bool:
        """Delete a LoRA adapter from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM loras WHERE name = ?', (lora_name,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Failed to delete LoRA: {e}")
            return False

    def update_lora_usage(self, lora_name: str) -> None:
        """Update usage statistics for a LoRA adapter."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''UPDATE loras
                    SET usage_count = usage_count + 1,
                        last_used = ?,
                        updated_at = ?
                    WHERE name = ?''', (datetime.now().isoformat(), datetime.now().isoformat(), lora_name))
                conn.commit()
        except Exception as e:
            print(f"Failed to update LoRA usage: {e}")

    def get_lora_names(self) -> List[str]:
        """Get list of installed LoRA adapter names."""
        loras = self.get_all_loras()
        return [lora.name for lora in loras]

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

    def get_models_by_category(self, category: ModelCategory) -> List[ModelInfo]:
        """Get models filtered by category."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''SELECT name, path, unique_id, display_name, model_type, description,
                                 categories, usage_notes, source_url, license_info, is_default, size_mb,
                                 installed_date, last_used, usage_count, aspect_ratio_1_1, aspect_ratio_9_16,
                                 aspect_ratio_16_9, default_steps, default_cfg FROM models
                                 WHERE categories LIKE ? ORDER BY name''', (f'%{category.value}%',))

                models = []
                for row in cursor.fetchall():
                    # Parse categories from JSON
                    categories_data = json.loads(row[6] or "[]")
                    categories = []
                    for cat in categories_data:
                        try:
                            categories.append(ModelCategory(cat))
                        except ValueError:
                            pass  # Skip invalid categories

                    # Handle optional model_type
                    model_type_value = None
                    if row[4]:  # model_type column
                        try:
                            model_type_value = ModelType(row[4])
                        except ValueError:
                            model_type_value = None

                    model = ModelInfo(
                        name=row[0], path=row[1], unique_id=row[2] or "",
                        display_name=row[3] or "",
                        model_type=model_type_value, description=row[5] or "",
                        categories=categories, usage_notes=row[7] or "",
                        source_url=row[8], license_info=row[9],
                        is_default=bool(row[10]), size_mb=row[11],
                        installed_date=row[12], last_used=row[13],
                        usage_count=row[14] or 0,
                        aspect_ratio_1_1=row[15] or "",
                        aspect_ratio_9_16=row[16] or "",
                        aspect_ratio_16_9=row[17] or "",
                        default_steps=row[18] or 20,
                        default_cfg=row[19] or 7.5
                    )
                    models.append(model)

                return models

        except Exception as e:
            print(f"Failed to get models by category: {e}")
            return []

    def search_models(self, query: str, categories: List[ModelCategory] = None) -> List[ModelInfo]:
        """Search models by name, description, or filter by categories."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Build query
                sql = '''SELECT name, path, unique_id, display_name, model_type, description,
                         categories, usage_notes, source_url, license_info, is_default, size_mb,
                         installed_date, last_used, usage_count, aspect_ratio_1_1, aspect_ratio_9_16,
                         aspect_ratio_16_9, default_steps, default_cfg FROM models'''

                conditions = []
                params = []

                # Add search query if provided
                if query:
                    query_lower = query.lower()
                    conditions.append('(LOWER(name) LIKE ? OR LOWER(display_name) LIKE ? OR LOWER(description) LIKE ?)')
                    params.extend([f'%{query_lower}%', f'%{query_lower}%', f'%{query_lower}%'])

                # Add category filter if provided
                if categories:
                    category_conditions = []
                    for category in categories:
                        category_conditions.append('categories LIKE ?')
                        params.append(f'%{category.value}%')
                    if category_conditions:
                        conditions.append('(' + ' OR '.join(category_conditions) + ')')

                # Combine conditions
                if conditions:
                    sql += ' WHERE ' + ' AND '.join(conditions)

                sql += ' ORDER BY name'

                cursor.execute(sql, params)

                models = []
                for row in cursor.fetchall():
                    # Parse categories from JSON
                    categories_data = json.loads(row[6] or "[]")
                    model_categories = []
                    for cat in categories_data:
                        try:
                            model_categories.append(ModelCategory(cat))
                        except ValueError:
                            pass  # Skip invalid categories

                    # Handle optional model_type
                    model_type_value = None
                    if row[4]:  # model_type column
                        try:
                            model_type_value = ModelType(row[4])
                        except ValueError:
                            model_type_value = None

                    model = ModelInfo(
                        name=row[0], path=row[1], unique_id=row[2] or "",
                        display_name=row[3] or "",
                        model_type=model_type_value, description=row[5] or "",
                        categories=model_categories, usage_notes=row[7] or "",
                        source_url=row[8], license_info=row[9],
                        is_default=bool(row[10]), size_mb=row[11],
                        installed_date=row[12], last_used=row[13],
                        usage_count=row[14] or 0,
                        aspect_ratio_1_1=row[15] or "",
                        aspect_ratio_9_16=row[16] or "",
                        aspect_ratio_16_9=row[17] or "",
                        default_steps=row[18] or 20,
                        default_cfg=row[19] or 7.5
                    )
                    models.append(model)

                return models

        except Exception as e:
            print(f"Failed to search models: {e}")
            return []

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get table counts
                stats = {}

                cursor.execute("SELECT COUNT(*) FROM models")
                stats['total_models'] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM text_models")
                stats['total_text_models'] = cursor.fetchone()[0]

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

    # Text model operations (OpenAI models)
    def save_text_model(self, model_id: str, display_name: str = None, description: str = None,
                       context_window: int = None, input_pricing: float = None,
                       output_pricing: float = None, is_active: bool = True) -> bool:
        """Save or update an OpenAI text model in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''INSERT OR REPLACE INTO text_models
                    (model_id, display_name, description, context_window, input_pricing,
                     output_pricing, is_active, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (
                    model_id, display_name, description, context_window,
                    input_pricing, output_pricing, is_active, datetime.now().isoformat()
                ))

                conn.commit()
                return True

        except Exception as e:
            print(f"Failed to save text model: {e}")
            return False

    def get_all_text_models(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get all text models from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if active_only:
                    cursor.execute('SELECT model_id, display_name, description, context_window, input_pricing, output_pricing FROM text_models WHERE is_active = 1 ORDER BY model_id')
                else:
                    cursor.execute('SELECT model_id, display_name, description, context_window, input_pricing, output_pricing FROM text_models ORDER BY model_id')

                models = []
                for row in cursor.fetchall():
                    models.append({
                        'model_id': row[0],
                        'display_name': row[1],
                        'description': row[2],
                        'context_window': row[3],
                        'input_pricing': row[4],
                        'output_pricing': row[5]
                    })

                return models

        except Exception as e:
            print(f"Failed to get text models: {e}")
            return []

    def get_text_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific text model by model_id."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT model_id, display_name, description, context_window, input_pricing, output_pricing FROM text_models WHERE model_id = ? AND is_active = 1 LIMIT 1', (model_id,))

                row = cursor.fetchone()
                if row:
                    return {
                        'model_id': row[0],
                        'display_name': row[1],
                        'description': row[2],
                        'context_window': row[3],
                        'input_pricing': row[4],
                        'output_pricing': row[5]
                    }

        except Exception as e:
            print(f"Failed to get text model: {e}")

        return None

    def has_text_models(self) -> bool:
        """Check if any text models are stored in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM text_models WHERE is_active = 1')
                count = cursor.fetchone()[0]
                return count > 0

        except Exception as e:
            print(f"Failed to check text models: {e}")
            return False

    def deactivate_text_model(self, model_id: str) -> bool:
        """Mark a text model as inactive."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('UPDATE text_models SET is_active = 0, updated_at = ? WHERE model_id = ?',
                             (datetime.now().isoformat(), model_id))
                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            print(f"Failed to deactivate text model: {e}")
            return False

    def clear_text_models(self) -> bool:
        """Clear all text models from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM text_models')
                conn.commit()
                return True

        except Exception as e:
            print(f"Failed to clear text models: {e}")
            return False

    # IP-Adapter operations
    def save_ip_adapter(self, ip_adapter: IPAdapterInfo) -> bool:
        """Save or update an IP-Adapter in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Convert categories to JSON string
                categories_json = json.dumps([str(cat) for cat in ip_adapter.categories]) if ip_adapter.categories else "[]"

                cursor.execute('''INSERT OR REPLACE INTO ip_adapters
                    (name, path, display_name, adapter_type, description, categories,
                     usage_notes, source_url, license_info, size_mb, installed_date,
                     last_used, usage_count, default_scale, recommended_use_cases, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                    ip_adapter.name, ip_adapter.path, ip_adapter.display_name,
                    ip_adapter.adapter_type, ip_adapter.description, categories_json,
                    ip_adapter.usage_notes, ip_adapter.source_url, ip_adapter.license_info,
                    ip_adapter.size_mb, ip_adapter.installed_date, ip_adapter.last_used,
                    ip_adapter.usage_count, ip_adapter.default_scale, ip_adapter.recommended_use_cases,
                    datetime.now().isoformat()
                ))

                conn.commit()
                return True

        except Exception as e:
            print(f"Failed to save IP-Adapter: {e}")
            return False

    def update_ip_adapter(self, old_name: str, ip_adapter: IPAdapterInfo) -> bool:
        """Update an existing IP-Adapter, handling name changes safely."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Convert categories to JSON string
                categories_json = json.dumps([str(cat) for cat in ip_adapter.categories]) if ip_adapter.categories else "[]"

                if ip_adapter.name == old_name:
                    # Name hasn't changed, use UPDATE
                    cursor.execute('''UPDATE ip_adapters SET
                        path = ?, display_name = ?, adapter_type = ?, description = ?,
                        categories = ?, usage_notes = ?, source_url = ?, license_info = ?,
                        size_mb = ?, installed_date = ?, last_used = ?, usage_count = ?,
                        default_scale = ?, recommended_use_cases = ?, updated_at = ?
                        WHERE name = ?''', (
                        ip_adapter.path, ip_adapter.display_name, ip_adapter.adapter_type,
                        ip_adapter.description, categories_json, ip_adapter.usage_notes,
                        ip_adapter.source_url, ip_adapter.license_info, ip_adapter.size_mb,
                        ip_adapter.installed_date, ip_adapter.last_used, ip_adapter.usage_count,
                        ip_adapter.default_scale, ip_adapter.recommended_use_cases, datetime.now().isoformat(), old_name
                    ))
                else:
                    # Name changed, insert new row then delete old
                    cursor.execute('''INSERT INTO ip_adapters
                        (name, path, display_name, adapter_type, description, categories,
                         usage_notes, source_url, license_info, size_mb, installed_date,
                         last_used, usage_count, default_scale, recommended_use_cases, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                        ip_adapter.name, ip_adapter.path, ip_adapter.display_name,
                        ip_adapter.adapter_type, ip_adapter.description, categories_json,
                        ip_adapter.usage_notes, ip_adapter.source_url, ip_adapter.license_info,
                        ip_adapter.size_mb, ip_adapter.installed_date, ip_adapter.last_used,
                        ip_adapter.usage_count, ip_adapter.default_scale, ip_adapter.recommended_use_cases, datetime.now().isoformat()
                    ))

                    # Then delete the old IP-Adapter
                    cursor.execute('DELETE FROM ip_adapters WHERE name = ?', (old_name,))

                conn.commit()
                return True

        except Exception as e:
            print(f"Failed to update IP-Adapter: {e}")
            return False

    def get_all_ip_adapters(self) -> List[IPAdapterInfo]:
        """Get all IP-Adapters from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM ip_adapters ORDER BY name')

                ip_adapters = []
                for row in cursor.fetchall():
                    # Parse categories from JSON
                    categories_data = json.loads(row[6] or "[]")
                    categories = []
                    for cat in categories_data:
                        try:
                            categories.append(ModelCategory(cat))
                        except ValueError:
                            pass

                    ip_adapter = IPAdapterInfo(
                        name=row[1], path=row[2], display_name=row[3] or "",
                        adapter_type=row[4] or "style", description=row[5] or "",
                        categories=categories, usage_notes=row[7] or "",
                        source_url=row[8], license_info=row[9], size_mb=row[10],
                        installed_date=row[11], last_used=row[12],
                        usage_count=row[13] or 0, default_scale=row[14] or 1.0,
                        recommended_use_cases=row[15] or ""
                    )
                    ip_adapters.append(ip_adapter)

                return ip_adapters

        except Exception as e:
            print(f"Failed to get IP-Adapters: {e}")
            return []

    def get_ip_adapters_by_type(self, adapter_type: str) -> List[IPAdapterInfo]:
        """Get IP-Adapters filtered by adapter type."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM ip_adapters WHERE adapter_type = ? ORDER BY name', (adapter_type,))

                ip_adapters = []
                for row in cursor.fetchall():
                    # Parse categories from JSON
                    categories_data = json.loads(row[6] or "[]")
                    categories = []
                    for cat in categories_data:
                        try:
                            categories.append(ModelCategory(cat))
                        except ValueError:
                            pass

                    ip_adapter = IPAdapterInfo(
                        name=row[1], path=row[2], display_name=row[3] or "",
                        adapter_type=row[4] or "style", description=row[5] or "",
                        categories=categories, usage_notes=row[7] or "",
                        source_url=row[8], license_info=row[9], size_mb=row[10],
                        installed_date=row[11], last_used=row[12],
                        usage_count=row[13] or 0, default_scale=row[14] or 1.0,
                        recommended_use_cases=row[15] or ""
                    )
                    ip_adapters.append(ip_adapter)

                return ip_adapters

        except Exception as e:
            print(f"Failed to get IP-Adapters by type: {e}")
            return []

    def delete_ip_adapter(self, ip_adapter_name: str) -> bool:
        """Delete an IP-Adapter from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM ip_adapters WHERE name = ?', (ip_adapter_name,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Failed to delete IP-Adapter: {e}")
            return False

    def update_ip_adapter_usage(self, ip_adapter_name: str) -> None:
        """Update usage statistics for an IP-Adapter."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''UPDATE ip_adapters
                    SET usage_count = usage_count + 1,
                        last_used = ?,
                        updated_at = ?
                    WHERE name = ?''', (datetime.now().isoformat(), datetime.now().isoformat(), ip_adapter_name))
                conn.commit()
        except Exception as e:
            print(f"Failed to update IP-Adapter usage: {e}")

    def get_ip_adapter_names(self) -> List[str]:
        """Get list of installed IP-Adapter names."""
        ip_adapters = self.get_all_ip_adapters()
        return [adapter.name for adapter in ip_adapters]
