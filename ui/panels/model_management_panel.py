"""
Model management panel with sidebar and main area.
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QScrollArea, QGroupBox
from PyQt5.QtWidgets import QMessageBox, QProgressDialog, QLineEdit, QTextEdit, QComboBox, QCheckBox, QListWidget, QListWidgetItem, QProgressBar, QInputDialog
from PyQt5.QtCore import pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QFont

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.model_manager import ModelManager
from models.model_info import ModelCategory, ModelType, ModelInfo


class ModelManagementPanel(QWidget):
    """Panel for model management functionality."""

    model_installed = pyqtSignal()  # Emitted when a model is installed

    def __init__(self, model_manager: ModelManager):
        super().__init__()
        self.model_manager = model_manager

        # Load saved default folder if available
        self.selected_folder = self.model_manager.db.get_setting("default_model_folder")

        # Create sidebar and main area
        self.sidebar = self._create_sidebar()
        self.main_area = self._create_main_area()

        # Update UI with loaded default folder
        self._update_folder_display()

    def _update_folder_display(self):
        """Update the folder display based on loaded default folder."""
        if self.selected_folder:
            self.folder_path_label.setText(self.selected_folder)
            self.scan_btn.setEnabled(True)
        else:
            self.folder_path_label.setText("No folder selected")
            self.scan_btn.setEnabled(False)

    def _create_sidebar(self) -> QWidget:
        """Create the sidebar widget."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Folder Selection Section
        folder_group = QGroupBox("Model Folder")
        folder_layout = QVBoxLayout()

        folder_top_layout = QHBoxLayout()
        self.folder_path_label = QLabel("No folder selected")
        self.folder_path_label.setWordWrap(True)
        folder_top_layout.addWidget(self.folder_path_label)

        browse_btn = QPushButton('📁 Browse')
        browse_btn.clicked.connect(self.select_model_folder)
        folder_top_layout.addWidget(browse_btn)

        folder_layout.addLayout(folder_top_layout)

        save_default_btn = QPushButton('💾 Save as Default')
        save_default_btn.clicked.connect(self.save_default_folder)
        folder_layout.addWidget(save_default_btn)

        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)

        # Scan Section
        scan_group = QGroupBox("Scan for Models")
        scan_layout = QVBoxLayout()

        self.scan_btn = QPushButton('🔍 Scan Folder')
        self.scan_btn.clicked.connect(self.scan_models)
        self.scan_btn.setEnabled(False)
        scan_layout.addWidget(self.scan_btn)

        scan_group.setLayout(scan_layout)
        layout.addWidget(scan_group)

        # Undo/Redo Section
        undo_group = QGroupBox("Operations")
        undo_layout = QVBoxLayout()

        buttons_layout = QHBoxLayout()
        self.undo_btn = QPushButton('↶ Undo')
        self.undo_btn.clicked.connect(self.undo_operation)
        self.undo_btn.setEnabled(False)
        buttons_layout.addWidget(self.undo_btn)

        self.redo_btn = QPushButton('↷ Redo')
        self.redo_btn.clicked.connect(self.redo_operation)
        self.redo_btn.setEnabled(False)
        buttons_layout.addWidget(self.redo_btn)

        undo_layout.addLayout(buttons_layout)
        undo_group.setLayout(undo_layout)
        layout.addWidget(undo_group)

        # Model Sharing Section
        sharing_group = QGroupBox("Model Sharing")
        sharing_layout = QVBoxLayout()

        export_btn = QPushButton('📤 Export Metadata')
        export_btn.clicked.connect(self.export_model_metadata)
        sharing_layout.addWidget(export_btn)

        import_btn = QPushButton('📥 Import Metadata')
        import_btn.clicked.connect(self.import_model_metadata)
        sharing_layout.addWidget(import_btn)

        sharing_group.setLayout(sharing_layout)
        layout.addWidget(sharing_group)

        # Database Management Section
        db_group = QGroupBox("Database Management")
        db_layout = QVBoxLayout()

        clear_db_btn = QPushButton('🗑️ Clear Database')
        clear_db_btn.setStyleSheet("QPushButton { color: red; font-weight: bold; }")
        clear_db_btn.clicked.connect(self.clear_database)
        db_layout.addWidget(clear_db_btn)

        db_group.setLayout(db_layout)
        layout.addWidget(db_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _create_main_area(self) -> QWidget:
        """Create the main area widget."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Available Models Section
        available_group = QGroupBox("Available Models in Folder")
        available_layout = QVBoxLayout()

        # Scroll area for available models - use full available space
        available_scroll = QScrollArea()
        available_scroll.setWidgetResizable(True)
        available_scroll.setMinimumHeight(150)  # Minimum height
        available_scroll.setSizePolicy(available_scroll.sizePolicy().Expanding, available_scroll.sizePolicy().Expanding)

        available_widget = QWidget()
        self.available_models_layout = QVBoxLayout()
        self.available_models_layout.setContentsMargins(0, 0, 0, 0)
        self.available_models_layout.setSpacing(0)  # No spacing between rows
        self.available_models_layout.addWidget(QLabel("No models scanned yet. Select a folder and click 'Scan Folder'."))

        available_widget.setLayout(self.available_models_layout)
        available_scroll.setWidget(available_widget)
        available_layout.addWidget(available_scroll)

        available_group.setLayout(available_layout)
        layout.addWidget(available_group, stretch=1)  # Give it stretch to expand

        # Installation Progress Section
        progress_group = QGroupBox("Installation Progress")
        progress_layout = QVBoxLayout()

        self.progress_label = QLabel("Ready to install models")
        progress_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)  # Hidden by default
        progress_layout.addWidget(self.progress_bar)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Installed Models Section
        installed_group = QGroupBox("Installed Models")
        installed_layout = QVBoxLayout()

        # Scroll area for installed models
        installed_scroll = QScrollArea()
        installed_scroll.setWidgetResizable(True)
        installed_scroll.setMaximumHeight(250)

        installed_widget = QWidget()
        self.installed_models_layout = QVBoxLayout()

        # Add installed models
        self._refresh_installed_models()
        self._update_undo_redo_buttons()

        # Add refresh button for installed models
        refresh_layout = QHBoxLayout()
        refresh_label = QLabel("Installed Models")
        refresh_layout.addWidget(refresh_label)

        self.refresh_installed_btn = QPushButton('🔄')
        self.refresh_installed_btn.setMaximumWidth(30)
        self.refresh_installed_btn.setToolTip('Refresh installed models list')
        self.refresh_installed_btn.clicked.connect(self._refresh_installed_models)
        refresh_layout.addWidget(self.refresh_installed_btn)

        installed_group.setTitle("")  # Remove title since we have it in the layout
        # Add the title layout at the top of the installed group
        title_widget = QWidget()
        title_widget.setLayout(refresh_layout)
        installed_layout.insertWidget(0, title_widget)

        installed_widget.setLayout(self.installed_models_layout)
        installed_scroll.setWidget(installed_widget)
        installed_layout.addWidget(installed_scroll)

        installed_group.setLayout(installed_layout)
        layout.addWidget(installed_group)

        widget.setLayout(layout)
        return widget

    def select_model_folder(self):
        """Select model folder."""
        folder_path = QFileDialog.getExistingDirectory(self.sidebar, "Select Model Folder")
        if folder_path:
            self.selected_folder = folder_path
            self.folder_path_label.setText(folder_path)
            self.scan_btn.setEnabled(True)

    def save_default_folder(self):
        """Save selected folder as default."""
        if hasattr(self, 'selected_folder'):
            # Save to database instead of settings file
            success = self.model_manager.db.save_setting("default_model_folder", self.selected_folder)
            if success:
                QMessageBox.information(self.sidebar, "Success", "Default model folder saved!")
            else:
                QMessageBox.critical(self.sidebar, "Error", "Failed to save default model folder!")
        else:
            QMessageBox.warning(self.sidebar, "Warning", "Please select a folder first!")

    def scan_models(self):
        """Scan selected folder for models and show available models with install buttons."""
        if not hasattr(self, 'selected_folder'):
            QMessageBox.warning(self.sidebar, "No Folder", "Please select a folder first.")
            return

        # Clear previous results
        self._clear_layout(self.available_models_layout)
        self.available_models_layout.addWidget(QLabel("Scanning for models..."))

        # Scan for model files
        model_files = self.model_manager.scan_models_in_folder(self.selected_folder)

        # Update UI with results
        self._clear_layout(self.available_models_layout)

        if not model_files:
            self.available_models_layout.addWidget(QLabel("No compatible model files found."))
        else:
            # Show available models with install buttons
            for model_info in model_files:
                model_widget = self._create_available_model_widget(model_info)
                self.available_models_layout.addWidget(model_widget)

    def _create_available_model_widget(self, model_info: dict) -> QWidget:
        """Create widget for available model with install button - card-based vertical layout."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # Header section - Model name with icon
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        name_label = QLabel(f"📦 {model_info['name']}")
        name_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #1976D2;")
        header_layout.addWidget(name_label)
        header_layout.addStretch()

        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        layout.addWidget(header_widget)

        # Separator line
        separator = QWidget()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #e0e0e0;")
        layout.addWidget(separator)

        # Metadata section - Type and Size
        metadata_layout = QHBoxLayout()
        metadata_layout.setContentsMargins(0, 0, 0, 0)
        metadata_layout.setSpacing(15)

        type_label = QLabel(f"Type: {model_info.get('model_type', 'Unknown')}")
        type_label.setStyleSheet("color: #666; font-size: 11px;")
        metadata_layout.addWidget(type_label)

        size_label = QLabel(f"Size: {model_info['size_mb']:.1f} MB")
        size_label.setStyleSheet("color: #666; font-size: 11px;")
        metadata_layout.addWidget(size_label)

        metadata_layout.addStretch()

        metadata_widget = QWidget()
        metadata_widget.setLayout(metadata_layout)
        layout.addWidget(metadata_widget)

        # Path section - Full path
        path_label = QLabel(f"Path: {model_info['path']}")
        path_label.setStyleSheet("color: #888; font-size: 10px; font-family: monospace;")
        path_label.setWordWrap(True)
        path_label.setMaximumHeight(40)  # Limit height for long paths
        layout.addWidget(path_label)

        # Install button section - Prominent and full width
        install_btn = QPushButton('Install Model')
        install_btn.setProperty("model_name", model_info['name'])
        install_btn.setProperty("model_path", model_info['path'])
        install_btn.clicked.connect(self._on_install_clicked)
        install_btn.setMinimumHeight(32)
        install_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """)
        layout.addWidget(install_btn)

        # Set layout and card styling
        widget.setLayout(layout)
        widget.setStyleSheet("""
            QWidget {
                border: 2px solid #2196F3;
                border-radius: 8px;
                background-color: #f8f9ff;
                margin: 4px;
            }
            QWidget:hover {
                background-color: #e8f0ff;
                border-color: #1976D2;
            }
        """)
        widget.setMinimumHeight(120)
        widget.setMaximumHeight(160)

        return widget

    def _create_scanned_model_widget(self, model_info: dict) -> QWidget:
        """Create widget for newly scanned and installed model."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Model name and type
        name_label = QLabel(f"📦 {model_info['name']}")
        name_label.setStyleSheet("font-weight: bold; color: green;")
        layout.addWidget(name_label)

        # Model details
        details_label = QLabel(f"Type: {model_info.get('model_type', 'Unknown')}\nSize: {model_info['size_mb']:.1f} MB")
        layout.addWidget(details_label)

        # Path (truncated for display)
        path_text = model_info['path']
        if len(path_text) > 60:
            path_text = "..." + path_text[-57:]
        path_label = QLabel(f"Path: {path_text}")
        path_label.setStyleSheet("color: #666; font-size: 10px;")
        path_label.setWordWrap(True)
        layout.addWidget(path_label)

        widget.setLayout(layout)
        widget.setStyleSheet("border: 1px solid #4CAF50; border-radius: 3px; padding: 5px; margin: 2px; background-color: #f8fff8;")
        return widget

    def _on_install_clicked(self):
        """Handle install button click."""
        button = self.sender()
        if button:
            model_name = button.property("model_name")
            model_path = button.property("model_path")
            if model_name and model_path:
                self._install_model(model_name, model_path)

    def _install_model(self, name: str, path: str):
        """Install a model with enhanced metadata collection and progress tracking in the panel."""
        try:
            print(f"INSTALLATION: Starting installation for model '{name}' from path '{path}'")

            # Validate inputs
            if not name or not path:
                QMessageBox.critical(self.sidebar, "Error", "Invalid model name or path")
                return

            if not os.path.exists(path):
                QMessageBox.critical(self.sidebar, "Error", f"Model file does not exist: {path}")
                return

            # Show installation dialog for metadata collection
            dialog = ModelInstallDialog(name, path, self)
            print("INSTALLATION: Dialog created, showing...")

            result = dialog.exec_()
            print(f"INSTALLATION: Dialog result: {result} (type: {type(result)})")
            print(f"INSTALLATION: AcceptRole={QMessageBox.AcceptRole}, Accepted={QMessageBox.Accepted}, RejectRole={QMessageBox.RejectRole}, Rejected={QMessageBox.Rejected})")

            # Check if the Install button was clicked
            if result == QMessageBox.AcceptRole or result == QMessageBox.Accepted or result == 1:
                print("INSTALLATION: Dialog accepted, getting metadata...")

                # Get metadata from dialog
                metadata = dialog.get_metadata()
                print(f"INSTALLATION: Metadata collected: {metadata}")

                # Show progress in the panel
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)
                self.progress_label.setText("Starting installation...")
                print("INSTALLATION: Progress UI initialized")

                # Progress callback function
                def update_progress(message: str, percentage: int):
                    print(f"INSTALLATION: Progress update - {percentage}% - {message}")
                    self.progress_label.setText(message)
                    self.progress_bar.setValue(percentage)
                    # Process events to keep UI responsive
                    from PyQt5.QtWidgets import QApplication
                    QApplication.processEvents()

                print("INSTALLATION: Calling model_manager.install_model...")

                # Install model with metadata and progress callback
                success, message = self.model_manager.install_model(
                    name, path,
                    display_name=metadata.get('display_name', ''),
                    categories=metadata.get('categories', []),
                    description=metadata.get('description', ''),
                    usage_notes=metadata.get('usage_notes', ''),
                    source_url=metadata.get('source_url'),
                    license_info=metadata.get('license_info'),
                    progress_callback=update_progress
                )

                print(f"INSTALLATION: Model installation result - Success: {success}, Message: {message}")

                if success:
                    self.progress_label.setText("Installation completed successfully!")
                    QMessageBox.information(self.sidebar, "Success", message)
                    print("INSTALLATION: Refreshing installed models list...")
                    self._refresh_installed_models()
                    self.model_installed.emit()  # Notify other panels that a model was installed
                    print("INSTALLATION: Installation completed successfully!")
                else:
                    self.progress_label.setText("Installation failed!")
                    QMessageBox.critical(self.sidebar, "Installation Failed", message)
                    print(f"INSTALLATION: Installation failed with message: {message}")

            else:
                print("INSTALLATION: Dialog was cancelled by user")
                # User cancelled - do nothing

        except Exception as e:
            print(f"INSTALLATION: Exception occurred: {str(e)}")
            import traceback
            traceback.print_exc()
            self.progress_label.setText("Installation error!")
            QMessageBox.critical(self.sidebar, "Error", f"Unexpected error during installation: {str(e)}")
        finally:
            # Hide progress bar after a short delay
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(2000, lambda: self._hide_progress())

    def _hide_progress(self):
        """Hide the progress bar and reset the label."""
        self.progress_bar.setVisible(False)
        self.progress_label.setText("Ready to install models")
        self.progress_bar.setValue(0)

    def _refresh_installed_models(self):
        """Refresh the installed models list."""
        self._clear_layout(self.installed_models_layout)

        installed_models = self.model_manager.get_installed_models()
        if not installed_models:
            self.installed_models_layout.addWidget(QLabel("No models installed."))
        else:
            for model in installed_models:
                model_widget = self._create_installed_model_widget(model)
                self.installed_models_layout.addWidget(model_widget)

    def _create_installed_model_widget(self, model) -> QWidget:
        """Create widget for installed model - compact horizontal layout."""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(15)

        # Model name with icon and default indicator
        name_text = f"📦 {model.name}"
        if model.is_default:
            name_text += " ⭐"
        name_label = QLabel(name_text)
        name_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(name_label, stretch=2)

        # Model size - handle corrupted data gracefully
        try:
            size_mb = float(model.size_mb) if model.size_mb and str(model.size_mb).replace('.', '').isdigit() else 0.0
        except (ValueError, TypeError):
            size_mb = 0.0
        size_label = QLabel(f"{size_mb:.1f} MB")
        size_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(size_label, stretch=0)

        # Model type
        type_label = QLabel(model.model_type.value if model.model_type else 'Unknown')
        type_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(type_label, stretch=1)

        # Status/Description (truncated)
        desc_text = model.description or "No description"
        if len(desc_text) > 25:
            desc_text = desc_text[:22] + "..."
        desc_label = QLabel(desc_text)
        desc_label.setStyleSheet("color: #888; font-size: 10px;")
        desc_label.setToolTip(model.description or "No description")
        layout.addWidget(desc_label, stretch=2)

        # Action buttons container
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(2)

        # Edit button
        edit_btn = QPushButton('⚙️')
        edit_btn.setFixedSize(24, 24)
        edit_btn.setToolTip('Edit Parameters')
        edit_btn.clicked.connect(lambda: self._edit_model_params(model.name))
        buttons_layout.addWidget(edit_btn)

        # Delete button
        delete_btn = QPushButton('🗑️')
        delete_btn.setFixedSize(24, 24)
        delete_btn.setToolTip('Delete Model')
        delete_btn.clicked.connect(lambda: self._delete_model(model.name))
        delete_btn.setEnabled(not model.is_default)  # Can't delete default
        buttons_layout.addWidget(delete_btn)

        # Set Default button (only if not default)
        if not model.is_default:
            set_default_btn = QPushButton('⭐')
            set_default_btn.setFixedSize(24, 24)
            set_default_btn.setToolTip('Set as Default')
            set_default_btn.clicked.connect(lambda: self._set_default_model(model.name))
            buttons_layout.addWidget(set_default_btn)

        buttons_widget.setLayout(buttons_layout)
        layout.addWidget(buttons_widget, stretch=0)

        # Set layout and styling
        widget.setLayout(layout)
        widget.setStyleSheet("""
            QWidget {
                border: 1px solid #4CAF50;
                border-radius: 3px;
                background-color: #f8fff8;
                margin: 0px;
            }
            QWidget:hover {
                background-color: #f0fff0;
                border-color: #388E3C;
            }
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 2px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #e8f5e8;
            }
            QPushButton:disabled {
                color: #ccc;
            }
        """)
        widget.setFixedHeight(28)

        return widget

    def _edit_model_params(self, model_name: str):
        """Edit model parameters."""
        try:
            # Refresh the installed models list first to get latest data
            self._refresh_installed_models()

            # Get the current model data
            installed_models = self.model_manager.get_installed_models()
            current_model = None
            for model in installed_models:
                if model.name == model_name:
                    current_model = model
                    break

            if not current_model:
                QMessageBox.critical(self.sidebar, "Error", f"Model '{model_name}' not found!")
                return

            # Create and show edit dialog
            dialog = ModelEditDialog(current_model, self)
            result = dialog.exec_()

            if result == QMessageBox.AcceptRole or result == QMessageBox.Accepted:
                # Get updated metadata
                updated_metadata = dialog.get_metadata()

                # Validate name uniqueness if changed
                if updated_metadata['name'] != current_model.name:
                    # Check if new name already exists
                    existing_names = [m.name for m in installed_models if m.name != current_model.name]
                    if updated_metadata['name'] in existing_names:
                        QMessageBox.critical(self.sidebar, "Error",
                                           f"Model name '{updated_metadata['name']}' already exists!")
                        return

                # Create updated model info
                updated_model = ModelInfo(
                    name=updated_metadata['name'],
                    path=current_model.path,  # Path stays the same
                    display_name=updated_metadata['display_name'],
                    model_type=updated_metadata['model_type'],
                    description=updated_metadata['description'],
                    categories=updated_metadata['categories'],
                    usage_notes=updated_metadata['usage_notes'],
                    source_url=updated_metadata['source_url'],
                    license_info=updated_metadata['license_info'],
                    is_default=current_model.is_default,  # Preserve default status
                    size_mb=current_model.size_mb,  # Preserve size
                    installed_date=current_model.installed_date,  # Preserve install date
                    last_used=current_model.last_used,  # Preserve usage data
                    usage_count=current_model.usage_count  # Preserve usage count
                )

                # Save to database using the safe update method
                if self.model_manager.db.update_model(current_model.name, updated_model):
                    QMessageBox.information(self.sidebar, "Success",
                                          f"Model '{model_name}' parameters updated successfully!")

                    # Refresh the UI
                    self._refresh_installed_models()
                else:
                    QMessageBox.critical(self.sidebar, "Error", "Failed to update model parameters!")

        except Exception as e:
            QMessageBox.critical(self.sidebar, "Error", f"Failed to edit model parameters: {str(e)}")

    def _delete_model(self, model_name: str):
        """Delete a model."""
        reply = QMessageBox.question(self.sidebar, "Confirm Delete",
                                   f"Are you sure you want to delete model '{model_name}'?",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            # Save state for undo before deletion
            self.model_manager._save_state_for_undo("delete_model", {"model_name": model_name})

            success = self.model_manager.delete_model(model_name)
            if success:
                QMessageBox.information(self.sidebar, "Deleted", f"Model '{model_name}' deleted successfully!")
                self._refresh_installed_models()
                self._update_undo_redo_buttons()
            else:
                QMessageBox.critical(self.sidebar, "Error", f"Failed to delete model '{model_name}'!")

    def _set_default_model(self, model_name: str):
        """Set model as default."""
        # Save state for undo before changing default
        self.model_manager._save_state_for_undo("set_default_model", {"model_name": model_name})

        success = self.model_manager.set_default_model(model_name)
        if success:
            QMessageBox.information(self.sidebar, "Set Default", f"Model '{model_name}' set as default!")
            self._refresh_installed_models()
            self._update_undo_redo_buttons()
        else:
            QMessageBox.critical(self.sidebar, "Error", f"Failed to set model '{model_name}' as default!")

    def undo_operation(self):
        """Undo the last operation."""
        success, message = self.model_manager.undo_last_operation()
        if success:
            QMessageBox.information(self.sidebar, "Undo Successful", message)
            self._refresh_installed_models()
            self._update_undo_redo_buttons()
        else:
            QMessageBox.warning(self.sidebar, "Undo Failed", message)

    def redo_operation(self):
        """Redo the last undone operation."""
        success, message = self.model_manager.redo_last_operation()
        if success:
            QMessageBox.information(self.sidebar, "Redo Successful", message)
            self._refresh_installed_models()
            self._update_undo_redo_buttons()
        else:
            QMessageBox.warning(self.sidebar, "Redo Failed", message)

    def _update_undo_redo_buttons(self):
        """Update the enabled state of undo/redo buttons."""
        self.undo_btn.setEnabled(self.model_manager.can_undo())
        self.redo_btn.setEnabled(self.model_manager.can_redo())

        # Update button text to show operation description
        operation_desc = self.model_manager.get_last_operation_description()
        if operation_desc:
            self.undo_btn.setText(f'↶ Undo {operation_desc}')
        else:
            self.undo_btn.setText('↶ Undo')

    def export_model_metadata(self):
        """Export model metadata for sharing."""
        # Get list of installed models for selection
        installed_models = self.model_manager.get_installed_models()
        if not installed_models:
            QMessageBox.warning(self.sidebar, "No Models", "No models installed to export.")
            return

        # Create a simple selection dialog
        from PyQt5.QtWidgets import QInputDialog
        model_names = [model.name for model in installed_models]
        model_name, ok = QInputDialog.getItem(
            self, "Select Model", "Choose model to export:", model_names, 0, False
        )

        if not ok or not model_name:
            return

        # Get export file path
        export_path, _ = QFileDialog.getSaveFileName(
            self, "Export Model Metadata", f"{model_name}_metadata.json",
            "JSON Files (*.json);;All Files (*)"
        )

        if not export_path:
            return

        # Export metadata
        success, message = self.model_manager.export_model_metadata(model_name, export_path)
        if success:
            QMessageBox.information(self.sidebar, "Export Successful", message)
        else:
            QMessageBox.critical(self.sidebar, "Export Failed", message)

    def import_model_metadata(self):
        """Import model metadata."""
        # Get import file path
        import_path, _ = QFileDialog.getOpenFileName(
            self, "Import Model Metadata", "", "JSON Files (*.json);;All Files (*)"
        )

        if not import_path:
            return

        # Import metadata
        success, message = self.model_manager.import_model_metadata(import_path)
        if success:
            QMessageBox.information(self.sidebar, "Import Successful", message)
            self._refresh_installed_models()
        else:
            QMessageBox.critical(self.sidebar, "Import Failed", message)

    def clear_database(self):
        """Clear all database entries to start fresh."""
        reply = QMessageBox.question(
            self.sidebar, "Clear Database",
            "Are you sure you want to clear ALL database entries?\n\n"
            "This will delete all installed models, settings, and operation history.\n"
            "This action cannot be undone!",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No  # Default to No
        )

        if reply == QMessageBox.Yes:
            # Confirm again with text input
            confirm_text, ok = QInputDialog.getText(
                self.sidebar, "Confirm Clear Database",
                "Type 'CLEAR' to confirm clearing all database entries:",
                QLineEdit.Normal, ""
            )

            if ok and confirm_text.upper() == "CLEAR":
                # Clear the database
                success = self.model_manager.clear_database()
                if success:
                    QMessageBox.information(self.sidebar, "Database Cleared",
                                          "All database entries have been cleared successfully!\n\n"
                                          "The application will now start fresh.")
                    self._refresh_installed_models()
                    self._update_undo_redo_buttons()
                else:
                    QMessageBox.critical(self.sidebar, "Clear Failed",
                                       "Failed to clear the database. Please try again.")
            else:
                QMessageBox.information(self.sidebar, "Cancelled", "Database clear operation cancelled.")
        else:
            QMessageBox.information(self.sidebar, "Cancelled", "Database clear operation cancelled.")

    def _clear_layout(self, layout):
        """Clear all widgets from layout."""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()


class ModelInstallDialog(QMessageBox):
    """Dialog for collecting model metadata during installation."""

    def __init__(self, model_name: str, model_path: str, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.model_path = model_path

        self.setWindowTitle("Install Model")
        self.setText(f"Install model '{model_name}'?")
        self.setInformativeText(f"Path: {model_path}\n\nPlease fill in the model details below and click 'Install' to proceed.")

        # Add custom widgets for metadata collection
        self._setup_metadata_widgets()

        # Add standard buttons (Install first, then Cancel)
        install_button = self.addButton("Install", QMessageBox.AcceptRole)
        self.addButton("Cancel", QMessageBox.RejectRole)

        # Set default button to Install
        install_button.setDefault(True)
        install_button.setFocus()

    def _setup_metadata_widgets(self):
        """Set up widgets for collecting model metadata."""
        # Create a widget to hold our custom controls
        widget = QWidget()
        layout = QVBoxLayout()

        # Display Name
        display_name_label = QLabel("Display Name:")
        layout.addWidget(display_name_label)
        self.display_name_edit = QLineEdit()
        self.display_name_edit.setText(self.model_name)  # Pre-fill with filename
        self.display_name_edit.setPlaceholderText("User-friendly name for this model")
        layout.addWidget(self.display_name_edit)

        # Description
        desc_label = QLabel("Description:")
        layout.addWidget(desc_label)
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(60)
        self.description_edit.setPlaceholderText("Enter a description for this model...")
        layout.addWidget(self.description_edit)

        # Categories
        cat_label = QLabel("Categories (select multiple):")
        layout.addWidget(cat_label)
        self.category_list = QListWidget()
        self.category_list.setMaximumHeight(100)
        self.category_list.setSelectionMode(QListWidget.MultiSelection)

        # Add category options
        for category in ModelCategory:
            item = QListWidgetItem(category.value.title())
            item.setData(1, category.value)  # Store the enum value
            self.category_list.addItem(item)

        layout.addWidget(self.category_list)

        # Usage Notes
        usage_label = QLabel("Usage Notes:")
        layout.addWidget(usage_label)
        self.usage_edit = QTextEdit()
        self.usage_edit.setMaximumHeight(60)
        self.usage_edit.setPlaceholderText("Any special usage notes or tips...")
        layout.addWidget(self.usage_edit)

        # Source URL
        source_label = QLabel("Source URL:")
        layout.addWidget(source_label)
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("https://...")
        layout.addWidget(self.source_edit)

        # License Info
        license_label = QLabel("License Information:")
        layout.addWidget(license_label)
        self.license_edit = QLineEdit()
        self.license_edit.setPlaceholderText("License type or attribution...")
        layout.addWidget(self.license_edit)

        widget.setLayout(layout)
        self.layout().addWidget(widget, 1, 0, 1, self.layout().columnCount())

    def get_metadata(self) -> dict:
        """Get the collected metadata."""
        # Get selected categories
        selected_categories = []
        for i in range(self.category_list.count()):
            item = self.category_list.item(i)
            if item.isSelected():
                selected_categories.append(item.data(1))  # Get the enum value

        return {
            'display_name': self.display_name_edit.text().strip(),
            'description': self.description_edit.toPlainText().strip(),
            'categories': selected_categories,
            'usage_notes': self.usage_edit.toPlainText().strip(),
            'source_url': self.source_edit.text().strip(),
            'license_info': self.license_edit.text().strip()
        }


class ModelEditDialog(QMessageBox):
    """Dialog for editing existing model metadata."""

    def __init__(self, model: ModelInfo, parent=None):
        super().__init__(parent)
        self.model = model

        self.setWindowTitle("Edit Model Parameters")
        self.setText(f"Edit parameters for '{model.name}'")
        self.setInformativeText("Modify the model details below and click 'Save' to update.")

        # Add custom widgets for metadata editing
        self._setup_edit_widgets()

        # Add standard buttons (Save first, then Cancel)
        save_button = self.addButton("Save", QMessageBox.AcceptRole)
        self.addButton("Cancel", QMessageBox.RejectRole)

        # Set default button to Save
        save_button.setDefault(True)
        save_button.setFocus()

    def _setup_edit_widgets(self):
        """Set up widgets for editing model metadata."""
        # Create a widget to hold our custom controls
        widget = QWidget()
        layout = QVBoxLayout()

        # Model Name (unique identifier)
        name_label = QLabel("Model Name (unique identifier):")
        layout.addWidget(name_label)
        self.name_edit = QLineEdit()
        self.name_edit.setText(str(self.model.name))
        self.name_edit.setPlaceholderText("Unique name for this model")
        layout.addWidget(self.name_edit)

        # Display Name
        display_name_label = QLabel("Display Name:")
        layout.addWidget(display_name_label)
        self.display_name_edit = QLineEdit()
        self.display_name_edit.setText(str(self.model.display_name or ""))
        self.display_name_edit.setPlaceholderText("User-friendly name for this model")
        layout.addWidget(self.display_name_edit)

        # Model Type
        model_type_label = QLabel("Model Type:")
        layout.addWidget(model_type_label)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItem("Stable Diffusion v1.4", ModelType.STABLE_DIFFUSION_V1_4)
        self.model_type_combo.addItem("Stable Diffusion v1.5", ModelType.STABLE_DIFFUSION_V1_5)
        self.model_type_combo.addItem("Stable Diffusion XL", ModelType.STABLE_DIFFUSION_XL)

        # Set current model type
        if self.model.model_type:
            if self.model.model_type == ModelType.STABLE_DIFFUSION_V1_4:
                self.model_type_combo.setCurrentIndex(0)
            elif self.model.model_type == ModelType.STABLE_DIFFUSION_V1_5:
                self.model_type_combo.setCurrentIndex(1)
            elif self.model.model_type == ModelType.STABLE_DIFFUSION_XL:
                self.model_type_combo.setCurrentIndex(2)
        else:
            self.model_type_combo.setCurrentIndex(0)  # Default to v1.4

        layout.addWidget(self.model_type_combo)

        # Description
        desc_label = QLabel("Description:")
        layout.addWidget(desc_label)
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(60)
        self.description_edit.setPlainText(str(self.model.description or ""))
        self.description_edit.setPlaceholderText("Enter a description for this model...")
        layout.addWidget(self.description_edit)

        # Categories
        cat_label = QLabel("Categories (select multiple):")
        layout.addWidget(cat_label)
        self.category_list = QListWidget()
        self.category_list.setMaximumHeight(100)
        self.category_list.setSelectionMode(QListWidget.MultiSelection)

        # Add category options and pre-select current categories
        current_categories = set(self.model.categories) if self.model.categories else set()
        for category in ModelCategory:
            item = QListWidgetItem(category.value.title())
            item.setData(1, category.value)  # Store the enum value
            if category in current_categories:
                item.setSelected(True)
            self.category_list.addItem(item)

        layout.addWidget(self.category_list)

        # Usage Notes
        usage_label = QLabel("Usage Notes:")
        layout.addWidget(usage_label)
        self.usage_edit = QTextEdit()
        self.usage_edit.setMaximumHeight(60)
        self.usage_edit.setPlainText(str(self.model.usage_notes or ""))
        self.usage_edit.setPlaceholderText("Any special usage notes or tips...")
        layout.addWidget(self.usage_edit)

        # Source URL
        source_label = QLabel("Source URL:")
        layout.addWidget(source_label)
        self.source_edit = QLineEdit()
        self.source_edit.setText(str(self.model.source_url or ""))
        self.source_edit.setPlaceholderText("https://...")
        layout.addWidget(self.source_edit)

        # License Info
        license_label = QLabel("License Information:")
        layout.addWidget(license_label)
        self.license_edit = QLineEdit()
        self.license_edit.setText(str(self.model.license_info or ""))
        self.license_edit.setPlaceholderText("License type or attribution...")
        layout.addWidget(self.license_edit)

        widget.setLayout(layout)
        self.layout().addWidget(widget, 1, 0, 1, self.layout().columnCount())

    def get_metadata(self) -> dict:
        """Get the edited metadata."""
        # Get selected categories
        selected_categories = []
        for i in range(self.category_list.count()):
            item = self.category_list.item(i)
            if item.isSelected():
                selected_categories.append(item.data(1))  # Get the enum value

        # Get selected model type
        current_index = self.model_type_combo.currentIndex()
        model_type = self.model_type_combo.itemData(current_index)

        return {
            'name': self.name_edit.text().strip(),
            'display_name': self.display_name_edit.text().strip(),
            'model_type': model_type,
            'description': self.description_edit.toPlainText().strip(),
            'categories': selected_categories,
            'usage_notes': self.usage_edit.toPlainText().strip(),
            'source_url': self.source_edit.text().strip(),
            'license_info': self.license_edit.text().strip()
        }
