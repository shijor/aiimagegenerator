"""
Model management panel with sidebar and main area - refactored to use modular components.
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QScrollArea, QGroupBox, QMessageBox, QInputDialog
from PyQt5.QtCore import pyqtSignal, Qt

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.model_management_service import ModelManagementService
from services.model_manager import ModelManager
from ui.components.model_list_item import ModelListItem, LoRAListItem, ProgressDisplay
from ui.dialogs.model_install_dialog import ModelInstallDialog
from ui.dialogs.model_edit_dialog import ModelEditDialog
from ui.dialogs.lora_install_dialog import LoRAInstallDialog
from ui.dialogs.lora_edit_dialog import LoRAEditDialog


class ModelManagementPanel(QWidget):
    """Panel for model management functionality - refactored with modular components."""

    model_installed = pyqtSignal()  # Emitted when a model is installed

    def __init__(self, model_manager: ModelManager):
        super().__init__()
        self.service = ModelManagementService(model_manager)

        # Load saved default folder if available
        self.selected_folder = self.service.get_default_folder()

        # Create sidebar and main area
        self.sidebar = self._create_sidebar()
        self.main_area = self._create_main_area()

        # Update UI with loaded default folder
        self._update_folder_display()

        # Refresh installed models list after UI is fully initialized
        self._refresh_installed_models()
        self._update_undo_redo_buttons()

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

        # Scroll area for available models - compact size based on content
        available_scroll = QScrollArea()
        available_scroll.setWidgetResizable(True)
        available_scroll.setMinimumHeight(80)  # Minimum height - reduced for more compact layout
        available_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)  # Use only needed space

        available_widget = QWidget()
        self.available_models_layout = QVBoxLayout()
        self.available_models_layout.setContentsMargins(0, 0, 0, 0)
        self.available_models_layout.setSpacing(0)  # No spacing between rows
        self.available_models_layout.addWidget(QLabel("No models scanned yet. Select a folder and click 'Scan Folder'."))

        available_widget.setLayout(self.available_models_layout)
        available_scroll.setWidget(available_widget)
        available_layout.addWidget(available_scroll, alignment=Qt.AlignTop)

        available_group.setLayout(available_layout)
        layout.addWidget(available_group, stretch=0)  # Use only needed space, no expansion

        # Installed Models & LoRAs Section
        installed_group = QGroupBox("Installed Models & LoRAs")
        installed_layout = QVBoxLayout()

        # Scroll area for installed models and LoRAs - compact size based on content
        installed_scroll = QScrollArea()
        installed_scroll.setWidgetResizable(True)
        installed_scroll.setMinimumHeight(80)  # Minimum height - reduced for more compact layout
        installed_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)  # Use only needed space

        installed_widget = QWidget()
        self.installed_models_layout = QVBoxLayout()
        self.installed_models_layout.setContentsMargins(0, 0, 0, 0)
        self.installed_models_layout.setSpacing(0)  # No spacing between rows

        # Add refresh button at the top of the content
        refresh_layout = QHBoxLayout()
        refresh_layout.setContentsMargins(0, 0, 0, 0)
        refresh_label = QLabel("🔄 Refresh List")
        refresh_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #1976D2; margin-bottom: 5px;")
        refresh_layout.addWidget(refresh_label)

        self.refresh_installed_btn = QPushButton('🔄')
        self.refresh_installed_btn.setMaximumWidth(30)
        self.refresh_installed_btn.setToolTip('Refresh installed models and LoRAs list')
        self.refresh_installed_btn.clicked.connect(self._refresh_installed_models)
        refresh_layout.addWidget(self.refresh_installed_btn)

        refresh_layout.addStretch()
        refresh_widget = QWidget()
        refresh_widget.setLayout(refresh_layout)
        refresh_widget.setFixedHeight(25)
        self.installed_models_layout.addWidget(refresh_widget)

        installed_widget.setLayout(self.installed_models_layout)
        installed_scroll.setWidget(installed_widget)
        installed_layout.addWidget(installed_scroll, alignment=Qt.AlignTop)

        installed_group.setLayout(installed_layout)
        layout.addWidget(installed_group)

        # Installation Progress Section (moved to bottom)
        progress_group = QGroupBox("Installation Progress")
        progress_layout = QVBoxLayout()
        progress_layout.setContentsMargins(10, 5, 10, 5)  # Minimal margins
        progress_layout.setSpacing(2)  # Minimal spacing

        self.progress_label = QLabel("Ready to install models")
        self.progress_label.setStyleSheet("font-size: 11px; margin: 0px; padding: 0px;")
        progress_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)  # Hidden by default
        self.progress_bar.setFixedHeight(16)  # Minimal height for progress bar
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #e0e0e0;
                border-radius: 2px;
                text-align: center;
                font-size: 10px;
                margin: 0px;
                padding: 0px;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 1px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)

        progress_group.setLayout(progress_layout)
        progress_group.setFixedHeight(50)  # Minimal fixed height to fit progress bar
        progress_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                margin-top: 5px;
                padding-top: 5px;
                font-size: 11px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 5px;
                padding: 0 3px 0 3px;
                font-size: 10px;
                font-weight: bold;
            }
        """)
        layout.addWidget(progress_group, stretch=0)  # Use only needed space, no expansion

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
            success = self.service.save_default_folder(self.selected_folder)
            if success:
                QMessageBox.information(self.sidebar, "Success", "Default model folder saved!")
            else:
                QMessageBox.critical(self.sidebar, "Error", "Failed to save default model folder!")
        else:
            QMessageBox.warning(self.sidebar, "Warning", "Please select a folder first!")

    def scan_models(self):
        """Scan selected folder for models and LoRAs and show available items with install buttons."""
        if not hasattr(self, 'selected_folder'):
            QMessageBox.warning(self.sidebar, "No Folder", "Please select a folder first.")
            return

        # Clear previous results
        self._clear_layout(self.available_models_layout)
        self.available_models_layout.addWidget(QLabel("Scanning for models and LoRAs..."))

        # Scan for model files and LoRA files
        model_files = self.model_manager.scan_models_in_folder(self.selected_folder)
        lora_files = self.model_manager.scan_loras_in_folder(self.selected_folder)

        # Separate valid and unrecognized models
        valid_models = [m for m in model_files if m.get('validation_status') == 'valid']
        unrecognized_models = [m for m in model_files if m.get('validation_status') == 'unrecognized']

        # Update UI with results
        self._clear_layout(self.available_models_layout)

        total_files = len(model_files) + len(lora_files)
        if total_files == 0:
            self.available_models_layout.addWidget(QLabel("No model or LoRA files found."))
        else:
            # Show valid models first
            if valid_models:
                models_header = QLabel("📦 VALID MODELS")
                models_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #1976D2; margin-top: 10px; margin-bottom: 5px;")
                self.available_models_layout.addWidget(models_header)

                for model_info in valid_models:
                    model_widget = self._create_available_model_widget(model_info)
                    self.available_models_layout.addWidget(model_widget)

            # Show unrecognized models
            if unrecognized_models:
                if valid_models:  # Add separator if we have valid models above
                    separator = QWidget()
                    separator.setFixedHeight(20)
                    self.available_models_layout.addWidget(separator)

                unrecognized_header = QLabel("❓ UNRECOGNIZED MODELS")
                unrecognized_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #FF9800; margin-top: 10px; margin-bottom: 5px;")
                self.available_models_layout.addWidget(unrecognized_header)

                # Add explanation
                explanation = QLabel("These models were found but failed validation. They may not be compatible or may need manual review.")
                explanation.setStyleSheet("color: #666; font-size: 11px; font-style: italic; margin-bottom: 5px;")
                explanation.setWordWrap(True)
                self.available_models_layout.addWidget(explanation)

                for model_info in unrecognized_models:
                    model_widget = self._create_unrecognized_model_widget(model_info)
                    self.available_models_layout.addWidget(model_widget)

            # Show available LoRAs
            if lora_files:
                if valid_models or unrecognized_models:  # Add separator if we have models above
                    separator = QWidget()
                    separator.setFixedHeight(20)
                    self.available_models_layout.addWidget(separator)

                loras_header = QLabel("🎭 LoRA ADAPTERS")
                loras_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #FF6B35; margin-top: 10px; margin-bottom: 5px;")
                self.available_models_layout.addWidget(loras_header)

                for lora_info in lora_files:
                    lora_widget = self._create_available_lora_widget(lora_info)
                    self.available_models_layout.addWidget(lora_widget)

        # Dynamically adjust scroll area height based on number of items
        self._adjust_available_models_height(total_files)

    def _adjust_available_models_height(self, num_items: int):
        """Dynamically adjust the available models scroll area height based on number of items."""
        # Calculate dynamic height: 35px per item + 60px padding, max 400px, min 80px
        dynamic_height = min(num_items * 35 + 60, 400)
        dynamic_height = max(dynamic_height, 80)

        # Find the scroll area in the main area layout
        main_area = self.main_area
        if main_area and main_area.layout():
            # The available group is the first item in the main area layout
            available_group = main_area.layout().itemAt(0).widget()
            if available_group and hasattr(available_group, 'layout'):
                # The scroll area is the first item in the available group layout
                available_layout = available_group.layout()
                if available_layout and available_layout.count() > 0:
                    scroll_area = available_layout.itemAt(0).widget()
                    if scroll_area and hasattr(scroll_area, 'setMinimumHeight'):
                        scroll_area.setMinimumHeight(dynamic_height)

    def _adjust_installed_models_height(self, num_items: int):
        """Dynamically adjust the installed models scroll area height based on number of items."""
        # Calculate dynamic height: 28px per item + 60px padding, max 400px, min 80px
        # Use 28px since installed items are more compact (28px height vs 30px for available)
        dynamic_height = min(num_items * 28 + 60, 400)
        dynamic_height = max(dynamic_height, 80)

        # Find the scroll area in the main area layout
        main_area = self.main_area
        if main_area and main_area.layout():
            # The installed group is the second item in the main area layout (after available)
            installed_group = main_area.layout().itemAt(1).widget()
            if installed_group and hasattr(installed_group, 'layout'):
                # The scroll area is the last item in the installed group layout
                installed_layout = installed_group.layout()
                if installed_layout and installed_layout.count() > 0:
                    scroll_area = installed_layout.itemAt(installed_layout.count() - 1).widget()
                    if scroll_area and hasattr(scroll_area, 'setMinimumHeight'):
                        scroll_area.setMinimumHeight(dynamic_height)

    def _create_available_model_widget(self, model_info: dict) -> QWidget:
        """Create compact table row widget for available model."""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(10)

        # Column 1: Icon + Name (flexible width)
        name_layout = QHBoxLayout()
        name_layout.setSpacing(6)

        icon_label = QLabel("📦")
        icon_label.setStyleSheet("font-size: 14px;")
        icon_label.setMinimumWidth(20)
        name_layout.addWidget(icon_label)

        name_label = QLabel(model_info['name'])
        name_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #1565C0;")
        name_label.setToolTip(model_info['name'])  # Full name on hover
        name_label.setMinimumWidth(150)  # Ensure minimum width for text
        name_layout.addWidget(name_label)

        name_layout.addStretch()
        layout.addLayout(name_layout, stretch=4)  # Give name column more space

        # Column 2: Type (fixed width)
        type_text = model_info.get('model_type', 'Unknown')
        type_label = QLabel(type_text)
        type_label.setStyleSheet("""
            font-size: 11px;
            color: #666;
            background-color: #f0f0f0;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: 500;
        """)
        type_label.setFixedWidth(120)
        type_label.setMinimumWidth(120)
        layout.addWidget(type_label)

        # Column 3: Size (fixed width)
        size_label = QLabel(f"{model_info['size_mb']:.1f} MB")
        size_label.setStyleSheet("color: #555; font-size: 11px; font-weight: 500;")
        size_label.setFixedWidth(70)
        size_label.setMinimumWidth(70)
        size_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(size_label)

        # Column 4: Path (flexible width, truncated)
        path_text = model_info['path']
        if len(path_text) > 50:
            path_text = "..." + path_text[-47:]
        path_label = QLabel(path_text)
        path_label.setStyleSheet("""
            color: #777;
            font-size: 10px;
            font-family: 'Segoe UI', monospace;
        """)
        path_label.setToolTip(model_info['path'])  # Full path on hover
        path_label.setMinimumWidth(200)  # Ensure minimum width for path text
        path_label.setWordWrap(False)  # Prevent wrapping
        layout.addWidget(path_label, stretch=3)

        # Column 5: Install button (fixed width)
        install_btn = QPushButton('Install')
        install_btn.setProperty("model_name", model_info['name'])
        install_btn.setProperty("model_path", model_info['path'])
        install_btn.clicked.connect(self._on_install_clicked)
        install_btn.setFixedSize(70, 24)
        install_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
                padding: 2px 8px;
            }
            QPushButton:hover {
                background-color: #42A5F5;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """)
        layout.addWidget(install_btn)

        # Set layout and compact styling
        widget.setLayout(layout)
        widget.setStyleSheet("""
            QWidget {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background-color: #ffffff;
                margin: 1px;
            }
            QWidget:hover {
                background-color: #f8f9ff;
                border-color: #2196F3;
            }
        """)
        widget.setMinimumHeight(32)  # Slightly taller to accommodate text
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        return widget

    def _create_available_lora_widget(self, lora_info: dict) -> QWidget:
        """Create compact table row widget for available LoRA."""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(10)

        # Column 1: Icon + Name (flexible width)
        name_layout = QHBoxLayout()
        name_layout.setSpacing(6)

        icon_label = QLabel("🎭")
        icon_label.setStyleSheet("font-size: 14px;")
        name_layout.addWidget(icon_label)

        name_label = QLabel(lora_info['name'])
        name_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #E65100;")
        name_label.setToolTip(lora_info['name'])  # Full name on hover
        name_layout.addWidget(name_label)

        name_layout.addStretch()
        layout.addLayout(name_layout, stretch=3)  # Give name column more space

        # Column 2: Type (fixed width)
        type_label = QLabel("LoRA Adapter")
        type_label.setStyleSheet("""
            font-size: 11px;
            color: #666;
            background-color: #fff3e0;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: 500;
        """)
        type_label.setFixedWidth(120)
        layout.addWidget(type_label)

        # Column 3: Size (fixed width)
        size_label = QLabel(f"{lora_info['size_mb']:.1f} MB")
        size_label.setStyleSheet("color: #555; font-size: 11px; font-weight: 500;")
        size_label.setFixedWidth(70)
        layout.addWidget(size_label)

        # Column 4: Path (flexible width, truncated)
        path_text = lora_info['path']
        if len(path_text) > 40:
            path_text = "..." + path_text[-37:]
        path_label = QLabel(path_text)
        path_label.setStyleSheet("""
            color: #777;
            font-size: 10px;
            font-family: 'Segoe UI', monospace;
        """)
        path_label.setToolTip(lora_info['path'])  # Full path on hover
        layout.addWidget(path_label, stretch=2)

        # Column 5: Install button (fixed width)
        install_btn = QPushButton('Install')
        install_btn.setProperty("lora_name", lora_info['name'])
        install_btn.setProperty("lora_path", lora_info['path'])
        install_btn.clicked.connect(self._on_install_lora_clicked)
        install_btn.setFixedSize(70, 24)
        install_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF6B35;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
                padding: 2px 8px;
            }
            QPushButton:hover {
                background-color: #FF8A65;
            }
            QPushButton:pressed {
                background-color: #E64A19;
            }
        """)
        layout.addWidget(install_btn)

        # Set layout and compact styling
        widget.setLayout(layout)
        widget.setStyleSheet("""
            QWidget {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background-color: #ffffff;
                margin: 1px;
            }
            QWidget:hover {
                background-color: #fff8f8;
                border-color: #FF6B35;
            }
        """)
        widget.setFixedHeight(30)

        return widget

    def _create_unrecognized_model_widget(self, model_info: dict) -> QWidget:
        """Create compact table row widget for unrecognized model."""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(10)

        # Column 1: Icon + Name (flexible width)
        name_layout = QHBoxLayout()
        name_layout.setSpacing(6)

        icon_label = QLabel("❓")
        icon_label.setStyleSheet("font-size: 14px;")
        name_layout.addWidget(icon_label)

        name_label = QLabel(model_info['name'])
        name_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #FF9800;")
        name_label.setToolTip(model_info['name'])  # Full name on hover
        name_layout.addWidget(name_label)

        name_layout.addStretch()
        layout.addLayout(name_layout, stretch=3)  # Give name column more space

        # Column 2: Type (fixed width)
        type_text = model_info.get('model_type', 'Unknown')
        type_label = QLabel(type_text)
        type_label.setStyleSheet("""
            font-size: 11px;
            color: #666;
            background-color: #fff3e0;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: 500;
        """)
        type_label.setFixedWidth(120)
        layout.addWidget(type_label)

        # Column 3: Size (fixed width)
        size_label = QLabel(f"{model_info['size_mb']:.1f} MB")
        size_label.setStyleSheet("color: #555; font-size: 11px; font-weight: 500;")
        size_label.setFixedWidth(70)
        layout.addWidget(size_label)

        # Column 4: Reason (flexible width, truncated)
        reason_text = model_info.get('validation_reason', 'Unknown reason')
        if len(reason_text) > 40:
            reason_text = "..." + reason_text[-37:]
        reason_label = QLabel(reason_text)
        reason_label.setStyleSheet("""
            color: #FF9800;
            font-size: 10px;
            font-style: italic;
        """)
        reason_label.setToolTip(model_info.get('validation_reason', 'Unknown reason'))  # Full reason on hover
        layout.addWidget(reason_label, stretch=2)

        # Column 5: Action buttons (fixed width)
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(2)

        # Info button
        info_btn = QPushButton('ℹ️')
        info_btn.setProperty("model_name", model_info['name'])
        info_btn.setProperty("model_path", model_info['path'])
        info_btn.setProperty("validation_reason", model_info.get('validation_reason', 'Unknown reason'))
        info_btn.clicked.connect(self._on_unrecognized_info_clicked)
        info_btn.setFixedSize(24, 24)
        info_btn.setToolTip('Show validation details')
        info_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FFB74D;
            }
            QPushButton:pressed {
                background-color: #F57C00;
            }
        """)
        buttons_layout.addWidget(info_btn)

        # Install button
        install_btn = QPushButton('Install')
        install_btn.setProperty("model_name", model_info['name'])
        install_btn.setProperty("model_path", model_info['path'])
        install_btn.setProperty("validation_reason", model_info.get('validation_reason', 'Unknown reason'))
        install_btn.clicked.connect(self._on_install_clicked)
        install_btn.setFixedSize(50, 24)
        install_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF6B35;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 10px;
                font-weight: bold;
                padding: 2px 4px;
            }
            QPushButton:hover {
                background-color: #FF8A65;
            }
            QPushButton:pressed {
                background-color: #E64A19;
            }
        """)
        buttons_layout.addWidget(install_btn)

        buttons_widget.setLayout(buttons_layout)
        layout.addWidget(buttons_widget)

        # Set layout and compact styling
        widget.setLayout(layout)
        widget.setStyleSheet("""
            QWidget {
                border: 1px solid #FF9800;
                border-radius: 4px;
                background-color: #fff8e1;
                margin: 1px;
            }
            QWidget:hover {
                background-color: #fffde7;
                border-color: #F57C00;
            }
        """)
        widget.setFixedHeight(30)

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

    def _on_unrecognized_info_clicked(self):
        """Handle info button click for unrecognized models."""
        button = self.sender()
        if button:
            model_name = button.property("model_name")
            model_path = button.property("model_path")
            validation_reason = button.property("validation_reason")

            # Show info dialog
            info_text = f"""
<b>Model:</b> {model_name}<br/>
<b>Path:</b> {model_path}<br/>
<b>Status:</b> Unrecognized<br/>
<b>Reason:</b> {validation_reason}<br/><br/>

This model was found but failed validation. It may not be compatible with the current image generation system or may require manual review before installation.
"""
            QMessageBox.information(self.sidebar, "Unrecognized Model Details", info_text)

    def _on_install_clicked(self):
        """Handle install button click."""
        button = self.sender()
        if button:
            model_name = button.property("model_name")
            model_path = button.property("model_path")
            # Check if this is an unrecognized model (has validation_reason property)
            is_unrecognized = button.property("validation_reason") is not None
            if model_name and model_path:
                self._install_model(model_name, model_path, skip_validation=is_unrecognized)

    def _on_install_lora_clicked(self):
        """Handle LoRA install button click."""
        button = self.sender()
        if button:
            lora_name = button.property("lora_name")
            lora_path = button.property("lora_path")
            if lora_name and lora_path:
                self._install_lora(lora_name, lora_path)

    def _install_model(self, name: str, path: str, skip_validation: bool = False):
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
            print(f"INSTALLATION: Dialog result: {result} (accepted: {dialog.accepted})")

            # Check if the Install button was clicked
            if dialog.accepted:
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
                    progress_callback=update_progress,
                    skip_validation=skip_validation
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

    def _install_lora(self, name: str, path: str):
        """Install a LoRA adapter with enhanced metadata collection."""
        try:
            print(f"LORA INSTALLATION: Starting installation for LoRA '{name}' from path '{path}'")

            # Validate inputs
            if not name or not path:
                QMessageBox.critical(self.sidebar, "Error", "Invalid LoRA name or path")
                return

            if not os.path.exists(path):
                QMessageBox.critical(self.sidebar, "Error", f"LoRA file does not exist: {path}")
                return

            # Show installation dialog for metadata collection
            dialog = LoRAInstallDialog(name, path, self)
            print("LORA INSTALLATION: Dialog created, showing...")

            result = dialog.exec_()
            print(f"LORA INSTALLATION: Dialog result: {result}")

            # Check if the Install button was clicked
            if dialog.accepted:
                print("LORA INSTALLATION: Dialog accepted, getting metadata...")

                # Get metadata from dialog
                metadata = dialog.get_metadata()
                print(f"LORA INSTALLATION: Metadata collected: {metadata}")

                # Show progress in the panel
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)
                self.progress_label.setText("Starting LoRA installation...")
                print("LORA INSTALLATION: Progress UI initialized")

                # Progress callback function
                def update_progress(message: str, percentage: int):
                    print(f"LORA INSTALLATION: Progress update - {percentage}% - {message}")
                    self.progress_label.setText(message)
                    self.progress_bar.setValue(percentage)
                    # Process events to keep UI responsive
                    from PyQt5.QtWidgets import QApplication
                    QApplication.processEvents()

                print("LORA INSTALLATION: Calling model_manager.install_lora...")

                # Install LoRA with metadata and progress callback
                success, message = self.model_manager.install_lora(
                    name, path,
                    display_name=metadata.get('display_name', ''),
                    base_model_type=metadata.get('base_model_type'),
                    categories=metadata.get('categories', []),
                    description=metadata.get('description', ''),
                    trigger_words=metadata.get('trigger_words', []),
                    usage_notes=metadata.get('usage_notes', ''),
                    source_url=metadata.get('source_url'),
                    license_info=metadata.get('license_info'),
                    default_scaling=metadata.get('default_scaling', 1.0),
                    progress_callback=update_progress
                )

                print(f"LORA INSTALLATION: LoRA installation result - Success: {success}, Message: {message}")

                if success:
                    self.progress_label.setText("LoRA installation completed successfully!")
                    QMessageBox.information(self.sidebar, "Success", message)
                    print("LORA INSTALLATION: Installation completed successfully!")
                else:
                    self.progress_label.setText("LoRA installation failed!")
                    QMessageBox.critical(self.sidebar, "Installation Failed", message)
                    print(f"LORA INSTALLATION: Installation failed with message: {message}")

            else:
                print("LORA INSTALLATION: Dialog was cancelled by user")
                # User cancelled - do nothing

        except Exception as e:
            print(f"LORA INSTALLATION: Exception occurred: {str(e)}")
            import traceback
            traceback.print_exc()
            self.progress_label.setText("LoRA installation error!")
            QMessageBox.critical(self.sidebar, "Error", f"Unexpected error during LoRA installation: {str(e)}")
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
        """Refresh the installed models and LoRAs list."""
        print("DEBUG: Starting _refresh_installed_models")
        self._clear_layout(self.installed_models_layout)

        # Get both models and LoRAs
        installed_models = self.model_manager.get_installed_models()
        installed_loras = self.model_manager.get_installed_loras()

        print(f"DEBUG: Retrieved {len(installed_models)} models and {len(installed_loras)} LoRAs from database")

        total_items = len(installed_models) + len(installed_loras)

        if total_items == 0:
            self.installed_models_layout.addWidget(QLabel("No models or LoRAs installed."))
        else:
            # Sort by installation date (most recent first)
            all_items = []

            # Add models with type indicator
            for model in installed_models:
                print(f"DEBUG: Processing model: {model.name}, display_name: {model.display_name}, description: {model.description}")
                all_items.append(('model', model))

            # Add LoRAs with type indicator
            for lora in installed_loras:
                all_items.append(('lora', lora))

            # Sort by installation date (most recent first), handling None values
            all_items.sort(key=lambda x: getattr(x[1], 'installed_date', None) or '', reverse=True)

            # Create widgets for each item
            for item_type, item in all_items:
                if item_type == 'model':
                    widget = self._create_installed_model_widget(item)
                    print(f"DEBUG: Created widget for model {item.name} with display_name: {item.display_name}, description: {item.description}")
                else:  # lora
                    widget = self._create_installed_lora_widget(item)
                self.installed_models_layout.addWidget(widget)

        # Dynamically adjust scroll area height based on number of items
        self._adjust_installed_models_height(total_items)

        # Force UI update
        self.update()
        self.repaint()

        print("DEBUG: Finished _refresh_installed_models")

    def _create_installed_model_widget(self, model) -> QWidget:
        """Create compact table row widget for installed model - matching available model layout style."""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(10)

        # Column 1: Icon + Name (flexible width)
        name_layout = QHBoxLayout()
        name_layout.setSpacing(6)

        icon_label = QLabel("📦")
        icon_label.setStyleSheet("font-size: 14px;")
        icon_label.setMinimumWidth(20)
        name_layout.addWidget(icon_label)

        display_name = model.display_name if model.display_name else model.name
        name_text = display_name
        if model.is_default:
            name_text += " ⭐"
        name_label = QLabel(name_text)
        name_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #1565C0;")
        name_label.setToolTip(f"Name: {model.name}\nDisplay: {display_name}")
        name_label.setMinimumWidth(150)  # Ensure minimum width for text
        name_layout.addWidget(name_label)

        name_layout.addStretch()
        layout.addLayout(name_layout, stretch=4)  # Give name column more space

        # Column 2: Type (fixed width)
        type_text = model.model_type.value if model.model_type else 'Unknown'
        type_label = QLabel(type_text)
        type_label.setStyleSheet("""
            font-size: 11px;
            color: #666;
            background-color: #f0f0f0;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: 500;
        """)
        type_label.setFixedWidth(120)
        type_label.setMinimumWidth(120)
        layout.addWidget(type_label)

        # Column 3: Size (fixed width)
        try:
            size_mb = float(model.size_mb) if model.size_mb and str(model.size_mb).replace('.', '').isdigit() else 0.0
        except (ValueError, TypeError):
            size_mb = 0.0
        size_label = QLabel(f"{size_mb:.1f} MB")
        size_label.setStyleSheet("color: #555; font-size: 11px; font-weight: 500;")
        size_label.setFixedWidth(70)
        size_label.setMinimumWidth(70)
        size_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(size_label)

        # Column 4: Description (flexible width, truncated)
        desc_text = model.description or "No description"
        if len(desc_text) > 50:
            desc_text = "..." + desc_text[-47:]
        desc_label = QLabel(desc_text)
        desc_label.setStyleSheet("""
            color: #777;
            font-size: 10px;
            font-family: 'Segoe UI', monospace;
        """)
        desc_label.setToolTip(model.description or "No description")
        desc_label.setMinimumWidth(200)  # Ensure minimum width for description text
        desc_label.setWordWrap(False)  # Prevent wrapping
        layout.addWidget(desc_label, stretch=3)

        # Column 5: Action buttons (fixed width)
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(2)

        # Edit button
        edit_btn = QPushButton('Edit')
        edit_btn.setProperty("model_unique_id", model.unique_id)
        edit_btn.clicked.connect(self._on_edit_installed_clicked)
        edit_btn.setFixedSize(45, 24)
        edit_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 10px;
                font-weight: bold;
                padding: 2px 4px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:pressed {
                background-color: #EF6C00;
            }
        """)
        buttons_layout.addWidget(edit_btn)

        # Delete button
        delete_btn = QPushButton('Delete')
        delete_btn.setProperty("model_unique_id", model.unique_id)
        delete_btn.clicked.connect(self._on_delete_installed_clicked)
        delete_btn.setFixedSize(55, 24)
        delete_btn.setEnabled(not model.is_default)  # Can't delete default
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 10px;
                font-weight: bold;
                padding: 2px 4px;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
            QPushButton:pressed {
                background-color: #B71C1C;
            }
            QPushButton:disabled {
                color: #ccc;
                background-color: #f5f5f5;
                border: 1px solid #cccccc;
            }
        """)
        buttons_layout.addWidget(delete_btn)



        buttons_widget.setLayout(buttons_layout)
        layout.addWidget(buttons_widget)

        # Set layout and compact styling
        widget.setLayout(layout)
        widget.setStyleSheet("""
            QWidget {
                border: 1px solid #4CAF50;
                border-radius: 4px;
                background-color: #f8fff8;
                margin: 1px;
            }
            QWidget:hover {
                background-color: #f0fff0;
                border-color: #388E3C;
            }
        """)
        widget.setMinimumHeight(32)  # Slightly taller to accommodate text
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        return widget

    def _create_installed_lora_widget(self, lora) -> QWidget:
        """Create compact table row widget for installed LoRA - matching available model layout style."""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(10)

        # Column 1: Icon + Name (flexible width)
        name_layout = QHBoxLayout()
        name_layout.setSpacing(6)

        icon_label = QLabel("🎭")
        icon_label.setStyleSheet("font-size: 14px;")
        icon_label.setMinimumWidth(20)
        name_layout.addWidget(icon_label)

        display_name = lora.display_name if lora.display_name else lora.name
        name_label = QLabel(display_name)
        name_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #E65100;")
        name_label.setToolTip(f"Name: {lora.name}\nDisplay: {display_name}")
        name_label.setMinimumWidth(150)  # Ensure minimum width for text
        name_layout.addWidget(name_label)

        name_layout.addStretch()
        layout.addLayout(name_layout, stretch=4)  # Give name column more space

        # Column 2: Base Model Type (fixed width)
        base_model_text = lora.base_model_type.value if lora.base_model_type else 'Any'
        type_label = QLabel(base_model_text)
        type_label.setStyleSheet("""
            font-size: 11px;
            color: #666;
            background-color: #fff3e0;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: 500;
        """)
        type_label.setFixedWidth(120)
        type_label.setMinimumWidth(120)
        layout.addWidget(type_label)

        # Column 3: Size (fixed width)
        try:
            size_mb = float(lora.size_mb) if lora.size_mb and str(lora.size_mb).replace('.', '').isdigit() else 0.0
        except (ValueError, TypeError):
            size_mb = 0.0
        size_label = QLabel(f"{size_mb:.1f} MB")
        size_label.setStyleSheet("color: #555; font-size: 11px; font-weight: 500;")
        size_label.setFixedWidth(70)
        size_label.setMinimumWidth(70)
        size_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(size_label)

        # Column 4: Trigger Words (flexible width, truncated)
        trigger_text = ", ".join(lora.trigger_words[:5]) if lora.trigger_words else "No triggers"
        if len(lora.trigger_words or []) > 5:
            trigger_text = trigger_text[:40] + "..."
        trigger_label = QLabel(trigger_text)
        trigger_label.setStyleSheet("""
            color: #777;
            font-size: 10px;
            font-family: 'Segoe UI', monospace;
        """)
        trigger_label.setToolTip(", ".join(lora.trigger_words) if lora.trigger_words else "No trigger words")
        trigger_label.setMinimumWidth(200)  # Ensure minimum width for trigger words text
        trigger_label.setWordWrap(False)  # Prevent wrapping
        layout.addWidget(trigger_label, stretch=3)

        # Column 5: Action buttons (fixed width)
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(2)

        # Edit button
        edit_btn = QPushButton('Edit')
        edit_btn.setProperty("lora_name", lora.name)
        edit_btn.clicked.connect(self._on_edit_lora_clicked)
        edit_btn.setFixedSize(45, 24)
        edit_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 10px;
                font-weight: bold;
                padding: 2px 4px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:pressed {
                background-color: #EF6C00;
            }
        """)
        buttons_layout.addWidget(edit_btn)

        # Delete button
        delete_btn = QPushButton('Delete')
        delete_btn.setProperty("lora_name", lora.name)
        delete_btn.clicked.connect(self._on_delete_lora_clicked)
        delete_btn.setFixedSize(55, 24)
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 10px;
                font-weight: bold;
                padding: 2px 4px;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
            QPushButton:pressed {
                background-color: #B71C1C;
            }
        """)
        buttons_layout.addWidget(delete_btn)

        buttons_widget.setLayout(buttons_layout)
        layout.addWidget(buttons_widget)

        # Set layout and compact styling
        widget.setLayout(layout)
        widget.setStyleSheet("""
            QWidget {
                border: 1px solid #FF7043;
                border-radius: 4px;
                background-color: #fff8f5;
                margin: 1px;
            }
            QWidget:hover {
                background-color: #fff0eb;
                border-color: #E64A19;
            }
        """)
        widget.setMinimumHeight(32)  # Slightly taller to accommodate text
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        return widget

    def _edit_lora_params(self, lora_name: str):
        """Edit LoRA parameters."""
        try:
            # Refresh the installed items list first to get latest data
            self._refresh_installed_models()

            # Get the current LoRA data
            installed_loras = self.model_manager.get_installed_loras()
            current_lora = None
            for lora in installed_loras:
                if lora.name == lora_name:
                    current_lora = lora
                    break

            if not current_lora:
                QMessageBox.critical(self.sidebar, "Error", f"LoRA '{lora_name}' not found!")
                return

            # Create and show edit dialog
            dialog = LoRAEditDialog(current_lora, self)
            result = dialog.exec_()

            # Only proceed with save if Save button was clicked (AcceptRole)
            if result == QMessageBox.AcceptRole:
                # Get updated metadata
                updated_metadata = dialog.get_metadata()

                # Validate name uniqueness if changed
                if updated_metadata['name'] != current_lora.name:
                    # Check if new name already exists
                    existing_names = [l.name for l in installed_loras if l.name != current_lora.name]
                    if updated_metadata['name'] in existing_names:
                        QMessageBox.critical(self.sidebar, "Error",
                                           f"LoRA name '{updated_metadata['name']}' already exists!")
                        return

                # Create updated LoRA info
                updated_lora = LoRAInfo(
                    name=updated_metadata['name'],
                    path=current_lora.path,  # Path stays the same
                    display_name=updated_metadata['display_name'],
                    base_model_type=updated_metadata['base_model_type'],
                    description=updated_metadata['description'],
                    trigger_words=updated_metadata['trigger_words'],
                    categories=updated_metadata['categories'],
                    usage_notes=updated_metadata['usage_notes'],
                    source_url=updated_metadata['source_url'],
                    license_info=updated_metadata['license_info'],
                    size_mb=current_lora.size_mb,  # Preserve size
                    installed_date=current_lora.installed_date,  # Preserve install date
                    last_used=current_lora.last_used,  # Preserve usage data
                    usage_count=current_lora.usage_count,  # Preserve usage count
                    default_scaling=updated_metadata['default_scaling']
                )

                # Save to database
                if self.model_manager.db.update_lora(current_lora.name, updated_lora):
                    QMessageBox.information(self.sidebar, "Success",
                                          f"LoRA '{lora_name}' parameters updated successfully!")

                    # Refresh the UI
                    self._refresh_installed_models()
                else:
                    QMessageBox.critical(self.sidebar, "Error", "Failed to update LoRA parameters!")
            # If result is RejectRole (Cancel) or any other value, do nothing

        except Exception as e:
            QMessageBox.critical(self.sidebar, "Error", f"Failed to edit LoRA parameters: {str(e)}")

    def _delete_lora(self, lora_name: str):
        """Delete a LoRA."""
        reply = QMessageBox.question(self.sidebar, "Confirm Delete",
                                   f"Are you sure you want to delete LoRA '{lora_name}'?",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            success = self.model_manager.delete_lora(lora_name)
            if success:
                QMessageBox.information(self.sidebar, "Deleted", f"LoRA '{lora_name}' deleted successfully!")
                self._refresh_installed_models()
            else:
                QMessageBox.critical(self.sidebar, "Error", f"Failed to delete LoRA '{lora_name}'!")



    def _validate_model_data(self, metadata: dict) -> List[str]:
        """Validate model metadata and return list of errors."""
        errors = []

        # Name validation
        if not metadata.get('name', '').strip():
            errors.append("Model name cannot be empty")

        # URL validation
        source_url = metadata.get('source_url', '').strip()
        if source_url and not (source_url.startswith('http://') or source_url.startswith('https://')):
            errors.append("Source URL must start with http:// or https://")

        # Numeric validation
        if not (1 <= metadata.get('default_steps', 20) <= 100):
            errors.append("Steps must be between 1 and 100")

        if not (1.0 <= metadata.get('default_cfg', 7.5) <= 20.0):
            errors.append("CFG scale must be between 1.0 and 20.0")

        # Aspect ratio validation
        import re
        aspect_pattern = re.compile(r'^\d+x\d+$')
        for ratio_field in ['aspect_ratio_1_1', 'aspect_ratio_9_16', 'aspect_ratio_16_9']:
            ratio = metadata.get(ratio_field, '').strip()
            if ratio and not aspect_pattern.match(ratio):
                errors.append(f"{ratio_field.replace('_', ' ').title()} must be in format 'WIDTHxHEIGHT' (e.g., '512x512')")

        return errors

    def _edit_model_params(self, unique_id: str):
        """Edit model parameters with comprehensive validation and error handling."""
        try:
            # Refresh the installed models list first to get latest data
            self._refresh_installed_models()

            # Get the current model data
            installed_models = self.model_manager.get_installed_models()
            current_model = None
            for model in installed_models:
                if model.unique_id == unique_id:
                    current_model = model
                    break

            if not current_model:
                QMessageBox.critical(self.sidebar, "Error", f"Model '{unique_id}' not found!")
                return

            # Create and show edit dialog
            dialog = ModelEditDialog(current_model, self)
            result = dialog.exec_()

            # Only proceed with save if Save button was clicked (AcceptRole)
            if result == QMessageBox.AcceptRole:
                # Get updated metadata
                updated_metadata = dialog.get_metadata()

                # Validate the data
                validation_errors = self._validate_model_data(updated_metadata)
                if validation_errors:
                    error_msg = "Validation errors:\n" + "\n".join(f"• {err}" for err in validation_errors)
                    QMessageBox.warning(self.sidebar, "Validation Error", error_msg)
                    return

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
                    unique_id=current_model.unique_id,  # Preserve unique_id
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
                    usage_count=current_model.usage_count,  # Preserve usage count
                    aspect_ratio_1_1=updated_metadata['aspect_ratio_1_1'],
                    aspect_ratio_9_16=updated_metadata['aspect_ratio_9_16'],
                    aspect_ratio_16_9=updated_metadata['aspect_ratio_16_9'],
                    default_steps=updated_metadata['default_steps'],
                    default_cfg=updated_metadata['default_cfg']
                )

                # Save to database using the safe update method
                success = self.model_manager.update_model_by_unique_id(current_model.unique_id, updated_model)
                if success:
                    QMessageBox.information(self.sidebar, "Success",
                                          f"Model '{updated_metadata['name']}' parameters updated successfully!")

                    # Refresh the UI - get fresh data from database
                    print(f"DEBUG: Refreshing UI after model update for {unique_id}")
                    print(f"DEBUG: Updated model data - name: {updated_model.name}, display_name: {updated_model.display_name}, description: {updated_model.description}")
                    self._refresh_installed_models()
                    print("DEBUG: UI refresh completed")
                else:
                    QMessageBox.critical(self.sidebar, "Save Failed",
                                       "Failed to save model parameters. Please check the application logs for details.")
            # If result is RejectRole (Cancel) or any other value, do nothing - just close the dialog

        except Exception as e:
            QMessageBox.critical(self.sidebar, "Error", f"Unexpected error during model editing: {str(e)}")
            import traceback
            traceback.print_exc()

    def _delete_model(self, unique_id: str):
        """Delete a model."""
        reply = QMessageBox.question(self.sidebar, "Confirm Delete",
                                   f"Are you sure you want to delete model '{unique_id}'?",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            # Save state for undo before deletion
            self.model_manager._save_operation_for_undo("delete_model", {"unique_id": unique_id})

            success = self.model_manager.delete_model(unique_id)
            if success:
                QMessageBox.information(self.sidebar, "Deleted", f"Model '{unique_id}' deleted successfully!")
                self._refresh_installed_models()
                self._update_undo_redo_buttons()
            else:
                QMessageBox.critical(self.sidebar, "Error", f"Failed to delete model '{unique_id}'!")

    def _set_default_model(self, model_name: str):
        """Set model as default."""
        # Save state for undo before changing default
        self.model_manager._save_operation_for_undo("set_default_model", {"model_name": model_name})

        success = self.model_manager.set_default_model(model_name)
        if success:
            QMessageBox.information(self.sidebar, "Set Default", f"Model '{model_name}' set as default!")
            self._refresh_installed_models()
            self._update_undo_redo_buttons()
        else:
            QMessageBox.critical(self.sidebar, "Error", f"Failed to set model '{model_name}' as default!")

    def _on_edit_installed_clicked(self):
        """Handle edit button click for installed models."""
        button = self.sender()
        if button:
            model_unique_id = button.property("model_unique_id")
            if model_unique_id:
                self._edit_model_params(model_unique_id)

    def _on_delete_installed_clicked(self):
        """Handle delete button click for installed models."""
        button = self.sender()
        if button:
            model_unique_id = button.property("model_unique_id")
            if model_unique_id:
                self._delete_model(model_unique_id)



    def _on_edit_lora_clicked(self):
        """Handle edit button click for installed LoRAs."""
        button = self.sender()
        if button:
            lora_name = button.property("lora_name")
            if lora_name:
                self._edit_lora_params(lora_name)

    def _on_delete_lora_clicked(self):
        """Handle delete button click for installed LoRAs."""
        button = self.sender()
        if button:
            lora_name = button.property("lora_name")
            if lora_name:
                self._delete_lora(lora_name)



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


class ModelInstallDialog(QDialog):
    """Dialog for collecting model metadata during installation."""

    def __init__(self, model_name: str, model_path: str, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.model_path = model_path

        self.setWindowTitle("Install AI Model")
        self.setModal(True)
        self.setFixedSize(700, 600)

        # Center the dialog
        if parent:
            parent_rect = parent.geometry()
            self.move(
                parent_rect.x() + (parent_rect.width() - self.width()) // 2,
                parent_rect.y() + (parent_rect.height() - self.height()) // 2
            )

        self.accepted = False  # Track if user accepted
        self._init_ui()

    def _init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header section with icon and title
        header_widget = QWidget()
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(15)

        # Icon
        icon_label = QLabel("📦")
        icon_font = QFont()
        icon_font.setPointSize(32)
        icon_label.setFont(icon_font)
        header_layout.addWidget(icon_label)

        # Title and subtitle
        title_widget = QWidget()
        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(5)

        title_label = QLabel("Install AI Model")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #1976D2;")
        title_layout.addWidget(title_label)

        model_label = QLabel(f"📁 {self.model_name}")
        model_font = QFont()
        model_font.setPointSize(12)
        model_label.setFont(model_font)
        model_label.setStyleSheet("color: #666;")
        title_layout.addWidget(model_label)

        title_widget.setLayout(title_layout)
        header_layout.addWidget(title_widget, stretch=1)

        header_widget.setLayout(header_layout)
        layout.addWidget(header_widget)

        # Add separator
        separator = QWidget()
        separator.setFixedHeight(2)
        separator.setStyleSheet("background-color: #e0e0e0; border-radius: 1px;")
        layout.addWidget(separator)

        # Scrollable content area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setMinimumHeight(350)
        scroll_area.setMaximumHeight(350)

        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(15)

        # File path info
        path_group = QWidget()
        path_layout = QVBoxLayout()
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.setSpacing(5)

        path_label = QLabel("📂 Source Path:")
        path_label.setStyleSheet("font-weight: bold; color: #333;")
        path_layout.addWidget(path_label)

        path_display = QLabel(self.model_path)
        path_display.setWordWrap(True)
        path_display.setStyleSheet("""
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 8px;
            color: #666;
            font-family: 'Segoe UI', monospace;
            font-size: 11px;
        """)
        path_layout.addWidget(path_display)

        path_group.setLayout(path_layout)
        content_layout.addWidget(path_group)

        # Metadata form
        form_widget = QWidget()
        form_layout = QVBoxLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(12)

        # Basic Information Section
        basic_group = QGroupBox("Basic Information")
        basic_layout = QVBoxLayout()
        basic_layout.setSpacing(8)

        # Display Name
        display_layout = QVBoxLayout()
        display_layout.setSpacing(3)
        display_label = QLabel("Display Name:")
        display_label.setStyleSheet("font-weight: bold;")
        display_layout.addWidget(display_label)
        self.display_name_edit = QLineEdit()
        self.display_name_edit.setText(self.model_name)
        self.display_name_edit.setPlaceholderText("User-friendly name for this model")
        self.display_name_edit.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background-color: #ffffff;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #1976D2;
            }
        """)
        display_layout.addWidget(self.display_name_edit)
        basic_layout.addLayout(display_layout)

        # Description
        desc_layout = QVBoxLayout()
        desc_layout.setSpacing(3)
        desc_label = QLabel("Description:")
        desc_label.setStyleSheet("font-weight: bold;")
        desc_layout.addWidget(desc_label)
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        self.description_edit.setPlaceholderText("Enter a description for this model...")
        self.description_edit.setStyleSheet("""
            QTextEdit {
                padding: 8px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background-color: #ffffff;
                font-size: 12px;
            }
            QTextEdit:focus {
                border-color: #1976D2;
            }
        """)
        desc_layout.addWidget(self.description_edit)
        basic_layout.addLayout(desc_layout)

        basic_group.setLayout(basic_layout)
        form_layout.addWidget(basic_group)

        # Categories Section
        categories_group = QGroupBox("Categories")
        categories_layout = QVBoxLayout()
        categories_layout.setSpacing(8)

        categories_desc = QLabel("Select categories that best describe this model:")
        categories_desc.setStyleSheet("color: #666; font-size: 11px;")
        categories_layout.addWidget(categories_desc)

        self.category_list = QListWidget()
        self.category_list.setMaximumHeight(120)
        self.category_list.setSelectionMode(QListWidget.MultiSelection)
        self.category_list.setStyleSheet("""
            QListWidget {
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background-color: #ffffff;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
                border-radius: 4px;
                margin: 2px;
            }
            QListWidget::item:selected {
                background-color: #1976D2;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #f0f8ff;
            }
        """)

        # Add category options
        for category in ModelCategory:
            item = QListWidgetItem(category.value.title())
            item.setData(1, category.value)  # Store the enum value
            self.category_list.addItem(item)

        categories_layout.addWidget(self.category_list)
        categories_group.setLayout(categories_layout)
        form_layout.addWidget(categories_group)

        # Additional Information Section
        additional_group = QGroupBox("Additional Information")
        additional_layout = QVBoxLayout()
        additional_layout.setSpacing(8)

        # Usage Notes
        usage_layout = QVBoxLayout()
        usage_layout.setSpacing(3)
        usage_label = QLabel("Usage Notes:")
        usage_label.setStyleSheet("font-weight: bold;")
        usage_layout.addWidget(usage_label)
        self.usage_edit = QTextEdit()
        self.usage_edit.setMaximumHeight(60)
        self.usage_edit.setPlaceholderText("Any special usage notes or tips...")
        self.usage_edit.setStyleSheet("""
            QTextEdit {
                padding: 8px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background-color: #ffffff;
                font-size: 12px;
            }
            QTextEdit:focus {
                border-color: #1976D2;
            }
        """)
        usage_layout.addWidget(self.usage_edit)
        additional_layout.addLayout(usage_layout)

        # Source URL and License in horizontal layout
        urls_layout = QHBoxLayout()
        urls_layout.setSpacing(15)

        # Source URL
        url_layout = QVBoxLayout()
        url_layout.setSpacing(3)
        source_label = QLabel("Source URL:")
        source_label.setStyleSheet("font-weight: bold;")
        url_layout.addWidget(source_label)
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("https://...")
        self.source_edit.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background-color: #ffffff;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #1976D2;
            }
        """)
        url_layout.addWidget(self.source_edit)
        urls_layout.addLayout(url_layout)

        # License Info
        license_layout = QVBoxLayout()
        license_layout.setSpacing(3)
        license_label = QLabel("License:")
        license_label.setStyleSheet("font-weight: bold;")
        license_layout.addWidget(license_label)
        self.license_edit = QLineEdit()
        self.license_edit.setPlaceholderText("License type or attribution...")
        self.license_edit.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background-color: #ffffff;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #1976D2;
            }
        """)
        license_layout.addWidget(self.license_edit)
        urls_layout.addLayout(license_layout)

        additional_layout.addLayout(urls_layout)
        additional_group.setLayout(additional_layout)
        form_layout.addWidget(additional_group)

        form_widget.setLayout(form_layout)
        content_layout.addWidget(form_widget)

        content_widget.setLayout(content_layout)
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)

        # Buttons section
        button_widget = QWidget()
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(10)

        button_layout.addStretch()

        # Cancel button
        self.cancel_btn = QPushButton("❌ Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setMinimumWidth(100)
        button_layout.addWidget(self.cancel_btn)

        # Install button
        self.install_btn = QPushButton("✅ Install Model")
        self.install_btn.clicked.connect(self.accept)
        self.install_btn.setMinimumHeight(40)
        self.install_btn.setMinimumWidth(140)
        self.install_btn.setDefault(True)
        button_layout.addWidget(self.install_btn)

        button_widget.setLayout(button_layout)
        layout.addWidget(button_widget)

        # Set layout
        self.setLayout(layout)

        # Modern stylesheet
        self.setStyleSheet("""
            QDialog {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 12px;
            }

            QGroupBox {
                font-weight: bold;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #fafafa;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #1976D2;
                font-size: 13px;
            }

            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 12px;
            }

            QPushButton:hover {
                background-color: #d32f2f;
                transform: translateY(-1px);
            }

            QPushButton:pressed {
                background-color: #b71c1c;
                transform: translateY(0px);
            }

            QPushButton#install_btn {
                background-color: #4CAF50;
            }

            QPushButton#install_btn:hover {
                background-color: #45a049;
            }

            QPushButton#install_btn:pressed {
                background-color: #388E3C;
            }
        """)

        # Set object name for install button styling
        self.install_btn.setObjectName("install_btn")

    def accept(self):
        """Handle accept button click."""
        self.accepted = True
        super().accept()

    def reject(self):
        """Handle reject button click."""
        self.accepted = False
        super().reject()

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
        # Remove the default text to prevent overlapping - we'll use a custom layout
        self.setText("")
        self.setInformativeText("")

        # Set minimum width for better layout
        self.setMinimumWidth(800)

        # Add custom widgets for metadata editing
        self._setup_edit_widgets()

        # Add standard buttons (Save first, then Cancel)
        save_button = self.addButton("Save", QMessageBox.AcceptRole)
        cancel_button = self.addButton("Cancel", QMessageBox.RejectRole)

        # Set default button to Save
        save_button.setDefault(True)
        save_button.setFocus()

        # Ensure proper button behavior
        cancel_button.setAutoDefault(False)

    def _setup_edit_widgets(self):
        """Set up widgets for editing model metadata with a more compact, wider layout."""
        # Create a widget to hold our custom controls
        widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Title section
        title_label = QLabel(f"Edit parameters for '{self.model.name}'")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #1976D2; margin-bottom: 5px;")
        main_layout.addWidget(title_label)

        # Main content area - split into two columns
        content_widget = QWidget()
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # Left column - Basic model information
        left_column = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)

        # Model Name (unique identifier)
        name_label = QLabel("Model Name:")
        name_label.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(name_label)
        self.name_edit = QLineEdit()
        self.name_edit.setText(str(self.model.name))
        self.name_edit.setPlaceholderText("Unique name for this model")
        left_layout.addWidget(self.name_edit)

        # Display Name
        display_name_label = QLabel("Display Name:")
        left_layout.addWidget(display_name_label)
        self.display_name_edit = QLineEdit()
        self.display_name_edit.setText(str(self.model.display_name or ""))
        self.display_name_edit.setPlaceholderText("User-friendly name")
        left_layout.addWidget(self.display_name_edit)

        # Model Type
        model_type_label = QLabel("Model Type:")
        left_layout.addWidget(model_type_label)
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

        left_layout.addWidget(self.model_type_combo)

        # Description
        desc_label = QLabel("Description:")
        left_layout.addWidget(desc_label)
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        self.description_edit.setPlainText(str(self.model.description or ""))
        self.description_edit.setPlaceholderText("Model description...")
        left_layout.addWidget(self.description_edit)

        left_column.setLayout(left_layout)
        content_layout.addWidget(left_column)

        # Right column - Categories and URLs
        right_column = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)

        # Categories
        cat_label = QLabel("Categories:")
        right_layout.addWidget(cat_label)
        self.category_list = QListWidget()
        self.category_list.setMaximumHeight(120)
        self.category_list.setSelectionMode(QListWidget.MultiSelection)

        # Add category options and pre-select current categories
        current_categories = set(self.model.categories) if self.model.categories else set()
        for category in ModelCategory:
            item = QListWidgetItem(category.value.title())
            item.setData(1, category.value)  # Store the enum value
            if category in current_categories:
                item.setSelected(True)
            self.category_list.addItem(item)

        right_layout.addWidget(self.category_list)

        # Usage Notes
        usage_label = QLabel("Usage Notes:")
        right_layout.addWidget(usage_label)
        self.usage_edit = QTextEdit()
        self.usage_edit.setMaximumHeight(60)
        self.usage_edit.setPlainText(str(self.model.usage_notes or ""))
        self.usage_edit.setPlaceholderText("Usage tips...")
        right_layout.addWidget(self.usage_edit)

        # Source URL and License in horizontal layout
        urls_layout = QHBoxLayout()
        urls_layout.setSpacing(10)

        # Source URL
        url_widget = QWidget()
        url_layout = QVBoxLayout()
        url_layout.setSpacing(2)
        source_label = QLabel("Source URL:")
        url_layout.addWidget(source_label)
        self.source_edit = QLineEdit()
        self.source_edit.setText(str(self.model.source_url or ""))
        self.source_edit.setPlaceholderText("https://...")
        url_layout.addWidget(self.source_edit)
        url_widget.setLayout(url_layout)
        urls_layout.addWidget(url_widget)

        # License Info
        license_widget = QWidget()
        license_layout = QVBoxLayout()
        license_layout.setSpacing(2)
        license_label = QLabel("License:")
        license_layout.addWidget(license_label)
        self.license_edit = QLineEdit()
        self.license_edit.setText(str(self.model.license_info or ""))
        self.license_edit.setPlaceholderText("License info...")
        license_layout.addWidget(self.license_edit)
        license_widget.setLayout(license_layout)
        urls_layout.addWidget(license_widget)

        right_layout.addLayout(urls_layout)

        right_column.setLayout(right_layout)
        content_layout.addWidget(right_column)

        content_widget.setLayout(content_layout)
        main_layout.addWidget(content_widget)

        # Bottom section - Parameters (side by side)
        params_widget = QWidget()
        params_layout = QHBoxLayout()
        params_layout.setSpacing(15)

        # Generation Parameters Section
        gen_params_group = QGroupBox("Generation Defaults")
        gen_params_layout = QVBoxLayout()
        gen_params_layout.setSpacing(8)

        # Default Steps and CFG in horizontal layout
        gen_inputs_layout = QHBoxLayout()
        gen_inputs_layout.setSpacing(15)

        # Steps
        steps_widget = QWidget()
        steps_layout = QVBoxLayout()
        steps_layout.setSpacing(2)
        steps_label = QLabel("Steps:")
        steps_layout.addWidget(steps_label)
        self.default_steps_spin = QSpinBox()
        self.default_steps_spin.setRange(1, 100)
        self.default_steps_spin.setValue(int(self.model.default_steps))
        self.default_steps_spin.setToolTip("Default inference steps")
        steps_layout.addWidget(self.default_steps_spin)
        steps_widget.setLayout(steps_layout)
        gen_inputs_layout.addWidget(steps_widget)

        # CFG Scale
        cfg_widget = QWidget()
        cfg_layout = QVBoxLayout()
        cfg_layout.setSpacing(2)
        cfg_label = QLabel("CFG Scale:")
        cfg_layout.addWidget(cfg_label)
        self.default_cfg_spin = QDoubleSpinBox()
        self.default_cfg_spin.setRange(1.0, 20.0)
        self.default_cfg_spin.setValue(float(self.model.default_cfg))
        self.default_cfg_spin.setSingleStep(0.1)
        self.default_cfg_spin.setToolTip("Default guidance scale")
        cfg_layout.addWidget(self.default_cfg_spin)
        cfg_widget.setLayout(cfg_layout)
        gen_inputs_layout.addWidget(cfg_widget)

        gen_params_layout.addLayout(gen_inputs_layout)

        # Auto-set button
        auto_set_gen_btn = QPushButton("Set Default")
        auto_set_gen_btn.setMaximumWidth(100)
        auto_set_gen_btn.clicked.connect(self._auto_set_generation_params)
        auto_set_gen_btn.setToolTip("Set defaults based on model type")
        gen_params_layout.addWidget(auto_set_gen_btn)

        gen_params_group.setLayout(gen_params_layout)
        params_layout.addWidget(gen_params_group)

        # Aspect Ratios Section
        aspect_group = QGroupBox("Aspect Ratios")
        aspect_layout = QVBoxLayout()
        aspect_layout.setSpacing(6)

        # Aspect ratios in separate rows
        ratios_layout = QVBoxLayout()
        ratios_layout.setSpacing(8)

        # 1:1 in first row
        ratio_1_1_widget = QWidget()
        ratio_1_1_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ratio_1_1_layout = QVBoxLayout()
        ratio_1_1_layout.setSpacing(2)
        ratio_1_1_label = QLabel("1:1:")
        ratio_1_1_layout.addWidget(ratio_1_1_label)
        self.aspect_ratio_1_1_edit = QLineEdit()
        self.aspect_ratio_1_1_edit.setText(str(self.model.aspect_ratio_1_1 or ""))
        self.aspect_ratio_1_1_edit.setPlaceholderText("512x512")
        self.aspect_ratio_1_1_edit.setMaximumWidth(200)
        ratio_1_1_layout.addWidget(self.aspect_ratio_1_1_edit)
        ratio_1_1_widget.setLayout(ratio_1_1_layout)
        ratios_layout.addWidget(ratio_1_1_widget)

        # 9:16 in second row
        ratio_9_16_widget = QWidget()
        ratio_9_16_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ratio_9_16_layout = QVBoxLayout()
        ratio_9_16_layout.setSpacing(2)
        ratio_9_16_label = QLabel("9:16:")
        ratio_9_16_layout.addWidget(ratio_9_16_label)
        self.aspect_ratio_9_16_edit = QLineEdit()
        self.aspect_ratio_9_16_edit.setText(str(self.model.aspect_ratio_9_16 or ""))
        self.aspect_ratio_9_16_edit.setPlaceholderText("384x672")
        self.aspect_ratio_9_16_edit.setMaximumWidth(200)
        ratio_9_16_layout.addWidget(self.aspect_ratio_9_16_edit)
        ratio_9_16_widget.setLayout(ratio_9_16_layout)
        ratios_layout.addWidget(ratio_9_16_widget)

        # 16:9 in third row
        ratio_16_9_widget = QWidget()
        ratio_16_9_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ratio_16_9_layout = QVBoxLayout()
        ratio_16_9_layout.setSpacing(2)
        ratio_16_9_label = QLabel("16:9:")
        ratio_16_9_layout.addWidget(ratio_16_9_label)
        self.aspect_ratio_16_9_edit = QLineEdit()
        self.aspect_ratio_16_9_edit.setText(str(self.model.aspect_ratio_16_9 or ""))
        self.aspect_ratio_16_9_edit.setPlaceholderText("672x384")
        self.aspect_ratio_16_9_edit.setMaximumWidth(200)
        ratio_16_9_layout.addWidget(self.aspect_ratio_16_9_edit)
        ratio_16_9_widget.setLayout(ratio_16_9_layout)
        ratios_layout.addWidget(ratio_16_9_widget)

        # Auto-set button
        auto_set_btn = QPushButton("Set Default")
        auto_set_btn.setMaximumWidth(100)
        auto_set_btn.clicked.connect(self._auto_set_aspect_ratios)
        auto_set_btn.setToolTip("Set defaults based on model type")
        ratios_layout.addWidget(auto_set_btn)

        aspect_layout.addLayout(ratios_layout)
        aspect_group.setLayout(aspect_layout)
        params_layout.addWidget(aspect_group)

        params_widget.setLayout(params_layout)
        main_layout.addWidget(params_widget)

        widget.setLayout(main_layout)
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
            'license_info': self.license_edit.text().strip(),
            'default_steps': self.default_steps_spin.value(),
            'default_cfg': self.default_cfg_spin.value(),
            'aspect_ratio_1_1': self.aspect_ratio_1_1_edit.text().strip(),
            'aspect_ratio_9_16': self.aspect_ratio_9_16_edit.text().strip(),
            'aspect_ratio_16_9': self.aspect_ratio_16_9_edit.text().strip()
        }

    def _auto_set_generation_params(self):
        """Auto-set generation parameters based on the selected model type."""
        # Get selected model type
        current_index = self.model_type_combo.currentIndex()
        model_type = self.model_type_combo.itemData(current_index)

        # Set default generation parameters based on model type
        if model_type == ModelType.STABLE_DIFFUSION_XL:
            # SDXL models often work better with different defaults
            self.default_steps_spin.setValue(25)  # SDXL typically needs more steps
            self.default_cfg_spin.setValue(7.0)   # Slightly lower CFG for SDXL
        else:
            # SD 1.4/1.5 defaults
            self.default_steps_spin.setValue(20)
            self.default_cfg_spin.setValue(7.5)

    def _auto_set_aspect_ratios(self):
        """Auto-set aspect ratios based on the selected model type."""
        # Get selected model type
        current_index = self.model_type_combo.currentIndex()
        model_type = self.model_type_combo.itemData(current_index)

        # Set default aspect ratios based on model type
        if model_type == ModelType.STABLE_DIFFUSION_XL:
            # SDXL base resolution is 1024x1024
            self.aspect_ratio_1_1_edit.setText("1024x1024")
            self.aspect_ratio_9_16_edit.setText("768x1344")  # portrait
            self.aspect_ratio_16_9_edit.setText("1344x768")  # landscape
        else:
            # SD 1.4/1.5 base resolution is 512x512
            self.aspect_ratio_1_1_edit.setText("512x512")
            self.aspect_ratio_9_16_edit.setText("384x672")  # portrait
            self.aspect_ratio_16_9_edit.setText("672x384")  # landscape


class LoRAInstallDialog(QDialog):
    """Dialog for collecting LoRA adapter metadata during installation."""

    def __init__(self, lora_name: str, lora_path: str, parent=None):
        super().__init__(parent)
        self.lora_name = lora_name
        self.lora_path = lora_path

        self.setWindowTitle("Install LoRA Adapter")
        self.setModal(True)
        self.setFixedSize(700, 600)

        # Center the dialog
        if parent:
            parent_rect = parent.geometry()
            self.move(
                parent_rect.x() + (parent_rect.width() - self.width()) // 2,
                parent_rect.y() + (parent_rect.height() - self.height()) // 2
            )

        self.accepted = False  # Track if user accepted
        self._init_ui()

    def _init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header section with icon and title
        header_widget = QWidget()
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(15)

        # Icon
        icon_label = QLabel("🎭")
        icon_font = QFont()
        icon_font.setPointSize(32)
        icon_label.setFont(icon_font)
        header_layout.addWidget(icon_label)

        # Title and subtitle
        title_widget = QWidget()
        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(5)

        title_label = QLabel("Install LoRA Adapter")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #FF6B35;")
        title_layout.addWidget(title_label)

        lora_label = QLabel(f"🎭 {self.lora_name}")
        lora_font = QFont()
        lora_font.setPointSize(12)
        lora_label.setFont(lora_font)
        lora_label.setStyleSheet("color: #666;")
        title_layout.addWidget(lora_label)

        title_widget.setLayout(title_layout)
        header_layout.addWidget(title_widget, stretch=1)

        header_widget.setLayout(header_layout)
        layout.addWidget(header_widget)

        # Add separator
        separator = QWidget()
        separator.setFixedHeight(2)
        separator.setStyleSheet("background-color: #e0e0e0; border-radius: 1px;")
        layout.addWidget(separator)

        # Scrollable content area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setMinimumHeight(350)
        scroll_area.setMaximumHeight(350)

        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(15)

        # File path info
        path_group = QWidget()
        path_layout = QVBoxLayout()
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.setSpacing(5)

        path_label = QLabel("📂 Source Path:")
        path_label.setStyleSheet("font-weight: bold; color: #333;")
        path_layout.addWidget(path_label)

        path_display = QLabel(self.lora_path)
        path_display.setWordWrap(True)
        path_display.setStyleSheet("""
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 8px;
            color: #666;
            font-family: 'Segoe UI', monospace;
            font-size: 11px;
        """)
        path_layout.addWidget(path_display)

        path_group.setLayout(path_layout)
        content_layout.addWidget(path_group)

        # Metadata form
        form_widget = QWidget()
        form_layout = QVBoxLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(12)

        # Basic Information Section
        basic_group = QGroupBox("Basic Information")
        basic_layout = QVBoxLayout()
        basic_layout.setSpacing(8)

        # Display Name
        display_layout = QVBoxLayout()
        display_layout.setSpacing(3)
        display_label = QLabel("Display Name:")
        display_label.setStyleSheet("font-weight: bold;")
        display_layout.addWidget(display_label)
        self.display_name_edit = QLineEdit()
        self.display_name_edit.setText(self.lora_name)
        self.display_name_edit.setPlaceholderText("User-friendly name for this LoRA")
        self.display_name_edit.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background-color: #ffffff;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #FF6B35;
            }
        """)
        display_layout.addWidget(self.display_name_edit)
        basic_layout.addLayout(display_layout)

        # Base Model Type
        base_model_layout = QVBoxLayout()
        base_model_layout.setSpacing(3)
        base_model_label = QLabel("Base Model Type:")
        base_model_label.setStyleSheet("font-weight: bold;")
        base_model_layout.addWidget(base_model_label)
        self.base_model_combo = QComboBox()
        self.base_model_combo.addItem("Stable Diffusion v1.4", ModelType.STABLE_DIFFUSION_V1_4)
        self.base_model_combo.addItem("Stable Diffusion v1.5", ModelType.STABLE_DIFFUSION_V1_5)
        self.base_model_combo.addItem("Stable Diffusion XL", ModelType.STABLE_DIFFUSION_XL)
        self.base_model_combo.setCurrentIndex(1)  # Default to v1.5
        self.base_model_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background-color: #ffffff;
                font-size: 12px;
            }
            QComboBox:focus {
                border-color: #FF6B35;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
        """)
        base_model_layout.addWidget(self.base_model_combo)
        basic_layout.addLayout(base_model_layout)

        # Description
        desc_layout = QVBoxLayout()
        desc_layout.setSpacing(3)
        desc_label = QLabel("Description:")
        desc_label.setStyleSheet("font-weight: bold;")
        desc_layout.addWidget(desc_label)
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        self.description_edit.setPlaceholderText("Enter a description for this LoRA...")
        self.description_edit.setStyleSheet("""
            QTextEdit {
                padding: 8px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background-color: #ffffff;
                font-size: 12px;
            }
            QTextEdit:focus {
                border-color: #FF6B35;
            }
        """)
        desc_layout.addWidget(self.description_edit)
        basic_layout.addLayout(desc_layout)

        basic_group.setLayout(basic_layout)
        form_layout.addWidget(basic_group)

        # LoRA Specific Section
        lora_group = QGroupBox("LoRA Configuration")
        lora_layout = QVBoxLayout()
        lora_layout.setSpacing(8)

        # Trigger Words
        trigger_layout = QVBoxLayout()
        trigger_layout.setSpacing(3)
        trigger_label = QLabel("Trigger Words:")
        trigger_label.setStyleSheet("font-weight: bold;")
        trigger_layout.addWidget(trigger_label)
        self.trigger_words_edit = QLineEdit()
        self.trigger_words_edit.setPlaceholderText("e.g., character name, style, quality terms")
        self.trigger_words_edit.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background-color: #ffffff;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #FF6B35;
            }
        """)
        trigger_layout.addWidget(self.trigger_words_edit)
        lora_layout.addLayout(trigger_layout)

        # Default Scaling
        scaling_layout = QVBoxLayout()
        scaling_layout.setSpacing(3)
        scaling_label = QLabel("Default Scaling:")
        scaling_label.setStyleSheet("font-weight: bold;")
        scaling_layout.addWidget(scaling_label)
        self.scaling_spin = QDoubleSpinBox()
        self.scaling_spin.setRange(0.0, 2.0)
        self.scaling_spin.setValue(1.0)
        self.scaling_spin.setSingleStep(0.1)
        self.scaling_spin.setToolTip("Default scaling factor for this LoRA (0.0-2.0)")
        self.scaling_spin.setStyleSheet("""
            QDoubleSpinBox {
                padding: 8px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background-color: #ffffff;
                font-size: 12px;
            }
            QDoubleSpinBox:focus {
                border-color: #FF6B35;
            }
        """)
        scaling_layout.addWidget(self.scaling_spin)
        lora_layout.addLayout(scaling_layout)

        lora_group.setLayout(lora_layout)
        form_layout.addWidget(lora_group)

        # Categories Section
        categories_group = QGroupBox("Categories")
        categories_layout = QVBoxLayout()
        categories_layout.setSpacing(8)

        categories_desc = QLabel("Select categories that best describe this LoRA:")
        categories_desc.setStyleSheet("color: #666; font-size: 11px;")
        categories_layout.addWidget(categories_desc)

        self.category_list = QListWidget()
        self.category_list.setMaximumHeight(120)
        self.category_list.setSelectionMode(QListWidget.MultiSelection)
        self.category_list.setStyleSheet("""
            QListWidget {
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background-color: #ffffff;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
                border-radius: 4px;
                margin: 2px;
            }
            QListWidget::item:selected {
                background-color: #FF6B35;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #fff0eb;
            }
        """)

        # Add category options
        for category in ModelCategory:
            item = QListWidgetItem(category.value.title())
            item.setData(1, category.value)  # Store the enum value
            self.category_list.addItem(item)

        categories_layout.addWidget(self.category_list)
        categories_group.setLayout(categories_layout)
        form_layout.addWidget(categories_group)

        # Additional Information Section
        additional_group = QGroupBox("Additional Information")
        additional_layout = QVBoxLayout()
        additional_layout.setSpacing(8)

        # Usage Notes
        usage_layout = QVBoxLayout()
        usage_layout.setSpacing(3)
        usage_label = QLabel("Usage Notes:")
        usage_label.setStyleSheet("font-weight: bold;")
        usage_layout.addWidget(usage_label)
        self.usage_edit = QTextEdit()
        self.usage_edit.setMaximumHeight(60)
        self.usage_edit.setPlaceholderText("Any special usage notes or tips...")
        self.usage_edit.setStyleSheet("""
            QTextEdit {
                padding: 8px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background-color: #ffffff;
                font-size: 12px;
            }
            QTextEdit:focus {
                border-color: #FF6B35;
            }
        """)
        usage_layout.addWidget(self.usage_edit)
        additional_layout.addLayout(usage_layout)

        # Source URL and License in horizontal layout
        urls_layout = QHBoxLayout()
        urls_layout.setSpacing(15)

        # Source URL
        url_layout = QVBoxLayout()
        url_layout.setSpacing(3)
        source_label = QLabel("Source URL:")
        source_label.setStyleSheet("font-weight: bold;")
        url_layout.addWidget(source_label)
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("https://...")
        self.source_edit.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background-color: #ffffff;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #FF6B35;
            }
        """)
        url_layout.addWidget(self.source_edit)
        urls_layout.addLayout(url_layout)

        # License Info
        license_layout = QVBoxLayout()
        license_layout.setSpacing(3)
        license_label = QLabel("License:")
        license_label.setStyleSheet("font-weight: bold;")
        license_layout.addWidget(license_label)
        self.license_edit = QLineEdit()
        self.license_edit.setPlaceholderText("License type or attribution...")
        self.license_edit.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background-color: #ffffff;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #FF6B35;
            }
        """)
        license_layout.addWidget(self.license_edit)
        urls_layout.addLayout(license_layout)

        additional_layout.addLayout(urls_layout)
        additional_group.setLayout(additional_layout)
        form_layout.addWidget(additional_group)

        form_widget.setLayout(form_layout)
        content_layout.addWidget(form_widget)

        content_widget.setLayout(content_layout)
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)

        # Buttons section
        button_widget = QWidget()
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(10)

        button_layout.addStretch()

        # Cancel button
        self.cancel_btn = QPushButton("❌ Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setMinimumWidth(100)
        button_layout.addWidget(self.cancel_btn)

        # Install button
        self.install_btn = QPushButton("✅ Install LoRA")
        self.install_btn.clicked.connect(self.accept)
        self.install_btn.setMinimumHeight(40)
        self.install_btn.setMinimumWidth(140)
        self.install_btn.setDefault(True)
        button_layout.addWidget(self.install_btn)

        button_widget.setLayout(button_layout)
        layout.addWidget(button_widget)

        # Set layout
        self.setLayout(layout)

        # Modern stylesheet
        self.setStyleSheet("""
            QDialog {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 12px;
            }

            QGroupBox {
                font-weight: bold;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #fafafa;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #FF6B35;
                font-size: 13px;
            }

            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 12px;
            }

            QPushButton:hover {
                background-color: #d32f2f;
                transform: translateY(-1px);
            }

            QPushButton:pressed {
                background-color: #b71c1c;
                transform: translateY(0px);
            }

            QPushButton#install_btn {
                background-color: #FF6B35;
            }

            QPushButton#install_btn:hover {
                background-color: #FF8A65;
            }

            QPushButton#install_btn:pressed {
                background-color: #E64A19;
            }
        """)

        # Set object name for install button styling
        self.install_btn.setObjectName("install_btn")

    def accept(self):
        """Handle accept button click."""
        self.accepted = True
        super().accept()

    def reject(self):
        """Handle reject button click."""
        self.accepted = False
        super().reject()

    def _setup_metadata_widgets(self):
        """Set up widgets for collecting LoRA metadata."""
        # Create a widget to hold our custom controls
        widget = QWidget()
        layout = QVBoxLayout()

        # Display Name
        display_name_label = QLabel("Display Name:")
        layout.addWidget(display_name_label)
        self.display_name_edit = QLineEdit()
        self.display_name_edit.setText(self.lora_name)  # Pre-fill with filename
        self.display_name_edit.setPlaceholderText("User-friendly name for this LoRA")
        layout.addWidget(self.display_name_edit)

        # Base Model Type
        base_model_label = QLabel("Base Model Type:")
        layout.addWidget(base_model_label)
        self.base_model_combo = QComboBox()
        self.base_model_combo.addItem("Stable Diffusion v1.4", ModelType.STABLE_DIFFUSION_V1_4)
        self.base_model_combo.addItem("Stable Diffusion v1.5", ModelType.STABLE_DIFFUSION_V1_5)
        self.base_model_combo.addItem("Stable Diffusion XL", ModelType.STABLE_DIFFUSION_XL)
        self.base_model_combo.setCurrentIndex(1)  # Default to v1.5
        layout.addWidget(self.base_model_combo)

        # Description
        desc_label = QLabel("Description:")
        layout.addWidget(desc_label)
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(60)
        self.description_edit.setPlaceholderText("Enter a description for this LoRA...")
        layout.addWidget(self.description_edit)

        # Trigger Words
        trigger_label = QLabel("Trigger Words (comma-separated):")
        layout.addWidget(trigger_label)
        self.trigger_words_edit = QLineEdit()
        self.trigger_words_edit.setPlaceholderText("e.g., character name, style, quality terms")
        layout.addWidget(self.trigger_words_edit)

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

        # Default Scaling
        scaling_label = QLabel("Default Scaling:")
        layout.addWidget(scaling_label)
        self.scaling_spin = QDoubleSpinBox()
        self.scaling_spin.setRange(0.0, 2.0)
        self.scaling_spin.setValue(1.0)
        self.scaling_spin.setSingleStep(0.1)
        self.scaling_spin.setToolTip("Default scaling factor for this LoRA (0.0-2.0)")
        layout.addWidget(self.scaling_spin)

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

        # Get selected base model type
        current_index = self.base_model_combo.currentIndex()
        base_model_type = self.base_model_combo.itemData(current_index)

        # Parse trigger words
        trigger_words_text = self.trigger_words_edit.text().strip()
        trigger_words = [word.strip() for word in trigger_words_text.split(',') if word.strip()]

        return {
            'display_name': self.display_name_edit.text().strip(),
            'base_model_type': base_model_type,
            'description': self.description_edit.toPlainText().strip(),
            'trigger_words': trigger_words,
            'categories': selected_categories,
            'default_scaling': self.scaling_spin.value(),
            'usage_notes': self.usage_edit.toPlainText().strip(),
            'source_url': self.source_edit.text().strip(),
            'license_info': self.license_edit.text().strip()
        }


class LoRAEditDialog(QMessageBox):
    """Dialog for editing existing LoRA adapter metadata."""

    def __init__(self, lora: LoRAInfo, parent=None):
        super().__init__(parent)
        self.lora = lora

        self.setWindowTitle("Edit LoRA Parameters")
        # Remove the default text to prevent overlapping - we'll use a custom layout
        self.setText("")
        self.setInformativeText("")

        # Set minimum width for better layout
        self.setMinimumWidth(700)

        # Add custom widgets for LoRA metadata editing
        self._setup_edit_widgets()

        # Add standard buttons (Save first, then Cancel)
        save_button = self.addButton("Save", QMessageBox.AcceptRole)
        cancel_button = self.addButton("Cancel", QMessageBox.RejectRole)

        # Set default button to Save
        save_button.setDefault(True)
        save_button.setFocus()

        # Ensure proper button behavior
        cancel_button.setAutoDefault(False)

    def _setup_edit_widgets(self):
        """Set up widgets for editing LoRA metadata."""
        # Create a widget to hold our custom controls
        widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Title section
        title_label = QLabel(f"Edit parameters for LoRA '{self.lora.name}'")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #E65100; margin-bottom: 5px;")
        main_layout.addWidget(title_label)

        # Main content area - split into two columns
        content_widget = QWidget()
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # Left column - Basic LoRA information
        left_column = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)

        # LoRA Name (unique identifier)
        name_label = QLabel("LoRA Name:")
        name_label.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(name_label)
        self.name_edit = QLineEdit()
        self.name_edit.setText(str(self.lora.name))
        self.name_edit.setPlaceholderText("Unique name for this LoRA")
        left_layout.addWidget(self.name_edit)

        # Display Name
        display_name_label = QLabel("Display Name:")
        left_layout.addWidget(display_name_label)
        self.display_name_edit = QLineEdit()
        self.display_name_edit.setText(str(self.lora.display_name or ""))
        self.display_name_edit.setPlaceholderText("User-friendly name")
        left_layout.addWidget(self.display_name_edit)

        # Base Model Type
        base_model_label = QLabel("Base Model Type:")
        left_layout.addWidget(base_model_label)
        self.base_model_combo = QComboBox()
        self.base_model_combo.addItem("Stable Diffusion v1.4", ModelType.STABLE_DIFFUSION_V1_4)
        self.base_model_combo.addItem("Stable Diffusion v1.5", ModelType.STABLE_DIFFUSION_V1_5)
        self.base_model_combo.addItem("Stable Diffusion XL", ModelType.STABLE_DIFFUSION_XL)

        # Set current base model type
        if self.lora.base_model_type:
            if self.lora.base_model_type == ModelType.STABLE_DIFFUSION_V1_4:
                self.base_model_combo.setCurrentIndex(0)
            elif self.lora.base_model_type == ModelType.STABLE_DIFFUSION_V1_5:
                self.base_model_combo.setCurrentIndex(1)
            elif self.lora.base_model_type == ModelType.STABLE_DIFFUSION_XL:
                self.base_model_combo.setCurrentIndex(2)
        else:
            self.base_model_combo.setCurrentIndex(1)  # Default to v1.5

        left_layout.addWidget(self.base_model_combo)

        # Description
        desc_label = QLabel("Description:")
        left_layout.addWidget(desc_label)
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        self.description_edit.setPlainText(str(self.lora.description or ""))
        self.description_edit.setPlaceholderText("LoRA description...")
        left_layout.addWidget(self.description_edit)

        left_column.setLayout(left_layout)
        content_layout.addWidget(left_column)

        # Right column - Categories and additional info
        right_column = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)

        # Categories
        cat_label = QLabel("Categories:")
        right_layout.addWidget(cat_label)
        self.category_list = QListWidget()
        self.category_list.setMaximumHeight(100)
        self.category_list.setSelectionMode(QListWidget.MultiSelection)

        # Add category options and pre-select current categories
        current_categories = set(self.lora.categories) if self.lora.categories else set()
        for category in ModelCategory:
            item = QListWidgetItem(category.value.title())
            item.setData(1, category.value)  # Store the enum value
            if category in current_categories:
                item.setSelected(True)
            self.category_list.addItem(item)

        right_layout.addWidget(self.category_list)

        # Trigger Words
        trigger_label = QLabel("Trigger Words:")
        right_layout.addWidget(trigger_label)
        self.trigger_words_edit = QLineEdit()
        self.trigger_words_edit.setText(", ".join(self.lora.trigger_words) if self.lora.trigger_words else "")
        self.trigger_words_edit.setPlaceholderText("e.g., character name, style, quality terms")
        right_layout.addWidget(self.trigger_words_edit)

        # Default Scaling
        scaling_label = QLabel("Default Scaling:")
        right_layout.addWidget(scaling_label)
        self.scaling_spin = QDoubleSpinBox()
        self.scaling_spin.setRange(0.0, 2.0)
        self.scaling_spin.setValue(float(self.lora.default_scaling))
        self.scaling_spin.setSingleStep(0.1)
        self.scaling_spin.setToolTip("Default scaling factor for this LoRA (0.0-2.0)")
        right_layout.addWidget(self.scaling_spin)

        # Usage Notes
        usage_label = QLabel("Usage Notes:")
        right_layout.addWidget(usage_label)
        self.usage_edit = QTextEdit()
        self.usage_edit.setMaximumHeight(60)
        self.usage_edit.setPlainText(str(self.lora.usage_notes or ""))
        self.usage_edit.setPlaceholderText("Usage tips...")
        right_layout.addWidget(self.usage_edit)

        # Source URL and License in horizontal layout
        urls_layout = QHBoxLayout()
        urls_layout.setSpacing(10)

        # Source URL
        url_widget = QWidget()
        url_layout = QVBoxLayout()
        url_layout.setSpacing(2)
        source_label = QLabel("Source URL:")
        url_layout.addWidget(source_label)
        self.source_edit = QLineEdit()
        self.source_edit.setText(str(self.lora.source_url or ""))
        self.source_edit.setPlaceholderText("https://...")
        url_layout.addWidget(self.source_edit)
        url_widget.setLayout(url_layout)
        urls_layout.addWidget(url_widget)

        # License Info
        license_widget = QWidget()
        license_layout = QVBoxLayout()
        license_layout.setSpacing(2)
        license_label = QLabel("License:")
        license_layout.addWidget(license_label)
        self.license_edit = QLineEdit()
        self.license_edit.setText(str(self.lora.license_info or ""))
        self.license_edit.setPlaceholderText("License info...")
        license_layout.addWidget(self.license_edit)
        license_widget.setLayout(license_layout)
        urls_layout.addWidget(license_widget)

        right_layout.addLayout(urls_layout)

        right_column.setLayout(right_layout)
        content_layout.addWidget(right_column)

        content_widget.setLayout(content_layout)
        main_layout.addWidget(content_widget)

        widget.setLayout(main_layout)
        self.layout().addWidget(widget, 1, 0, 1, self.layout().columnCount())

    def get_metadata(self) -> dict:
        """Get the edited LoRA metadata."""
        # Get selected categories
        selected_categories = []
        for i in range(self.category_list.count()):
            item = self.category_list.item(i)
            if item.isSelected():
                selected_categories.append(item.data(1))  # Get the enum value

        # Get selected base model type
        current_index = self.base_model_combo.currentIndex()
        base_model_type = self.base_model_combo.itemData(current_index)

        # Parse trigger words
        trigger_words_text = self.trigger_words_edit.text().strip()
        trigger_words = [word.strip() for word in trigger_words_text.split(',') if word.strip()]

        return {
            'name': self.name_edit.text().strip(),
            'display_name': self.display_name_edit.text().strip(),
            'base_model_type': base_model_type,
            'description': self.description_edit.toPlainText().strip(),
            'trigger_words': trigger_words,
            'categories': selected_categories,
            'default_scaling': self.scaling_spin.value(),
            'usage_notes': self.usage_edit.toPlainText().strip(),
            'source_url': self.source_edit.text().strip(),
            'license_info': self.license_edit.text().strip()
        }
