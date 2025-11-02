"""
Settings panel with sidebar and main area.
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QGroupBox, QDialog, QRadioButton, QButtonGroup, QHBoxLayout, QSpinBox
from PyQt5.QtWidgets import QMessageBox, QDialogButtonBox, QLineEdit, QTextEdit, QFileDialog, QCheckBox, QComboBox

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.settings import SettingsManager, AppSettings
from models.model_info import ModelType
from services.openai_service import OpenAIService


class SettingsPanel:
    """Panel for application settings."""

    def __init__(self, settings_manager: SettingsManager, model_manager=None):
        self.settings_manager = settings_manager
        self.model_manager = model_manager

        # Create sidebar and main area
        self.sidebar = self._create_sidebar()
        self.main_area = self._create_main_area()

    def _create_sidebar(self) -> QWidget:
        """Create the sidebar widget."""
        widget = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Application Settings"))

        theme_btn = QPushButton('🎨 Theme Settings')
        theme_btn.clicked.connect(self._show_theme_dialog)
        layout.addWidget(theme_btn)

        perf_btn = QPushButton('⚡ Performance')
        perf_btn.clicked.connect(lambda: QMessageBox.information(widget, "Performance",
                                                               "Performance settings would open here.\n\n"
                                                               "Future features:\n"
                                                               "- Memory optimization\n"
                                                               "- GPU settings\n"
                                                               "- Cache management"))
        layout.addWidget(perf_btn)

        text_btn = QPushButton('✨ Text Enhancement')
        text_btn.clicked.connect(self._show_text_enhancement_dialog)
        layout.addWidget(text_btn)

        timeout_btn = QPushButton('⏱️ Auto-Unload Settings')
        timeout_btn.clicked.connect(self._show_model_timeout_dialog)
        layout.addWidget(timeout_btn)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _show_theme_dialog(self):
        """Show the theme selection dialog."""
        dialog = ThemeSelectionDialog(self.settings_manager, self.sidebar)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            # Refresh the main area to show updated theme
            self._refresh_main_area()

    def _show_text_enhancement_dialog(self):
        """Show the text enhancement settings dialog."""
        dialog = TextEnhancementDialog(self.settings_manager, self.model_manager, self.sidebar)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            # Refresh the main area to show updated settings
            self._refresh_main_area()

    def _show_model_timeout_dialog(self):
        """Show the model timeout settings dialog."""
        dialog = ModelTimeoutDialog(self.settings_manager, self.sidebar)
        dialog.exec_()

    def _refresh_main_area(self):
        """Refresh the main area to show updated settings."""
        # Recreate the main area widget
        old_main_area = self.main_area
        self.main_area = self._create_main_area()

        # Find the parent widget and replace the main area
        if hasattr(self, '_parent_widget'):
            parent_layout = self._parent_widget.layout()
            if parent_layout:
                # Find the index of the old main area
                for i in range(parent_layout.count()):
                    if parent_layout.itemAt(i).widget() == old_main_area:
                        parent_layout.replaceWidget(old_main_area, self.main_area)
                        old_main_area.deleteLater()
                        break

    def _create_main_area(self) -> QWidget:
        """Create the main area widget."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Current Settings Overview
        overview_group = QGroupBox("Current Settings")
        overview_layout = QVBoxLayout()

        # Load current settings
        settings = self.settings_manager.load_settings()

        theme_label = QLabel(f"Theme: {settings.theme}")
        overview_layout.addWidget(theme_label)

        perf_label = QLabel(f"Performance Mode: {settings.performance_mode}")
        overview_layout.addWidget(perf_label)

        res_label = QLabel(f"Output Resolution: {settings.output_resolution}")
        overview_layout.addWidget(res_label)

        cache_label = QLabel(f"Cache Location: {settings.cache_location}")
        overview_layout.addWidget(cache_label)

        if settings.default_model_folder:
            folder_label = QLabel(f"Default Model Folder: {settings.default_model_folder}")
            folder_label.setWordWrap(True)
            overview_layout.addWidget(folder_label)
        else:
            folder_label = QLabel("Default Model Folder: Not set")
            overview_layout.addWidget(folder_label)

        overview_group.setLayout(overview_layout)
        layout.addWidget(overview_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget


class ModelTimeoutDialog(QDialog):
    """Dialog for configuring model timeout settings."""

    def __init__(self, settings_manager: SettingsManager, parent=None):
        super().__init__(parent)
        self.settings_manager = settings_manager

        self.setWindowTitle("Auto-Unload Settings")
        self.setModal(True)
        self.setFixedSize(400, 300)

        self._init_ui()

    def _init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(20)

        # Title
        title_label = QLabel("Configure Auto-Unload Settings")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #1976D2;")
        layout.addWidget(title_label)

        # Load current settings
        current_settings = self.settings_manager.load_settings()

        # Image model timeout
        image_model_group = QGroupBox("Image Model")
        image_model_layout = QVBoxLayout()
        self.image_model_timeout_checkbox = QCheckBox("Enable auto-unload")
        self.image_model_timeout_checkbox.setChecked(current_settings.model_timeout_enabled)
        image_model_layout.addWidget(self.image_model_timeout_checkbox)

        self.image_model_timeout_spinbox = QSpinBox()
        self.image_model_timeout_spinbox.setRange(1, 120) # 1 to 120 minutes
        self.image_model_timeout_spinbox.setValue(current_settings.model_timeout_minutes)
        self.image_model_timeout_spinbox.setSuffix(" minutes")
        image_model_layout.addWidget(self.image_model_timeout_spinbox)
        image_model_group.setLayout(image_model_layout)
        layout.addWidget(image_model_group)

        # Text model timeout
        text_model_group = QGroupBox("Text Model")
        text_model_layout = QVBoxLayout()
        self.text_model_timeout_checkbox = QCheckBox("Enable auto-unload")
        self.text_model_timeout_checkbox.setChecked(current_settings.text_model_timeout_enabled)
        text_model_layout.addWidget(self.text_model_timeout_checkbox)

        self.text_model_timeout_spinbox = QSpinBox()
        self.text_model_timeout_spinbox.setRange(1, 120) # 1 to 120 minutes
        self.text_model_timeout_spinbox.setValue(current_settings.text_model_timeout_minutes)
        self.text_model_timeout_spinbox.setSuffix(" minutes")
        text_model_layout.addWidget(self.text_model_timeout_spinbox)
        text_model_group.setLayout(text_model_layout)
        layout.addWidget(text_model_group)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._save_settings)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def _save_settings(self):
        """Save the timeout settings."""
        settings = self.settings_manager.load_settings()
        settings.model_timeout_enabled = self.image_model_timeout_checkbox.isChecked()
        settings.model_timeout_minutes = self.image_model_timeout_spinbox.value()
        settings.text_model_timeout_enabled = self.text_model_timeout_checkbox.isChecked()
        settings.text_model_timeout_minutes = self.text_model_timeout_spinbox.value()
        self.settings_manager.save_settings(settings)
        self.accept()


class ThemeSelectionDialog(QDialog):
    """Dialog for selecting application theme."""

    def __init__(self, settings_manager: SettingsManager, parent=None):
        super().__init__(parent)
        self.settings_manager = settings_manager

        self.setWindowTitle("Theme Settings")
        self.setModal(True)
        self.setFixedSize(400, 250)

        # Center the dialog relative to the main window
        main_window = self._find_main_window()
        if main_window:
            main_rect = main_window.geometry()
            self.move(
                main_rect.x() + (main_rect.width() - self.width()) // 2,
                main_rect.y() + (main_rect.height() - self.height()) // 2
            )

        self._init_ui()

    def _init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(20)

        # Title
        title_label = QLabel("Choose Application Theme")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #1976D2;")
        layout.addWidget(title_label)

        # Theme options
        theme_group = QGroupBox("Theme Options")
        theme_layout = QVBoxLayout()
        theme_layout.setSpacing(10)

        # Load current settings
        current_settings = self.settings_manager.load_settings()

        # Radio buttons for theme selection
        self.theme_group = QButtonGroup()

        # Light theme
        light_radio = QRadioButton("☀️ Light Theme")
        light_radio.setToolTip("Clean white backgrounds with dark text")
        self.theme_group.addButton(light_radio, 0)
        theme_layout.addWidget(light_radio)

        # Dark theme
        dark_radio = QRadioButton("🌙 Dark Theme")
        dark_radio.setToolTip("Dark backgrounds with light text")
        self.theme_group.addButton(dark_radio, 1)
        theme_layout.addWidget(dark_radio)

        # Default theme
        default_radio = QRadioButton("⚙️ Default Theme")
        default_radio.setToolTip("System default appearance")
        self.theme_group.addButton(default_radio, 2)
        theme_layout.addWidget(default_radio)

        # Set current selection based on saved theme
        if current_settings.theme == "light":
            light_radio.setChecked(True)
        elif current_settings.theme == "dark":
            dark_radio.setChecked(True)
        else:
            default_radio.setChecked(True)

        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)

        # Preview section
        preview_label = QLabel("Theme changes will be applied immediately.")
        preview_label.setStyleSheet("color: #666; font-size: 11px; font-style: italic;")
        layout.addWidget(preview_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        apply_btn = QPushButton("Apply Theme")
        apply_btn.clicked.connect(self._apply_theme)
        apply_btn.setDefault(True)
        button_layout.addWidget(apply_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _apply_theme(self):
        """Apply the selected theme."""
        selected_id = self.theme_group.checkedId()

        # Map button IDs to theme names
        theme_map = {
            0: "light",
            1: "dark",
            2: "default"
        }

        selected_theme = theme_map.get(selected_id, "default")

        # Save the theme setting
        settings = self.settings_manager.load_settings()
        settings.theme = selected_theme
        self.settings_manager.save_settings(settings)

        # Apply the theme to the application
        self._apply_theme_to_app(selected_theme)

        # Accept the dialog
        self.accept()

    def _apply_theme_to_app(self, theme: str):
        """Apply the theme to the entire application."""
        # Get the main application window
        main_window = self._find_main_window()

        if main_window:
            if theme == "light":
                stylesheet = self._get_light_theme_stylesheet()
            elif theme == "dark":
                stylesheet = self._get_dark_theme_stylesheet()
            else:  # default
                stylesheet = ""

            main_window.setStyleSheet(stylesheet)

            # Force a repaint of all widgets
            main_window.update()
            for child in main_window.findChildren(QWidget):
                child.update()

    def _find_main_window(self):
        """Find the main application window."""
        # Walk up the parent hierarchy to find the main window
        current = self.parent()
        while current:
            if hasattr(current, 'setStyleSheet'):  # Main window should have this method
                # Check if it's the main window by looking for typical main window attributes
                if hasattr(current, 'centralWidget') or hasattr(current, 'menuBar'):
                    return current
            current = current.parent()
        return None

    def _get_light_theme_stylesheet(self) -> str:
        """Get the light theme stylesheet."""
        return """
            /* Light Theme Styles */
            QWidget {
                background-color: #ffffff;
                color: #333333;
            }

            QGroupBox {
                font-weight: bold;
                border: 2px solid #e0e0e0;
                border-radius: 5px;
                margin-top: 1ex;
                background-color: #fafafa;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #1976D2;
                font-weight: bold;
            }

            QPushButton {
                background-color: #f5f5f5;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 5px 10px;
                color: #333333;
            }

            QPushButton:hover {
                background-color: #e8f4fd;
                border-color: #1976D2;
            }

            QPushButton:pressed {
                background-color: #d1e7f5;
            }

            QLineEdit, QTextEdit, QComboBox {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 3px;
                color: #333333;
            }

            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border-color: #1976D2;
            }

            QListWidget {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                color: #333333;
            }

            QListWidget::item:selected {
                background-color: #e8f4fd;
                color: #1976D2;
            }

            QScrollBar:vertical {
                background-color: #f5f5f5;
                width: 12px;
            }

            QScrollBar::handle:vertical {
                background-color: #cccccc;
                border-radius: 6px;
            }

            QScrollBar::handle:vertical:hover {
                background-color: #aaaaaa;
            }
        """


    def _get_dark_theme_stylesheet(self) -> str:
        return """
            /* Dark Theme Styles */
            QWidget {
                background-color: #2b2b2b;
                color: #e0e0e0;
            }

            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 1ex;
                background-color: #3a3a3a;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #64b5f6;
                font-weight: bold;
            }

            QPushButton {
                background-color: #404040;
                border: 1px solid #666666;
                border-radius: 4px;
                padding: 5px 10px;
                color: #e0e0e0;
            }

            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #64b5f6;
            }

            QPushButton:pressed {
                background-color: #333333;
            }

            QLineEdit, QTextEdit, QComboBox {
                background-color: #404040;
                border: 1px solid #666666;
                border-radius: 3px;
                padding: 3px;
                color: #e0e0e0;
            }

            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border-color: #64b5f6;
            }

            QListWidget {
                background-color: #404040;
                border: 1px solid #666666;
                color: #e0e0e0;
            }

            QListWidget::item:selected {
                background-color: #4a4a4a;
                color: #64b5f6;
            }

            QListWidget::item:hover {
                background-color: #454545;
            }

            QScrollBar:vertical {
                background-color: #3a3a3a;
                width: 12px;
            }

            QScrollBar::handle:vertical {
                background-color: #666666;
                border-radius: 6px;
            }

            QScrollBar::handle:vertical:hover {
                background-color: #888888;
            }

            QLabel {
                color: #e0e0e0;
            }

            QTextEdit {
                background-color: #404040;
                color: #e0e0e0;
            }
        """


class TextEnhancementDialog(QDialog):
    """Dialog for configuring text enhancement settings."""

    def __init__(self, settings_manager: SettingsManager, model_manager=None, parent=None):
        super().__init__(parent)
        self.settings_manager = settings_manager
        self.model_manager = model_manager

        self.setWindowTitle("Text Enhancement Settings")
        self.setModal(True)

        # Make dialog resizable with minimum and default sizes
        self.setMinimumSize(550, 650)
        self.resize(600, 700)  # Default size - larger to fit all content

        # Center the dialog relative to the main window
        main_window = self._find_main_window()
        if main_window:
            main_rect = main_window.geometry()
            self.move(
                main_rect.x() + (main_rect.width() - self.width()) // 2,
                main_rect.y() + (main_rect.height() - self.height()) // 2
            )

        self._init_ui()

    def _init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title_label = QLabel("✨ Text Enhancement Settings")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #1976D2;")
        layout.addWidget(title_label)

        # Load current settings
        current_settings = self.settings_manager.load_settings()

        # Model path section
        model_group = QGroupBox("Text Model Configuration")
        model_layout = QVBoxLayout()
        model_layout.setContentsMargins(15, 15, 15, 15)
        model_layout.setSpacing(10)

        # Available text models dropdown
        if self.model_manager:
            available_models = self.model_manager.get_installed_models()
            text_models = [model for model in available_models if model.model_type == ModelType.TEXT_MODEL]

            if text_models:
                available_label = QLabel("Available Text Models:")
                available_label.setStyleSheet("font-weight: bold;")
                model_layout.addWidget(available_label)

                self.available_models_combo = QComboBox()
                self.available_models_combo.setToolTip("Select from installed text models")
                self.available_models_combo.addItem("Select installed model...", "")

                for model in text_models:
                    display_name = model.display_name or model.name
                    self.available_models_combo.addItem(f"📄 {display_name}", model.path)

                self.available_models_combo.currentTextChanged.connect(self._on_available_model_selected)
                model_layout.addWidget(self.available_models_combo)

                # Separator
                separator = QWidget()
                separator.setFixedHeight(10)
                model_layout.addWidget(separator)

        model_path_label = QLabel("Text Model Path:")
        model_path_label.setToolTip("Path to the local text model file (e.g., GPT-2, Llama model)")
        model_layout.addWidget(model_path_label)

        # Model path input with browse buttons
        path_layout = QHBoxLayout()
        self.model_path_input = QLineEdit()
        self.model_path_input.setText(current_settings.text_model_path or "")
        self.model_path_input.setPlaceholderText("Enter model identifier (e.g., gpt2-medium) or select file/folder...")
        path_layout.addWidget(self.model_path_input)

        # Browse file button (for GGUF models)
        browse_file_btn = QPushButton("Browse File...")
        browse_file_btn.clicked.connect(self._browse_model_file)
        browse_file_btn.setToolTip("Select a GGUF model file")
        path_layout.addWidget(browse_file_btn)

        # Browse folder button (for transformers models)
        browse_folder_btn = QPushButton("Browse Folder...")
        browse_folder_btn.clicked.connect(self._browse_model_folder)
        browse_folder_btn.setToolTip("Select a transformers model folder")
        path_layout.addWidget(browse_folder_btn)

        model_layout.addLayout(path_layout)

        # Model type info
        model_type_label = QLabel("For transformers models, enter the model identifier (e.g., 'gpt2-medium') or select the model folder.\nFor GGUF models, select the .gguf file, or choose from installed text models above.")
        model_type_label.setStyleSheet("color: #666; font-size: 11px; font-style: italic;")
        model_type_label.setWordWrap(True)
        model_layout.addWidget(model_type_label)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # OpenAI API section
        openai_group = QGroupBox("OpenAI API Configuration")
        openai_layout = QVBoxLayout()
        openai_layout.setContentsMargins(15, 15, 15, 15)
        openai_layout.setSpacing(10)

        # Enable OpenAI checkbox and model selection
        openai_controls_layout = QHBoxLayout()

        self.enable_openai_checkbox = QCheckBox("Enable OpenAI API for text enhancement")
        self.enable_openai_checkbox.setChecked(current_settings.enable_openai_enhancement)
        self.enable_openai_checkbox.setToolTip("When enabled, OpenAI API will be used instead of local models for prompt enhancement")
        openai_controls_layout.addWidget(self.enable_openai_checkbox)

        # Model selection dropdown
        model_label = QLabel("Model:")
        model_label.setToolTip("Select the OpenAI model to use for text enhancement")
        openai_controls_layout.addWidget(model_label)

        self.openai_model_combo = QComboBox()
        self.openai_model_combo.setToolTip("Select the OpenAI model to use for text enhancement")
        self.openai_model_combo.setMinimumWidth(200)
        openai_controls_layout.addWidget(self.openai_model_combo)

        # Refresh models button
        refresh_btn = QPushButton("🔄")
        refresh_btn.setToolTip("Refresh available models from OpenAI API")
        refresh_btn.clicked.connect(self._refresh_openai_models)
        refresh_btn.setMaximumWidth(30)
        openai_controls_layout.addWidget(refresh_btn)

        openai_layout.addLayout(openai_controls_layout)

        # API Key input
        api_key_label = QLabel("OpenAI API Key:")
        api_key_label.setToolTip("Your OpenAI API key (starts with 'sk-')")
        openai_layout.addWidget(api_key_label)

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)  # Hide the API key
        self.api_key_input.setText(current_settings.openai_api_key or "")
        self.api_key_input.setPlaceholderText("sk-...")
        openai_layout.addWidget(self.api_key_input)

        # API key info
        api_key_info = QLabel("Get your API key from https://platform.openai.com/api-keys\nNote: API calls incur costs. GPT-3.5-turbo is used for cost efficiency.")
        api_key_info.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        api_key_info.setWordWrap(True)
        openai_layout.addWidget(api_key_info)

        openai_group.setLayout(openai_layout)
        layout.addWidget(openai_group)

        # Info section
        info_label = QLabel("Choose between local models (private, free) or OpenAI API (high-quality, requires API key).\nThe enhancement prompt tells the AI how to improve user prompts for better image generation.")
        info_label.setStyleSheet("color: #666; font-size: 11px; font-style: italic;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._save_settings)
        save_btn.setDefault(True)
        button_layout.addWidget(save_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Load OpenAI models after UI is set up
        self._load_openai_models()

    def _load_openai_models(self):
        """Load available OpenAI models into the combo box."""
        try:
            openai_service = OpenAIService()
            models = openai_service.get_available_models()

            self.openai_model_combo.clear()

            if not models:
                # No models in database, show placeholder
                self.openai_model_combo.addItem("No models loaded - enter API key and refresh", "")
                self.openai_model_combo.setEnabled(False)
                return

            # Enable the combo box
            self.openai_model_combo.setEnabled(True)

            # Add models to combo box
            current_model = self.settings_manager.load_settings().openai_model or "gpt-3.5-turbo"
            selected_index = 0

            for i, model in enumerate(models):
                display_text = model.get('display_name', model['model_id'])
                self.openai_model_combo.addItem(display_text, model['model_id'])

                if model['model_id'] == current_model:
                    selected_index = i

            # Set current selection
            if self.openai_model_combo.count() > 0:
                self.openai_model_combo.setCurrentIndex(selected_index)

        except Exception as e:
            print(f"Error loading OpenAI models: {e}")
            self.openai_model_combo.clear()
            self.openai_model_combo.addItem("Error loading models", "")
            self.openai_model_combo.setEnabled(False)

    def _refresh_openai_models(self):
        """Refresh OpenAI models from API."""
        api_key = self.api_key_input.text().strip()

        if not api_key:
            QMessageBox.warning(
                self,
                "API Key Required",
                "Please enter your OpenAI API key before refreshing models."
            )
            return

        # Show progress
        refresh_btn = self.sender()
        original_text = refresh_btn.text()
        refresh_btn.setText("⏳")
        refresh_btn.setEnabled(False)

        try:
            openai_service = OpenAIService()
            success, message = openai_service.fetch_and_store_models(api_key)

            if success:
                QMessageBox.information(
                    self,
                    "Models Refreshed",
                    message
                )
                # Reload models in combo box
                self._load_openai_models()
            else:
                QMessageBox.warning(
                    self,
                    "Refresh Failed",
                    f"Failed to refresh models: {message}"
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while refreshing models: {str(e)}"
            )

        finally:
            # Restore button
            refresh_btn.setText(original_text)
            refresh_btn.setEnabled(True)

    def _browse_model_file(self):
        """Open file dialog to browse for text model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Text Model File",
            "",
            "Model Files (*.bin *.pt *.pth *.safetensors *.gguf);;All Files (*)"
        )

        if file_path:
            self.model_path_input.setText(file_path)

    def _browse_model_folder(self):
        """Open folder dialog to browse for transformers model folder."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Transformers Model Folder",
            "./models",  # Start in models directory
            QFileDialog.ShowDirsOnly
        )

        if folder_path:
            # Extract model name from folder path
            # E.g., "./models/models--gpt2-medium" -> "gpt2-medium"
            folder_name = os.path.basename(folder_path)
            if folder_name.startswith('models--'):
                model_name = folder_name.replace('models--', '').replace('-', '/')
                self.model_path_input.setText(model_name)
            else:
                # If it's not a standard huggingface folder, use the path as-is
                self.model_path_input.setText(folder_path)

    def _save_settings(self):
        """Save the text enhancement settings."""
        # Get current settings
        settings = self.settings_manager.load_settings()

        # Update with new values
        settings.text_model_path = self.model_path_input.text().strip() or None
        settings.enable_openai_enhancement = self.enable_openai_checkbox.isChecked()
        settings.openai_api_key = self.api_key_input.text().strip() or None

        # Get selected OpenAI model
        if self.openai_model_combo.currentData():
            settings.openai_model = self.openai_model_combo.currentData()
        else:
            # Fallback to default if no model selected
            settings.openai_model = "gpt-3.5-turbo"

        # Validate OpenAI settings
        if settings.enable_openai_enhancement and not settings.openai_api_key:
            QMessageBox.warning(
                self,
                "Validation Error",
                "OpenAI API key is required when OpenAI enhancement is enabled."
            )
            return

        # Save settings
        self.settings_manager.save_settings(settings)

        # Accept the dialog
        self.accept()

    def _on_available_model_selected(self, text):
        """Handle selection from available models dropdown."""
        if hasattr(self, 'available_models_combo') and self.available_models_combo.currentData():
            selected_path = self.available_models_combo.currentData()
            if selected_path:  # Not the placeholder
                self.model_path_input.setText(selected_path)

    def _find_main_window(self):
        """Find the main application window."""
        # Walk up the parent hierarchy to find the main window
        current = self.parent()
        while current:
            if hasattr(current, 'setStyleSheet'):  # Main window should have this method
                # Check if it's the main window by looking for typical main window attributes
                if hasattr(current, 'centralWidget') or hasattr(current, 'menuBar'):
                    return current
            current = current.parent()
        return None
