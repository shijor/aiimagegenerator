"""
Unified dialog for manual installation of models, LoRAs, and IP-Adapters.
"""
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QScrollArea, QGroupBox, QLineEdit, QTextEdit, QListWidget,
                             QListWidgetItem, QWidget, QComboBox, QFileDialog, QMessageBox,
                             QDoubleSpinBox)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from models.model_info import ModelCategory, ModelType, LoRAInfo, IPAdapterInfo


class ManualInstallDialog(QDialog):
    """Unified dialog for manual installation of different model types."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Manual Model Installation")
        self.setModal(True)
        self.setFixedSize(800, 700)

        # Center the dialog
        if parent:
            parent_rect = parent.geometry()
            self.move(
                parent_rect.x() + (parent_rect.width() - self.width()) // 2,
                parent_rect.y() + (parent_rect.height() - self.height()) // 2
            )

        self.accepted = False  # Track if user accepted
        self.current_type = "Model"  # Default type
        self._init_ui()

    def _init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header section
        header_widget = QWidget()
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(15)

        # Icon
        self.icon_label = QLabel("📦")
        icon_font = QFont()
        icon_font.setPointSize(32)
        self.icon_label.setFont(icon_font)
        header_layout.addWidget(self.icon_label)

        # Title and type selector
        title_widget = QWidget()
        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(5)

        title_label = QLabel("Manual Model Installation")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #1976D2;")
        title_layout.addWidget(title_label)

        # Type selector
        type_layout = QHBoxLayout()
        type_label = QLabel("Install Type:")
        type_label.setStyleSheet("font-weight: bold;")
        type_layout.addWidget(type_label)

        self.type_combo = QComboBox()
        self.type_combo.addItems(["Model", "LoRA", "IP-Adapter"])
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        self.type_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 2px solid #e0e0e0;
                border-radius: 4px;
                background-color: #ffffff;
                min-width: 120px;
            }
            QComboBox:focus {
                border-color: #1976D2;
            }
        """)
        type_layout.addWidget(self.type_combo)
        type_layout.addStretch()
        title_layout.addLayout(type_layout)

        title_widget.setLayout(title_layout)
        header_layout.addWidget(title_widget, stretch=1)

        header_widget.setLayout(header_layout)
        layout.addWidget(header_widget)

        # Separator
        separator = QWidget()
        separator.setFixedHeight(2)
        separator.setStyleSheet("background-color: #e0e0e0; border-radius: 1px;")
        layout.addWidget(separator)

        # Scrollable content area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setMinimumHeight(450)
        scroll_area.setMaximumHeight(450)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(15)

        # File selection section (common to all types)
        self._create_file_selection_section()

        # Dynamic form sections
        self.form_sections = {}
        self._create_model_form()
        self._create_lora_form()
        self._create_ip_adapter_form()

        # Show default form
        self._show_form_for_type("Model")

        self.content_widget.setLayout(self.content_layout)
        scroll_area.setWidget(self.content_widget)
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
        self.install_btn = QPushButton("✅ Install")
        self.install_btn.clicked.connect(self.accept)
        self.install_btn.setMinimumHeight(40)
        self.install_btn.setMinimumWidth(120)
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

    def _create_file_selection_section(self):
        """Create the file selection section common to all types."""
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        file_layout.setSpacing(8)

        # File path input
        path_layout = QHBoxLayout()
        path_layout.setSpacing(10)

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select or enter the model file path...")
        self.file_path_edit.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background-color: #ffffff;
                font-size: 12px;
                font-family: 'Segoe UI', monospace;
            }
            QLineEdit:focus {
                border-color: #1976D2;
            }
        """)
        path_layout.addWidget(self.file_path_edit)

        browse_btn = QPushButton("📁 Browse")
        browse_btn.clicked.connect(self._browse_file)
        browse_btn.setFixedWidth(80)
        path_layout.addWidget(browse_btn)

        file_layout.addLayout(path_layout)

        # File info display
        self.file_info_label = QLabel("No file selected")
        self.file_info_label.setStyleSheet("color: #666; font-size: 11px; margin-top: 5px;")
        file_layout.addWidget(self.file_info_label)

        file_group.setLayout(file_layout)
        self.content_layout.addWidget(file_group)

    def _create_model_form(self):
        """Create the form section for Model installation."""
        form_widget = QWidget()
        form_layout = QVBoxLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(12)

        # Basic Information Section
        basic_group = QGroupBox("Basic Information")
        basic_layout = QVBoxLayout()
        basic_layout.setSpacing(8)

        # Model Name
        name_layout = QVBoxLayout()
        name_layout.setSpacing(3)
        name_label = QLabel("Model Name:")
        name_label.setStyleSheet("font-weight: bold;")
        name_layout.addWidget(name_label)
        self.model_name_edit = QLineEdit()
        self.model_name_edit.setPlaceholderText("Unique identifier for this model")
        self.model_name_edit.setStyleSheet(self._get_input_style())
        name_layout.addWidget(self.model_name_edit)
        basic_layout.addLayout(name_layout)

        # Display Name
        display_layout = QVBoxLayout()
        display_layout.setSpacing(3)
        display_label = QLabel("Display Name:")
        display_label.setStyleSheet("font-weight: bold;")
        display_layout.addWidget(display_label)
        self.model_display_name_edit = QLineEdit()
        self.model_display_name_edit.setPlaceholderText("User-friendly name for this model")
        self.model_display_name_edit.setStyleSheet(self._get_input_style())
        display_layout.addWidget(self.model_display_name_edit)
        basic_layout.addLayout(display_layout)

        # Model Type
        type_layout = QVBoxLayout()
        type_layout.setSpacing(3)
        type_label = QLabel("Model Type:")
        type_label.setStyleSheet("font-weight: bold;")
        type_layout.addWidget(type_label)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems([t.value for t in ModelType])
        self.model_type_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 2px solid #e0e0e0;
                border-radius: 4px;
                background-color: #ffffff;
            }
            QComboBox:focus {
                border-color: #1976D2;
            }
        """)
        type_layout.addWidget(self.model_type_combo)
        basic_layout.addLayout(type_layout)

        # Description
        desc_layout = QVBoxLayout()
        desc_layout.setSpacing(3)
        desc_label = QLabel("Description:")
        desc_label.setStyleSheet("font-weight: bold;")
        desc_layout.addWidget(desc_label)
        self.model_description_edit = QTextEdit()
        self.model_description_edit.setMaximumHeight(80)
        self.model_description_edit.setPlaceholderText("Enter a description for this model...")
        self.model_description_edit.setStyleSheet(self._get_textarea_style())
        desc_layout.addWidget(self.model_description_edit)
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

        self.model_category_list = QListWidget()
        self.model_category_list.setMaximumHeight(120)
        self.model_category_list.setSelectionMode(QListWidget.MultiSelection)
        self.model_category_list.setStyleSheet(self._get_list_style())

        # Add category options
        for category in ModelCategory:
            item = QListWidgetItem(category.value.title())
            item.setData(1, category.value)  # Store the enum value
            self.model_category_list.addItem(item)

        categories_layout.addWidget(self.model_category_list)
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
        self.model_usage_edit = QTextEdit()
        self.model_usage_edit.setMaximumHeight(60)
        self.model_usage_edit.setPlaceholderText("Any special usage notes or tips...")
        self.model_usage_edit.setStyleSheet(self._get_textarea_style())
        usage_layout.addWidget(self.model_usage_edit)
        additional_layout.addLayout(usage_layout)

        # URLs and License
        urls_layout = QHBoxLayout()
        urls_layout.setSpacing(15)

        # Source URL
        url_layout = QVBoxLayout()
        url_layout.setSpacing(3)
        source_label = QLabel("Source URL:")
        source_label.setStyleSheet("font-weight: bold;")
        url_layout.addWidget(source_label)
        self.model_source_edit = QLineEdit()
        self.model_source_edit.setPlaceholderText("https://...")
        self.model_source_edit.setStyleSheet(self._get_input_style())
        url_layout.addWidget(self.model_source_edit)
        urls_layout.addLayout(url_layout)

        # License Info
        license_layout = QVBoxLayout()
        license_layout.setSpacing(3)
        license_label = QLabel("License:")
        license_label.setStyleSheet("font-weight: bold;")
        license_layout.addWidget(license_label)
        self.model_license_edit = QLineEdit()
        self.model_license_edit.setPlaceholderText("License type or attribution...")
        self.model_license_edit.setStyleSheet(self._get_input_style())
        license_layout.addWidget(self.model_license_edit)
        urls_layout.addLayout(license_layout)

        additional_layout.addLayout(urls_layout)
        additional_group.setLayout(additional_layout)
        form_layout.addWidget(additional_group)

        form_widget.setLayout(form_layout)
        self.form_sections["Model"] = form_widget

    def _create_lora_form(self):
        """Create the form section for LoRA installation."""
        form_widget = QWidget()
        form_layout = QVBoxLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(12)

        # Basic Information Section
        basic_group = QGroupBox("Basic Information")
        basic_layout = QVBoxLayout()
        basic_layout.setSpacing(8)

        # LoRA Name
        name_layout = QVBoxLayout()
        name_layout.setSpacing(3)
        name_label = QLabel("LoRA Name:")
        name_label.setStyleSheet("font-weight: bold;")
        name_layout.addWidget(name_label)
        self.lora_name_edit = QLineEdit()
        self.lora_name_edit.setPlaceholderText("Unique identifier for this LoRA")
        self.lora_name_edit.setStyleSheet(self._get_input_style())
        name_layout.addWidget(self.lora_name_edit)
        basic_layout.addLayout(name_layout)

        # Display Name
        display_layout = QVBoxLayout()
        display_layout.setSpacing(3)
        display_label = QLabel("Display Name:")
        display_label.setStyleSheet("font-weight: bold;")
        display_layout.addWidget(display_label)
        self.lora_display_name_edit = QLineEdit()
        self.lora_display_name_edit.setPlaceholderText("User-friendly name for this LoRA")
        self.lora_display_name_edit.setStyleSheet(self._get_input_style())
        display_layout.addWidget(self.lora_display_name_edit)
        basic_layout.addLayout(display_layout)

        # Base Model Type
        base_layout = QVBoxLayout()
        base_layout.setSpacing(3)
        base_label = QLabel("Base Model Type:")
        base_label.setStyleSheet("font-weight: bold;")
        base_layout.addWidget(base_label)
        self.lora_base_model_combo = QComboBox()
        self.lora_base_model_combo.addItems([t.value for t in ModelType])
        self.lora_base_model_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 2px solid #e0e0e0;
                border-radius: 4px;
                background-color: #ffffff;
            }
            QComboBox:focus {
                border-color: #1976D2;
            }
        """)
        base_layout.addWidget(self.lora_base_model_combo)
        basic_layout.addLayout(base_layout)

        # Description
        desc_layout = QVBoxLayout()
        desc_layout.setSpacing(3)
        desc_label = QLabel("Description:")
        desc_label.setStyleSheet("font-weight: bold;")
        desc_layout.addWidget(desc_label)
        self.lora_description_edit = QTextEdit()
        self.lora_description_edit.setMaximumHeight(80)
        self.lora_description_edit.setPlaceholderText("Enter a description for this LoRA...")
        self.lora_description_edit.setStyleSheet(self._get_textarea_style())
        desc_layout.addWidget(self.lora_description_edit)
        basic_layout.addLayout(desc_layout)

        basic_group.setLayout(basic_layout)
        form_layout.addWidget(basic_group)

        # LoRA-Specific Settings
        lora_group = QGroupBox("LoRA Settings")
        lora_layout = QVBoxLayout()
        lora_layout.setSpacing(8)

        # Default Scaling
        scale_layout = QHBoxLayout()
        scale_layout.setSpacing(10)
        scale_label = QLabel("Default Scaling:")
        scale_label.setStyleSheet("font-weight: bold;")
        scale_layout.addWidget(scale_label)

        self.lora_scaling_spin = QDoubleSpinBox()
        self.lora_scaling_spin.setRange(0.0, 5.0)
        self.lora_scaling_spin.setValue(1.0)
        self.lora_scaling_spin.setSingleStep(0.1)
        self.lora_scaling_spin.setFixedWidth(80)
        scale_layout.addWidget(self.lora_scaling_spin)
        scale_layout.addStretch()
        lora_layout.addLayout(scale_layout)

        # Trigger Words
        trigger_layout = QVBoxLayout()
        trigger_layout.setSpacing(3)
        trigger_label = QLabel("Trigger Words:")
        trigger_label.setStyleSheet("font-weight: bold;")
        trigger_layout.addWidget(trigger_label)
        self.lora_trigger_edit = QTextEdit()
        self.lora_trigger_edit.setMaximumHeight(60)
        self.lora_trigger_edit.setPlaceholderText("Comma-separated trigger words (e.g., 'style1, character1, theme1')")
        self.lora_trigger_edit.setStyleSheet(self._get_textarea_style())
        trigger_layout.addWidget(self.lora_trigger_edit)
        lora_layout.addLayout(trigger_layout)

        lora_group.setLayout(lora_layout)
        form_layout.addWidget(lora_group)

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
        self.lora_usage_edit = QTextEdit()
        self.lora_usage_edit.setMaximumHeight(60)
        self.lora_usage_edit.setPlaceholderText("Any special usage notes or tips...")
        self.lora_usage_edit.setStyleSheet(self._get_textarea_style())
        usage_layout.addWidget(self.lora_usage_edit)
        additional_layout.addLayout(usage_layout)

        # URLs and License
        urls_layout = QHBoxLayout()
        urls_layout.setSpacing(15)

        # Source URL
        url_layout = QVBoxLayout()
        url_layout.setSpacing(3)
        source_label = QLabel("Source URL:")
        source_label.setStyleSheet("font-weight: bold;")
        url_layout.addWidget(source_label)
        self.lora_source_edit = QLineEdit()
        self.lora_source_edit.setPlaceholderText("https://...")
        self.lora_source_edit.setStyleSheet(self._get_input_style())
        url_layout.addWidget(self.lora_source_edit)
        urls_layout.addLayout(url_layout)

        # License Info
        license_layout = QVBoxLayout()
        license_layout.setSpacing(3)
        license_label = QLabel("License:")
        license_label.setStyleSheet("font-weight: bold;")
        license_layout.addWidget(license_label)
        self.lora_license_edit = QLineEdit()
        self.lora_license_edit.setPlaceholderText("License type or attribution...")
        self.lora_license_edit.setStyleSheet(self._get_input_style())
        license_layout.addWidget(self.lora_license_edit)
        urls_layout.addLayout(license_layout)

        additional_layout.addLayout(urls_layout)
        additional_group.setLayout(additional_layout)
        form_layout.addWidget(additional_group)

        form_widget.setLayout(form_layout)
        self.form_sections["LoRA"] = form_widget

    def _create_ip_adapter_form(self):
        """Create the form section for IP-Adapter installation."""
        form_widget = QWidget()
        form_layout = QVBoxLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(12)

        # Basic Information Section
        basic_group = QGroupBox("Basic Information")
        basic_layout = QVBoxLayout()
        basic_layout.setSpacing(8)

        # IP-Adapter Name
        name_layout = QVBoxLayout()
        name_layout.setSpacing(3)
        name_label = QLabel("IP-Adapter Name:")
        name_label.setStyleSheet("font-weight: bold;")
        name_layout.addWidget(name_label)
        self.ip_adapter_name_edit = QLineEdit()
        self.ip_adapter_name_edit.setPlaceholderText("Unique identifier for this IP-Adapter")
        self.ip_adapter_name_edit.setStyleSheet(self._get_input_style())
        name_layout.addWidget(self.ip_adapter_name_edit)
        basic_layout.addLayout(name_layout)

        # Display Name
        display_layout = QVBoxLayout()
        display_layout.setSpacing(3)
        display_label = QLabel("Display Name:")
        display_label.setStyleSheet("font-weight: bold;")
        display_layout.addWidget(display_label)
        self.ip_adapter_display_name_edit = QLineEdit()
        self.ip_adapter_display_name_edit.setPlaceholderText("User-friendly name for this IP-Adapter")
        self.ip_adapter_display_name_edit.setStyleSheet(self._get_input_style())
        display_layout.addWidget(self.ip_adapter_display_name_edit)
        basic_layout.addLayout(display_layout)

        # Adapter Type
        type_layout = QVBoxLayout()
        type_layout.setSpacing(3)
        type_label = QLabel("Adapter Type:")
        type_label.setStyleSheet("font-weight: bold;")
        type_layout.addWidget(type_label)
        self.ip_adapter_type_combo = QComboBox()
        self.ip_adapter_type_combo.addItems(["General", "Face", "Style", "Composition"])
        self.ip_adapter_type_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 2px solid #e0e0e0;
                border-radius: 4px;
                background-color: #ffffff;
            }
            QComboBox:focus {
                border-color: #1976D2;
            }
        """)
        type_layout.addWidget(self.ip_adapter_type_combo)
        basic_layout.addLayout(type_layout)

        # Description
        desc_layout = QVBoxLayout()
        desc_layout.setSpacing(3)
        desc_label = QLabel("Description:")
        desc_label.setStyleSheet("font-weight: bold;")
        desc_layout.addWidget(desc_label)
        self.ip_adapter_description_edit = QTextEdit()
        self.ip_adapter_description_edit.setMaximumHeight(80)
        self.ip_adapter_description_edit.setPlaceholderText("Enter a description for this IP-Adapter...")
        self.ip_adapter_description_edit.setStyleSheet(self._get_textarea_style())
        desc_layout.addWidget(self.ip_adapter_description_edit)
        basic_layout.addLayout(desc_layout)

        basic_group.setLayout(basic_layout)
        form_layout.addWidget(basic_group)

        # IP-Adapter Settings
        ip_group = QGroupBox("IP-Adapter Settings")
        ip_layout = QVBoxLayout()
        ip_layout.setSpacing(8)

        # Default Scale
        scale_layout = QHBoxLayout()
        scale_layout.setSpacing(10)
        scale_label = QLabel("Default Scale:")
        scale_label.setStyleSheet("font-weight: bold;")
        scale_layout.addWidget(scale_label)

        self.ip_adapter_scale_spin = QDoubleSpinBox()
        self.ip_adapter_scale_spin.setRange(0.0, 2.0)
        self.ip_adapter_scale_spin.setValue(1.0)
        self.ip_adapter_scale_spin.setSingleStep(0.1)
        self.ip_adapter_scale_spin.setFixedWidth(80)
        scale_layout.addWidget(self.ip_adapter_scale_spin)
        scale_layout.addStretch()
        ip_layout.addLayout(scale_layout)

        ip_group.setLayout(ip_layout)
        form_layout.addWidget(ip_group)

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
        self.ip_adapter_usage_edit = QTextEdit()
        self.ip_adapter_usage_edit.setMaximumHeight(60)
        self.ip_adapter_usage_edit.setPlaceholderText("Any special usage notes or tips...")
        self.ip_adapter_usage_edit.setStyleSheet(self._get_textarea_style())
        usage_layout.addWidget(self.ip_adapter_usage_edit)
        additional_layout.addLayout(usage_layout)

        # URLs and License
        urls_layout = QHBoxLayout()
        urls_layout.setSpacing(15)

        # Source URL
        url_layout = QVBoxLayout()
        url_layout.setSpacing(3)
        source_label = QLabel("Source URL:")
        source_label.setStyleSheet("font-weight: bold;")
        url_layout.addWidget(source_label)
        self.ip_adapter_source_edit = QLineEdit()
        self.ip_adapter_source_edit.setPlaceholderText("https://...")
        self.ip_adapter_source_edit.setStyleSheet(self._get_input_style())
        url_layout.addWidget(self.ip_adapter_source_edit)
        urls_layout.addLayout(url_layout)

        # License Info
        license_layout = QVBoxLayout()
        license_layout.setSpacing(3)
        license_label = QLabel("License:")
        license_label.setStyleSheet("font-weight: bold;")
        license_layout.addWidget(license_label)
        self.ip_adapter_license_edit = QLineEdit()
        self.ip_adapter_license_edit.setPlaceholderText("License type or attribution...")
        self.ip_adapter_license_edit.setStyleSheet(self._get_input_style())
        license_layout.addWidget(self.ip_adapter_license_edit)
        urls_layout.addLayout(license_layout)

        additional_layout.addLayout(urls_layout)
        additional_group.setLayout(additional_layout)
        form_layout.addWidget(additional_group)

        form_widget.setLayout(form_layout)
        self.form_sections["IP-Adapter"] = form_widget

    def _get_input_style(self):
        """Get consistent input field styling."""
        return """
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
        """

    def _get_textarea_style(self):
        """Get consistent textarea styling."""
        return """
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
        """

    def _get_list_style(self):
        """Get consistent list styling."""
        return """
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
        """

    def _on_type_changed(self, new_type):
        """Handle type selection change."""
        self.current_type = new_type
        self._update_icon()
        self._show_form_for_type(new_type)

    def _update_icon(self):
        """Update the icon based on selected type."""
        icons = {
            "Model": "📦",
            "LoRA": "🎭",
            "IP-Adapter": "🎨"
        }
        self.icon_label.setText(icons.get(self.current_type, "📦"))

    def _show_form_for_type(self, type_name):
        """Show the appropriate form section for the selected type."""
        # Hide all form sections
        for section in self.form_sections.values():
            section.hide()
            if section.parent():
                self.content_layout.removeWidget(section)

        # Show the selected form section
        if type_name in self.form_sections:
            form_section = self.form_sections[type_name]
            self.content_layout.addWidget(form_section)
            form_section.show()

    def _browse_file(self):
        """Browse for a file to install."""
        file_filters = {
            "Model": "Model Files (*.safetensors *.ckpt);;All Files (*)",
            "LoRA": "LoRA Files (*.safetensors *.pt);;All Files (*)",
            "IP-Adapter": "IP-Adapter Files (*.safetensors *.bin);;All Files (*)"
        }

        file_filter = file_filters.get(self.current_type, "All Files (*)")

        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select {self.current_type} File", "", file_filter
        )

        if file_path:
            self.file_path_edit.setText(file_path)
            self._update_file_info(file_path)

    def _update_file_info(self, file_path):
        """Update file information display."""
        import os

        if not file_path or not os.path.exists(file_path):
            self.file_info_label.setText("No file selected")
            return

        try:
            file_size = os.path.getsize(file_path)
            size_mb = file_size / (1024 * 1024)
            self.file_info_label.setText(f"Size: {size_mb:.2f} MB")
        except Exception:
            self.file_info_label.setText("File information unavailable")

    def accept(self):
        """Handle accept button click."""
        # Validate inputs
        if not self._validate_inputs():
            return

        self.accepted = True
        super().accept()

    def reject(self):
        """Handle reject button click."""
        self.accepted = False
        super().reject()

    def _validate_inputs(self):
        """Validate user inputs before accepting."""
        import os

        file_path = self.file_path_edit.text().strip()
        if not file_path:
            QMessageBox.warning(self, "Validation Error", "Please select a file to install.")
            return False

        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Validation Error", "The selected file does not exist.")
            return False

        # Type-specific validation
        if self.current_type == "Model":
            if not self.model_name_edit.text().strip():
                QMessageBox.warning(self, "Validation Error", "Please enter a model name.")
                return False
        elif self.current_type == "LoRA":
            if not self.lora_name_edit.text().strip():
                QMessageBox.warning(self, "Validation Error", "Please enter a LoRA name.")
                return False
        elif self.current_type == "IP-Adapter":
            if not self.ip_adapter_name_edit.text().strip():
                QMessageBox.warning(self, "Validation Error", "Please enter an IP-Adapter name.")
                return False

        return True

    def get_install_data(self):
        """Get the installation data for the selected type."""
        file_path = self.file_path_edit.text().strip()

        if self.current_type == "Model":
            # Get selected categories
            selected_categories = []
            for i in range(self.model_category_list.count()):
                item = self.model_category_list.item(i)
                if item.isSelected():
                    selected_categories.append(item.data(1))  # Get the enum value

            return {
                'type': 'model',
                'file_path': file_path,
                'name': self.model_name_edit.text().strip(),
                'display_name': self.model_display_name_edit.text().strip(),
                'model_type': self.model_type_combo.currentText(),
                'description': self.model_description_edit.toPlainText().strip(),
                'categories': selected_categories,
                'usage_notes': self.model_usage_edit.toPlainText().strip(),
                'source_url': self.model_source_edit.text().strip(),
                'license_info': self.model_license_edit.text().strip()
            }

        elif self.current_type == "LoRA":
            # Parse trigger words
            trigger_text = self.lora_trigger_edit.toPlainText().strip()
            trigger_words = [word.strip() for word in trigger_text.split(',') if word.strip()]

            return {
                'type': 'lora',
                'file_path': file_path,
                'name': self.lora_name_edit.text().strip(),
                'display_name': self.lora_display_name_edit.text().strip(),
                'base_model_type': self.lora_base_model_combo.currentText(),
                'description': self.lora_description_edit.toPlainText().strip(),
                'trigger_words': trigger_words,
                'default_scaling': self.lora_scaling_spin.value(),
                'usage_notes': self.lora_usage_edit.toPlainText().strip(),
                'source_url': self.lora_source_edit.text().strip(),
                'license_info': self.lora_license_edit.text().strip()
            }

        elif self.current_type == "IP-Adapter":
            return {
                'type': 'ip_adapter',
                'file_path': file_path,
                'name': self.ip_adapter_name_edit.text().strip(),
                'display_name': self.ip_adapter_display_name_edit.text().strip(),
                'adapter_type': self.ip_adapter_type_combo.currentText(),
                'description': self.ip_adapter_description_edit.toPlainText().strip(),
                'default_scale': self.ip_adapter_scale_spin.value(),
                'usage_notes': self.ip_adapter_usage_edit.toPlainText().strip(),
                'source_url': self.ip_adapter_source_edit.text().strip(),
                'license_info': self.ip_adapter_license_edit.text().strip()
            }

        return None
