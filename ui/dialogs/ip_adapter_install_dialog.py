"""
Dialog for collecting IP-Adapter metadata during installation.
"""
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QGroupBox, QLineEdit, QTextEdit, QListWidget, QListWidgetItem, QComboBox, QDoubleSpinBox, QWidget
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from models.model_info import ModelCategory

class IPAdapterInstallDialog(QDialog):
    """Dialog for collecting IP-Adapter metadata during installation."""

    def __init__(self, adapter_name: str, adapter_path: str, parent=None):
        super().__init__(parent)
        self.adapter_name = adapter_name
        self.adapter_path = adapter_path

        self.setWindowTitle("Install IP-Adapter")
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
        icon_label = QLabel("🎨")
        icon_font = QFont()
        icon_font.setPointSize(32)
        icon_label.setFont(icon_font)
        header_layout.addWidget(icon_label)

        # Title and subtitle
        title_widget = QWidget()
        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(5)

        title_label = QLabel("Install IP-Adapter")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #FF6B35;")
        title_layout.addWidget(title_label)

        adapter_label = QLabel(f"🎨 {self.adapter_name}")
        adapter_font = QFont()
        adapter_font.setPointSize(12)
        adapter_label.setFont(adapter_font)
        adapter_label.setStyleSheet("color: #666;")
        title_layout.addWidget(adapter_label)

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

        path_display = QLabel(self.adapter_path)
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
        self.display_name_edit.setText(self.adapter_name)
        self.display_name_edit.setPlaceholderText("User-friendly name for this IP-Adapter")
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

        # Adapter Type
        adapter_type_layout = QVBoxLayout()
        adapter_type_layout.setSpacing(3)
        adapter_type_label = QLabel("Adapter Type:")
        adapter_type_label.setStyleSheet("font-weight: bold;")
        adapter_type_layout.addWidget(adapter_type_label)
        self.adapter_type_combo = QComboBox()
        self.adapter_type_combo.addItem("Style Transfer", "style")
        self.adapter_type_combo.addItem("Composition/Layout", "composition")
        self.adapter_type_combo.addItem("Color Palette", "color")
        self.adapter_type_combo.addItem("Lighting/Effects", "lighting")
        self.adapter_type_combo.addItem("Other", "other")
        self.adapter_type_combo.setCurrentIndex(0)  # Default to style
        self.adapter_type_combo.setStyleSheet("""
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
        adapter_type_layout.addWidget(self.adapter_type_combo)
        basic_layout.addLayout(adapter_type_layout)

        # Description
        desc_layout = QVBoxLayout()
        desc_layout.setSpacing(3)
        desc_label = QLabel("Description:")
        desc_label.setStyleSheet("font-weight: bold;")
        desc_layout.addWidget(desc_label)
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        self.description_edit.setPlaceholderText("Describe what this IP-Adapter does (e.g., 'Transfers artistic painting style', 'Creates cinematic lighting')...")
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

        # IP-Adapter Configuration Section
        adapter_group = QGroupBox("IP-Adapter Configuration")
        adapter_layout = QVBoxLayout()
        adapter_layout.setSpacing(8)

        # Default Scale
        scale_layout = QVBoxLayout()
        scale_layout.setSpacing(3)
        scale_label = QLabel("Default Scale:")
        scale_label.setStyleSheet("font-weight: bold;")
        scale_layout.addWidget(scale_label)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.0, 2.0)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setToolTip("Default conditioning scale for this IP-Adapter (0.0-2.0)")
        self.scale_spin.setStyleSheet("""
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
        scale_layout.addWidget(self.scale_spin)
        adapter_layout.addLayout(scale_layout)

        # Recommended Use Cases
        use_cases_layout = QVBoxLayout()
        use_cases_layout.setSpacing(3)
        use_cases_label = QLabel("Recommended Use Cases:")
        use_cases_label.setStyleSheet("font-weight: bold;")
        use_cases_layout.addWidget(use_cases_label)
        self.use_cases_edit = QTextEdit()
        self.use_cases_edit.setMaximumHeight(60)
        self.use_cases_edit.setPlaceholderText("Describe when to use this adapter (e.g., 'Best for portraits', 'Works well with landscapes')...")
        self.use_cases_edit.setStyleSheet("""
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
        use_cases_layout.addWidget(self.use_cases_edit)
        adapter_layout.addLayout(use_cases_layout)

        adapter_group.setLayout(adapter_layout)
        form_layout.addWidget(adapter_group)

        # Categories Section
        categories_group = QGroupBox("Categories")
        categories_layout = QVBoxLayout()
        categories_layout.setSpacing(8)

        categories_desc = QLabel("Select categories that best describe this IP-Adapter:")
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
        self.usage_edit.setPlaceholderText("Any special usage notes, tips, or compatibility information...")
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
        self.install_btn = QPushButton("✅ Install IP-Adapter")
        self.install_btn.clicked.connect(self.accept)
        self.install_btn.setMinimumHeight(40)
        self.install_btn.setMinimumWidth(160)
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

    def get_metadata(self) -> dict:
        """Get the collected metadata."""
        # Get selected categories
        selected_categories = []
        for i in range(self.category_list.count()):
            item = self.category_list.item(i)
            if item.isSelected():
                selected_categories.append(item.data(1))  # Get the enum value

        # Get selected adapter type
        current_index = self.adapter_type_combo.currentIndex()
        adapter_type = self.adapter_type_combo.itemData(current_index)

        return {
            'display_name': self.display_name_edit.text().strip(),
            'adapter_type': adapter_type,
            'description': self.description_edit.toPlainText().strip(),
            'categories': selected_categories,
            'default_scale': self.scale_spin.value(),
            'recommended_use_cases': self.use_cases_edit.toPlainText().strip(),
            'usage_notes': self.usage_edit.toPlainText().strip(),
            'source_url': self.source_edit.text().strip(),
            'license_info': self.license_edit.text().strip()
        }
