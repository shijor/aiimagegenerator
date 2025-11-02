"""
Dialog for editing existing IP-Adapter metadata.
"""
from PyQt5.QtWidgets import QMessageBox, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit, QListWidget, QListWidgetItem, QComboBox, QDoubleSpinBox, QWidget
from models.model_info import IPAdapterInfo, ModelCategory

class IPAdapterEditDialog(QMessageBox):
    """Dialog for editing existing IP-Adapter metadata."""

    def __init__(self, ip_adapter: IPAdapterInfo, parent=None):
        super().__init__(parent)
        self.ip_adapter = ip_adapter

        self.setWindowTitle("Edit IP-Adapter Parameters")
        # Remove the default text to prevent overlapping - we'll use a custom layout
        self.setText("")
        self.setInformativeText("")

        # Set minimum width for better layout
        self.setMinimumWidth(700)

        # Add custom widgets for IP-Adapter metadata editing
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
        """Set up widgets for editing IP-Adapter metadata."""
        # Create a widget to hold our custom controls
        widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Title section
        title_label = QLabel(f"Edit parameters for IP-Adapter '{self.ip_adapter.name}'")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #E65100; margin-bottom: 5px;")
        main_layout.addWidget(title_label)

        # Main content area - split into two columns
        content_widget = QWidget()
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # Left column - Basic IP-Adapter information
        left_column = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)

        # IP-Adapter Name (unique identifier)
        name_label = QLabel("IP-Adapter Name:")
        name_label.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(name_label)
        self.name_edit = QLineEdit()
        self.name_edit.setText(str(self.ip_adapter.name))
        self.name_edit.setPlaceholderText("Unique name for this IP-Adapter")
        left_layout.addWidget(self.name_edit)

        # Display Name
        display_name_label = QLabel("Display Name:")
        left_layout.addWidget(display_name_label)
        self.display_name_edit = QLineEdit()
        self.display_name_edit.setText(str(self.ip_adapter.display_name or ""))
        self.display_name_edit.setPlaceholderText("User-friendly name")
        left_layout.addWidget(self.display_name_edit)

        # Adapter Type
        adapter_type_label = QLabel("Adapter Type:")
        left_layout.addWidget(adapter_type_label)
        self.adapter_type_combo = QComboBox()
        self.adapter_type_combo.addItem("Style Transfer", "style")
        self.adapter_type_combo.addItem("Composition/Layout", "composition")
        self.adapter_type_combo.addItem("Color Palette", "color")
        self.adapter_type_combo.addItem("Lighting/Effects", "lighting")
        self.adapter_type_combo.addItem("Other", "other")

        # Set current adapter type
        adapter_type_map = {
            "style": 0,
            "composition": 1,
            "color": 2,
            "lighting": 3,
            "other": 4
        }
        current_type_index = adapter_type_map.get(self.ip_adapter.adapter_type, 0)
        self.adapter_type_combo.setCurrentIndex(current_type_index)

        left_layout.addWidget(self.adapter_type_combo)

        # Description
        desc_label = QLabel("Description:")
        left_layout.addWidget(desc_label)
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        self.description_edit.setPlainText(str(self.ip_adapter.description or ""))
        self.description_edit.setPlaceholderText("IP-Adapter description...")
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
        current_categories = set(self.ip_adapter.categories) if self.ip_adapter.categories else set()
        for category in ModelCategory:
            item = QListWidgetItem(category.value.title())
            item.setData(1, category.value)  # Store the enum value
            if category in current_categories:
                item.setSelected(True)
            self.category_list.addItem(item)

        right_layout.addWidget(self.category_list)

        # Default Scale
        scale_label = QLabel("Default Scale:")
        right_layout.addWidget(scale_label)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.0, 2.0)
        self.scale_spin.setValue(float(self.ip_adapter.default_scale))
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setToolTip("Default conditioning scale for this IP-Adapter (0.0-2.0)")
        right_layout.addWidget(self.scale_spin)

        # Recommended Use Cases
        use_cases_label = QLabel("Recommended Use Cases:")
        right_layout.addWidget(use_cases_label)
        self.use_cases_edit = QTextEdit()
        self.use_cases_edit.setMaximumHeight(60)
        # Try to get recommended_use_cases from the IPAdapterInfo object
        use_cases_text = getattr(self.ip_adapter, 'recommended_use_cases', '') or ''
        self.use_cases_edit.setPlainText(use_cases_text)
        self.use_cases_edit.setPlaceholderText("When to use this adapter...")
        right_layout.addWidget(self.use_cases_edit)

        # Usage Notes
        usage_label = QLabel("Usage Notes:")
        right_layout.addWidget(usage_label)
        self.usage_edit = QTextEdit()
        self.usage_edit.setMaximumHeight(60)
        self.usage_edit.setPlainText(str(self.ip_adapter.usage_notes or ""))
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
        self.source_edit.setText(str(self.ip_adapter.source_url or ""))
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
        self.license_edit.setText(str(self.ip_adapter.license_info or ""))
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
        """Get the edited IP-Adapter metadata."""
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
            'name': self.name_edit.text().strip(),
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
