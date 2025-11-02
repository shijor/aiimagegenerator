"""
Dialog for editing existing LoRA adapter metadata.
"""
from PyQt5.QtWidgets import QMessageBox, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit, QListWidget, QListWidgetItem, QComboBox, QDoubleSpinBox, QWidget
from models.model_info import LoRAInfo, ModelType, ModelCategory

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