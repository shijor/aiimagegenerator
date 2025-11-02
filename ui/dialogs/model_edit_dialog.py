"""
Dialog for editing existing model metadata.
"""
from PyQt5.QtWidgets import QMessageBox, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit, QListWidget, QListWidgetItem, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox, QSizePolicy
from PyQt5.QtCore import Qt
from models.model_info import ModelInfo, ModelType, ModelCategory

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