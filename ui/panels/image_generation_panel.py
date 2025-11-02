"""
Image generation panel with sidebar and main area.
"""
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox, QFormLayout, QGroupBox, QProgressBar, QTextEdit, QComboBox, QCheckBox, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.model_manager import ModelManager
from services.image_generator import ImageGenerationService, ImageGenerator, ModelLoader
from services.text_enhancer import TextEnhancerService
from models.generation_params import GenerationParams
from models.model_info import ModelType
from ui.dialogs.model_loading_dialog import ModelLoadingDialog
from ui.dialogs.enhance_prompt_dialog import EnhancePromptDialog


class ImageGenerationPanel:
    """Panel for image generation functionality."""

    def __init__(self, model_manager: ModelManager, image_service: ImageGenerationService, text_enhancer: TextEnhancerService = None):
        self.model_manager = model_manager
        self.image_service = image_service
        self.text_enhancer = text_enhancer or TextEnhancerService()
        self.current_generator = None  # Track current image generation thread
        self.pending_prompt = None  # Store prompt for delayed generation

        # Create sidebar and main area
        self.sidebar = self._create_sidebar()
        self.main_area = self._create_main_area()

    def _create_sidebar(self) -> QWidget:
        """Create the sidebar widget."""
        widget = QWidget()

        layout = QVBoxLayout()

        # Prompts Section
        prompts_group = QGroupBox("Prompts")
        prompts_layout = QVBoxLayout()

        # Prompt
        prompt_label = QLabel("Prompt:")
        prompts_layout.addWidget(prompt_label)

        # Prompt input with enhance button
        prompt_input_layout = QVBoxLayout()
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText('Enter your prompt here...')
        self.prompt_input.setMaximumHeight(100)
        prompt_input_layout.addWidget(self.prompt_input)

        # Enhance button under prompt
        self.enhance_btn = QPushButton('✨ Enhance Prompt')
        self.enhance_btn.clicked.connect(self.enhance_prompt)
        self.enhance_btn.setToolTip('Use AI to enhance your prompt for better image generation')
        self.enhance_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        prompt_input_layout.addWidget(self.enhance_btn)

        prompts_layout.addLayout(prompt_input_layout)

        # Negative Prompt
        negative_label = QLabel("Negative Prompt:")
        prompts_layout.addWidget(negative_label)
        self.negative_prompt_input = QTextEdit()
        self.negative_prompt_input.setPlaceholderText('What to avoid in the image (optional)...')
        self.negative_prompt_input.setMaximumHeight(60)
        prompts_layout.addWidget(self.negative_prompt_input)

        prompts_group.setLayout(prompts_layout)
        layout.addWidget(prompts_group)

        # LoRA Adapters Section
        lora_group = QGroupBox("LoRA Adapters")
        lora_layout = QVBoxLayout()

        # LoRA list and controls
        self.lora_list_widget = QWidget()
        self.lora_list_layout = QVBoxLayout()
        self.lora_list_widget.setLayout(self.lora_list_layout)

        # Add LoRA button
        add_lora_btn = QPushButton("➕ Add LoRA")
        add_lora_btn.clicked.connect(self._add_lora_adapter)
        lora_layout.addWidget(add_lora_btn)

        # Scroll area for LoRA items
        from PyQt5.QtWidgets import QScrollArea
        lora_scroll = QScrollArea()
        lora_scroll.setWidget(self.lora_list_widget)
        lora_scroll.setWidgetResizable(True)
        lora_scroll.setMaximumHeight(200)
        lora_layout.addWidget(lora_scroll)

        lora_group.setLayout(lora_layout)
        layout.addWidget(lora_group)

        # IP-Adapter Section
        ip_adapter_group = QGroupBox("IP-Adapter")
        ip_adapter_layout = QVBoxLayout()

        # IP-Adapter controls
        self.ip_adapter_list_widget = QWidget()
        self.ip_adapter_list_layout = QVBoxLayout()
        self.ip_adapter_list_widget.setLayout(self.ip_adapter_list_layout)

        # Add IP-Adapter button
        add_ip_adapter_btn = QPushButton("🎨 Add IP-Adapter")
        add_ip_adapter_btn.clicked.connect(self._add_ip_adapter)
        ip_adapter_layout.addWidget(add_ip_adapter_btn)

        # IP-Adapter mode selection
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Mode:")
        self.ip_adapter_mode_select = QComboBox()
        self.ip_adapter_mode_select.addItems(["Style", "Composition", "Style and Composition", "Prompt Priority"])
        self.ip_adapter_mode_select.setCurrentText("Style")  # Default to Style
        self.ip_adapter_mode_select.setToolTip("Select how the IP-Adapter influences the generation:\n• Style: Strong artistic influence\n• Composition: Layout influence\n• Style and Composition: Balanced\n• Prompt Priority: Weak influence, prompt dominates")
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.ip_adapter_mode_select)
        mode_layout.addStretch()
        ip_adapter_layout.addLayout(mode_layout)

        # Reference image upload section
        ref_image_layout = QHBoxLayout()
        self.ref_image_path = None
        self.ref_image_label = QLabel("No reference image")
        self.ref_image_label.setStyleSheet("border: 1px solid #ccc; padding: 5px; background-color: #f9f9f9;")
        self.ref_image_label.setMinimumHeight(60)
        ref_image_layout.addWidget(self.ref_image_label)

        upload_ref_btn = QPushButton("📷 Upload Reference")
        upload_ref_btn.clicked.connect(self._upload_reference_image)
        ref_image_layout.addWidget(upload_ref_btn)

        clear_ref_btn = QPushButton("❌ Clear")
        clear_ref_btn.clicked.connect(self._clear_reference_image)
        ref_image_layout.addWidget(clear_ref_btn)

        ip_adapter_layout.addLayout(ref_image_layout)

        # Scroll area for IP-Adapter items
        ip_adapter_scroll = QScrollArea()
        ip_adapter_scroll.setWidget(self.ip_adapter_list_widget)
        ip_adapter_scroll.setWidgetResizable(True)
        ip_adapter_scroll.setMaximumHeight(150)
        ip_adapter_layout.addWidget(ip_adapter_scroll)

        ip_adapter_group.setLayout(ip_adapter_layout)
        layout.addWidget(ip_adapter_group)

        # Parameters Group
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout()

        # Model Selection with refresh button
        model_layout = QHBoxLayout()
        self.model_select = QComboBox()
        self._refresh_model_dropdown()
        self.model_select.currentTextChanged.connect(self._on_model_changed)
        # Connect to model selection changes to update defaults
        self.model_select.currentTextChanged.connect(self._update_generation_defaults)
        model_layout.addWidget(self.model_select)

        # Refresh button
        self.refresh_models_btn = QPushButton('🔄')
        self.refresh_models_btn.setMaximumWidth(30)
        self.refresh_models_btn.setToolTip('Refresh model list')
        self.refresh_models_btn.clicked.connect(self._refresh_model_dropdown)
        model_layout.addWidget(self.refresh_models_btn)

        params_layout.addRow("Model:", model_layout)

        self.steps_input = QSpinBox()
        self.steps_input.setRange(1, 100)
        self.steps_input.setValue(20)
        params_layout.addRow("Steps:", self.steps_input)

        self.guidance_input = QDoubleSpinBox()
        self.guidance_input.setRange(1.0, 20.0)
        self.guidance_input.setValue(7.5)
        self.guidance_input.setSingleStep(0.1)
        params_layout.addRow("Guidance Scale:", self.guidance_input)

        # Seed input with random toggle
        seed_layout = QHBoxLayout()
        self.seed_input = QLineEdit()
        self.seed_input.setPlaceholderText("0")
        self.seed_input.setText("0")
        self.seed_input.setMinimumWidth(100)
        self.seed_input.setMaximumWidth(120)
        # Add input validation for numbers only
        from PyQt5.QtGui import QIntValidator
        self.seed_input.setValidator(QIntValidator(0, 999999))
        seed_layout.addWidget(self.seed_input)

        self.random_seed_checkbox = QCheckBox("Random")
        self.random_seed_checkbox.setChecked(True)  # Default to random
        self.random_seed_checkbox.stateChanged.connect(self._on_random_seed_toggled)
        seed_layout.addWidget(self.random_seed_checkbox)

        # Initialize seed input state
        self._on_random_seed_toggled()

        params_layout.addRow("Seed:", seed_layout)

        # Aspect Ratio
        self.aspect_ratio_select = QComboBox()
        self.aspect_ratio_select.addItems(["1:1", "9:16", "16:9", "Custom"])
        self.aspect_ratio_select.currentTextChanged.connect(self._on_aspect_ratio_changed)
        params_layout.addRow("Aspect Ratio:", self.aspect_ratio_select)

        # Width and Height
        self.width_input = QSpinBox()
        self.width_input.setRange(64, 2048)
        self.width_input.setValue(512)
        self.width_input.setSingleStep(64)
        params_layout.addRow("Width:", self.width_input)

        self.height_input = QSpinBox()
        self.height_input.setRange(64, 2048)
        self.height_input.setValue(512)
        self.height_input.setSingleStep(64)
        params_layout.addRow("Height:", self.height_input)

        # Advanced Options Group
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QFormLayout()

        # Scheduler Selection
        self.scheduler_select = QComboBox()
        self.scheduler_select.addItems([
            "DPM++ 2M Karras", "DDIM", "DPM++ 2M", "Euler A",
            "DPM++ SDE Karras", "Euler", "DPM++ SDE", "UniPC",
            "Heun", "KDPM2 A", "KDPM2", "PNDM"
        ])
        self.scheduler_select.setCurrentText("DPM++ 2M Karras")
        advanced_layout.addRow("Scheduler:", self.scheduler_select)

        # Quantization
        self.quantization_select = QComboBox()
        self.quantization_select.addItems(["None", "8-bit", "4-bit"])
        self.quantization_select.setCurrentText("None")
        advanced_layout.addRow("Quantization:", self.quantization_select)

        # Optimization Checkboxes
        self.xformers_checkbox = QCheckBox("Use xFormers (faster)")
        self.xformers_checkbox.setChecked(True)
        advanced_layout.addRow(self.xformers_checkbox)

        self.vae_tiling_checkbox = QCheckBox("VAE Tiling (large images)")
        self.vae_tiling_checkbox.setChecked(True)
        advanced_layout.addRow(self.vae_tiling_checkbox)

        self.cpu_offload_checkbox = QCheckBox("CPU Offload (saves VRAM)")
        self.cpu_offload_checkbox.setChecked(False)
        self.cpu_offload_checkbox.setToolTip("Move model components to CPU when not needed - slower but uses less VRAM")
        advanced_layout.addRow(self.cpu_offload_checkbox)

        self.nsfw_filter_checkbox = QCheckBox("Enable NSFW Filter")
        self.nsfw_filter_checkbox.setChecked(True)
        self.nsfw_filter_checkbox.setToolTip("Filter out not-safe-for-work content from generated images")
        advanced_layout.addRow(self.nsfw_filter_checkbox)

        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Generate and Cancel Buttons
        buttons_layout = QHBoxLayout()

        self.generate_btn = QPushButton('🎨 Generate Image')
        self.generate_btn.clicked.connect(self.generate_image)
        buttons_layout.addWidget(self.generate_btn)

        self.cancel_btn = QPushButton('❌ Cancel')
        self.cancel_btn.clicked.connect(self.cancel_generation)
        self.cancel_btn.setEnabled(False)  # Initially disabled
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        buttons_layout.addWidget(self.cancel_btn)

        layout.addLayout(buttons_layout)

        layout.addStretch()

        widget.setLayout(layout)
        return widget

    def _refresh_model_dropdown(self):
        """Refresh the model dropdown with current installed models using display names."""
        current_selection = self.model_select.currentText()
        self.model_select.clear()

        # Get all installed models - they should all be valid for image generation
        # since they were validated during installation
        installed_models = self.model_manager.get_installed_models()

        # Create mapping between display names and model objects
        self.model_display_mapping = {}
        display_names = []

        for model in installed_models:
            # Use display name if available, otherwise fall back to model name
            display_name = model.display_name or model.name
            display_names.append(display_name)
            self.model_display_mapping[display_name] = model

        self.model_select.addItems(display_names)

        # Try to restore previous selection by finding the corresponding display name
        if current_selection:
            # Check if current_selection is already a display name
            if current_selection in self.model_display_mapping:
                self.model_select.setCurrentText(current_selection)
            else:
                # Try to find the model that was previously selected and get its display name
                for model in installed_models:
                    if model.name == current_selection:
                        display_name = model.display_name or model.name
                        self.model_select.setCurrentText(display_name)
                        break

        # If no selection, default to the first model or "Stable Diffusion v1.4" if available
        if self.model_select.currentText() == "":
            default_model_name = "Stable Diffusion v1.4"
            for model in installed_models:
                if model.name == default_model_name:
                    display_name = model.display_name or model.name
                    self.model_select.setCurrentText(display_name)
                    break

            # If default not found, select the first available model
            if self.model_select.currentText() == "" and display_names:
                self.model_select.setCurrentIndex(0)

    def _create_main_area(self) -> QWidget:
        """Create the main area widget."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Image Display - Make it responsive and fill available space
        self.image_label = QLabel("No image generated yet")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(200, 200)  # Smaller minimum size
        self.image_label.setSizePolicy(QWidget().sizePolicy().Expanding, QWidget().sizePolicy().Expanding)
        self.image_label.setStyleSheet("""
            border: 2px dashed #aaa;
            background-color: #f0f0f0;
            QLabel {
                qproperty-alignment: AlignCenter;
            }
        """)
        layout.addWidget(self.image_label, stretch=1)  # Give it stretch priority

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(25)
        # Make progress bar width match the image canvas
        from PyQt5.QtWidgets import QSizePolicy
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.progress_bar)

        # Save and Reset Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)

        self.save_btn = QPushButton('💾 Save Image')
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)
        self.save_btn.setMinimumHeight(35)
        buttons_layout.addWidget(self.save_btn)

        self.reset_btn = QPushButton('🔄 Reset')
        self.reset_btn.clicked.connect(self.reset_image)
        self.reset_btn.setEnabled(False)  # Initially disabled until image is generated
        self.reset_btn.setMinimumHeight(35)
        self.reset_btn.setToolTip('Clear the generated image from the canvas')
        buttons_layout.addWidget(self.reset_btn)

        layout.addLayout(buttons_layout)

        # Reduce margins and spacing for better space utilization
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        widget.setLayout(layout)
        return widget

    def generate_image(self):
        """Generate image with current parameters using lazy loading."""
        print("🔍 DEBUG: Generate image button clicked")

        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            print("🔍 DEBUG: No prompt provided - aborting generation")
            return

        print(f"🔍 DEBUG: Prompt length: {len(prompt)} characters")
        print(f"🔍 DEBUG: Prompt preview: '{prompt[:100]}{'...' if len(prompt) > 100 else ''}'")

        # Store prompt for later use
        self.pending_prompt = prompt

        # Get selected model using display name mapping
        selected_display_name = self.model_select.currentText()
        if not selected_display_name:
            return

        # Get the model object from the display name mapping
        selected_model = self.model_display_mapping.get(selected_display_name)
        if not selected_model:
            self.on_generation_error("Selected model not found!")
            return

        # Validate model file exists before loading
        if not self._validate_model_file(selected_model):
            return

        # Check if a model is already loaded
        current_model_path = self.image_service._current_model_path
        if current_model_path == selected_model.path and self.image_service.model is not None:
            # Model is already loaded, proceed with generation
            self._start_image_generation()
        else:
            # Need to load model first (lazy loading)
            self._load_model_for_generation(selected_model)

    def _load_model_for_generation(self, selected_model):
        """Load model with loading dialog for lazy loading."""
        # Unload current model if different
        if self.image_service._current_model_path and self.image_service._current_model_path != selected_model.path:
            self.image_service.unload_model()

        # Create and show loading dialog
        self.loading_dialog = ModelLoadingDialog(
            selected_model.display_name or selected_model.name,
            selected_model.path,
            selected_model.size_mb,
            self.generate_btn.parentWidget().window()
        )

        # Connect dialog signals
        self.loading_dialog.cancelled.connect(self._on_model_loading_cancelled)

        # Get optimization parameters for model loading
        quantization = self.quantization_select.currentText()
        use_xformers = self.xformers_checkbox.isChecked()
        cpu_offload = self.cpu_offload_checkbox.isChecked()

        # Create model loader thread
        self.current_model_loader = self.image_service.load_model_async(
            selected_model.path,
            quantization=quantization,
            use_xformers=use_xformers,
            cpu_offload=cpu_offload
        )

        # Connect loader signals
        self.current_model_loader.progress.connect(self.loading_dialog.update_progress)
        self.current_model_loader.finished.connect(lambda model: self._on_model_loaded_for_generation(model, selected_model.path))
        self.current_model_loader.error.connect(self._on_model_loading_error_for_generation)

        # Show dialog and start loading
        self.loading_dialog.show()
        self.current_model_loader.start()

    def _on_model_loaded_for_generation(self, model, model_path: str):
        """Handle successful model loading for generation."""
        # Close loading dialog
        if hasattr(self, 'loading_dialog') and self.loading_dialog:
            self.loading_dialog.accept()
            self.loading_dialog = None

        # Set the loaded model
        self.image_service.model = model
        self.image_service._current_model_path = model_path

        # Now start image generation
        self._start_image_generation()

    def _on_model_loading_error_for_generation(self, error_msg: str):
        """Handle model loading error during generation."""
        # Close loading dialog
        if hasattr(self, 'loading_dialog') and self.loading_dialog:
            self.loading_dialog.reject()
            self.loading_dialog = None

        # Show error
        self.on_generation_error(f"Failed to load model: {error_msg}")

    def _on_model_loading_cancelled(self):
        """Handle model loading cancellation."""
        # Stop the model loader thread
        if hasattr(self, 'current_model_loader') and self.current_model_loader and self.current_model_loader.isRunning():
            # Note: ModelLoader doesn't have a built-in cancel mechanism
            # We'll just close the dialog and show cancellation message
            pass

        # Close dialog
        if hasattr(self, 'loading_dialog') and self.loading_dialog:
            self.loading_dialog.reject()
            self.loading_dialog = None

        # Show cancellation message
        self.image_label.setText("Model loading cancelled")

    def _start_image_generation(self):
        """Start image generation with loaded model."""
        print("🔍 DEBUG: Starting image generation setup")

        # Get parameters
        steps = self.steps_input.value()
        guidance = self.guidance_input.value()

        # Handle seed based on random checkbox
        if self.random_seed_checkbox.isChecked():
            import random
            seed = random.randint(0, 999999)
            print(f"🔍 DEBUG: Generated random seed: {seed}")
        else:
            seed = int(self.seed_input.text() or "0")
            print(f"🔍 DEBUG: Using manual seed: {seed}")

        # Store the seed that will be used for generation
        self.current_seed = seed

        negative = self.negative_prompt_input.toPlainText().strip()
        width = self.width_input.value()
        height = self.height_input.value()
        scheduler = self.scheduler_select.currentText()
        quantization = self.quantization_select.currentText()
        use_xformers = self.xformers_checkbox.isChecked()
        vae_tiling = self.vae_tiling_checkbox.isChecked()
        enable_nsfw_filter = self.nsfw_filter_checkbox.isChecked()

        print(f"🔍 DEBUG: Generation parameters:")
        print(f"   - Steps: {steps}")
        print(f"   - Guidance Scale: {guidance}")
        print(f"   - Seed: {seed}")
        print(f"   - Dimensions: {width}x{height}")
        print(f"   - Scheduler: {scheduler}")
        print(f"   - Quantization: {quantization}")
        print(f"   - xFormers: {use_xformers}")
        print(f"   - VAE Tiling: {vae_tiling}")
        print(f"   - NSFW Filter: {enable_nsfw_filter}")
        print(f"   - Negative prompt: '{negative[:50]}{'...' if len(negative) > 50 else ''}'")

        # Get selected LoRA adapters
        lora_adapters = self._get_selected_loras()
        print(f"🔍 DEBUG: Selected LoRA adapters: {len(lora_adapters)}")
        for lora in lora_adapters:
            print(f"   - {lora['name']} (scale: {lora['scaling']})")

        # Get selected IP-Adapters
        ip_adapters = self._get_selected_ip_adapters()
        print(f"🔍 DEBUG: Selected IP-Adapters: {len(ip_adapters)}")
        for ip_adapter in ip_adapters:
            print(f"   - {ip_adapter['name']} (scale: {ip_adapter['scale']})")

        # Get reference image path
        reference_image_path = self.ref_image_path
        if reference_image_path:
            print(f"🔍 DEBUG: Reference image: {reference_image_path}")

        # Get IP-Adapter mode
        ip_adapter_mode = self.ip_adapter_mode_select.currentText().lower().replace(" ", "_")
        print(f"🔍 DEBUG: IP-Adapter mode: {ip_adapter_mode}")

        params = GenerationParams(
            steps=steps,
            guidance_scale=guidance,
            seed=seed,
            negative_prompt=negative,
            width=width,
            height=height,
            scheduler=scheduler,
            quantization=quantization,
            use_xformers=use_xformers,
            vae_tiling=vae_tiling,
            enable_nsfw_filter=enable_nsfw_filter,
            lora_adapters=lora_adapters,
            ip_adapters=ip_adapters,
            reference_image_path=reference_image_path,
            ip_adapter_mode=ip_adapter_mode
        )

        # Disable UI and enable cancel
        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.image_label.setText("Generating image...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate

        # Generate image
        self.current_generator = self.image_service.generate_image(self.pending_prompt, params)
        self.current_generator.finished.connect(self.on_generation_finished)
        self.current_generator.error.connect(self.on_generation_error)
        self.current_generator.start()

    def on_generation_finished(self, image):
        """Handle successful image generation."""
        self.current_image = image

        # Convert PIL to QPixmap
        img_array = np.array(image)
        height, width, channel = img_array.shape
        bytes_per_line = 3 * width
        q_img = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Scale image to fit available space while maintaining aspect ratio
        self._display_scaled_image(pixmap)

        # Display the actual seed used in the seed input field
        if hasattr(self, 'current_seed'):
            self.seed_input.setText(str(self.current_seed))

        # Re-enable UI and disable cancel
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)

    def cancel_generation(self):
        """Cancel the current image generation."""
        if self.current_generator and self.current_generator.isRunning():
            print("Cancelling image generation...")
            self.current_generator.requestInterruption()
            self.current_generator.wait(5000)  # Wait up to 5 seconds for clean shutdown

            # Reset UI state
            self.image_label.setText("Generation cancelled")
            self.progress_bar.setVisible(False)
            self.generate_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            self.save_btn.setEnabled(False)

            # Clear the current generator reference
            self.current_generator = None

    def on_generation_error(self, error_msg: str):
        """Handle image generation error."""
        self.image_label.setText(f"Error: {error_msg}")
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.save_btn.setEnabled(False)



    def on_model_loading_progress(self, message: str, percentage: int):
        """Handle model loading progress updates."""
        self.image_label.setText(message)
        self.progress_bar.setValue(percentage)

    def on_model_loaded(self, model, model_path: str):
        """Handle successful model loading."""
        self.image_service.model = model
        self.image_service._current_model_path = model_path
        self.progress_bar.setVisible(False)

        # Now proceed with image generation
        self._generate_image_after_model_loaded()

    def on_model_loading_error(self, error_msg: str):
        """Handle model loading error."""
        self.image_label.setText(f"Error: {error_msg}")
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

    def _generate_image_after_model_loaded(self):
        """Generate image after model has been loaded."""
        # Get parameters
        steps = self.steps_input.value()
        guidance = self.guidance_input.value()
        seed = int(self.seed_input.text() or "0")
        negative = self.negative_prompt_input.toPlainText().strip()
        width = self.width_input.value()
        height = self.height_input.value()
        scheduler = self.scheduler_select.currentText()
        quantization = self.quantization_select.currentText()
        use_xformers = self.xformers_checkbox.isChecked()
        vae_tiling = self.vae_tiling_checkbox.isChecked()

        params = GenerationParams(
            steps=steps,
            guidance_scale=guidance,
            seed=seed,
            negative_prompt=negative,
            width=width,
            height=height,
            scheduler=scheduler,
            quantization=quantization,
            use_xformers=use_xformers,
            vae_tiling=vae_tiling
        )

        # Disable UI and enable cancel
        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.image_label.setText("Generating image...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate

        # Generate image
        self.current_generator = self.image_service.generate_image(self.pending_prompt, params)
        self.current_generator.finished.connect(self.on_generation_finished)
        self.current_generator.error.connect(self.on_generation_error)
        self.current_generator.start()

    def _display_scaled_image(self, pixmap: QPixmap):
        """Scale and display image to fit available space."""
        if pixmap.isNull():
            return

        # Get the available space in the image label
        available_size = self.image_label.size()

        # Calculate the scaled size while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            available_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # Set the scaled pixmap
        self.image_label.setPixmap(scaled_pixmap)

    def _parse_aspect_ratio(self, aspect_ratio_str: str) -> tuple[int, int]:
        """Parse aspect ratio string like '512x512' into width, height tuple."""
        if not aspect_ratio_str or 'x' not in aspect_ratio_str:
            return 512, 512  # Default fallback

        try:
            width_str, height_str = aspect_ratio_str.split('x', 1)
            width = int(width_str.strip())
            height = int(height_str.strip())
            return width, height
        except (ValueError, AttributeError):
            return 512, 512  # Default fallback

    def _on_aspect_ratio_changed(self, aspect_ratio: str):
        """Handle aspect ratio selection change."""
        # Get the currently selected model
        selected_display_name = self.model_select.currentText()
        if not selected_display_name or selected_display_name not in self.model_display_mapping:
            # Fallback to default values if no model selected
            if aspect_ratio == "1:1":
                self.width_input.setValue(512)
                self.height_input.setValue(512)
            elif aspect_ratio == "9:16":
                self.width_input.setValue(384)
                self.height_input.setValue(640)
            elif aspect_ratio == "16:9":
                self.width_input.setValue(640)
                self.height_input.setValue(384)
            elif aspect_ratio == "Custom":
                pass  # Don't change current values
            return

        selected_model = self.model_display_mapping[selected_display_name]

        if aspect_ratio == "1:1":
            # Use model's 1:1 aspect ratio setting
            width, height = self._parse_aspect_ratio(selected_model.aspect_ratio_1_1)
            self.width_input.setValue(width)
            self.height_input.setValue(height)
        elif aspect_ratio == "9:16":
            # Use model's 9:16 aspect ratio setting (portrait)
            width, height = self._parse_aspect_ratio(selected_model.aspect_ratio_9_16)
            self.width_input.setValue(width)
            self.height_input.setValue(height)
        elif aspect_ratio == "16:9":
            # Use model's 16:9 aspect ratio setting (landscape)
            width, height = self._parse_aspect_ratio(selected_model.aspect_ratio_16_9)
            self.width_input.setValue(width)
            self.height_input.setValue(height)
        elif aspect_ratio == "Custom":
            # Don't change current values for custom
            pass

    def _on_random_seed_toggled(self):
        """Handle random seed checkbox toggle."""
        is_random = self.random_seed_checkbox.isChecked()
        self.seed_input.setEnabled(not is_random)

        # Keep the currently displayed seed value unchanged
        # Only enable/disable the input field

    def _on_model_changed(self, display_name: str):
        """Handle model selection change."""
        if not display_name or display_name not in self.model_display_mapping:
            return

        # Apply the current aspect ratio setting to the new model
        current_aspect_ratio = self.aspect_ratio_select.currentText()
        if current_aspect_ratio and current_aspect_ratio != "Custom":
            self._on_aspect_ratio_changed(current_aspect_ratio)

    def _update_generation_defaults(self, display_name: str):
        """Update generation parameters to model's defaults when model is selected."""
        if not display_name or display_name not in self.model_display_mapping:
            return

        selected_model = self.model_display_mapping[display_name]

        # Update steps and guidance scale to model's defaults
        self.steps_input.setValue(int(selected_model.default_steps))
        self.guidance_input.setValue(float(selected_model.default_cfg))

    def _validate_model_file(self, model) -> bool:
        """Validate that the model file exists and is accessible."""
        if not os.path.exists(model.path):
            # Model file is missing - alert user and remove from database
            reply = QMessageBox.question(
                None, "Model File Missing",
                f"The model file for '{model.name}' is no longer available at:\n\n{model.path}\n\n"
                "Would you like to remove this model from the database?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes  # Default to Yes since file is missing
            )

            if reply == QMessageBox.Yes:
                # Remove the model from database
                success = self.model_manager.delete_model(model.name)
                if success:
                    QMessageBox.information(None, "Model Removed",
                                          f"Model '{model.name}' has been removed from the database.")
                    # Refresh the model dropdown
                    self._refresh_model_dropdown()
                else:
                    QMessageBox.warning(None, "Removal Failed",
                                      f"Failed to remove model '{model.name}' from database.")

            return False

        # Additional validation for single-file models
        if os.path.isfile(model.path):
            try:
                # Check if file is readable and has reasonable size
                file_size = os.path.getsize(model.path)
                if file_size < 1024 * 1024:  # Less than 1MB
                    QMessageBox.warning(None, "Invalid Model",
                                      f"Model '{model.name}' appears to be invalid (file too small).")
                    return False
            except OSError:
                QMessageBox.warning(None, "Model Access Error",
                                  f"Cannot access model file for '{model.name}'.")
                return False

        return True

    def _add_lora_adapter(self):
        """Add a LoRA adapter selection widget."""
        # Get the currently selected model to filter compatible LoRAs
        selected_display_name = self.model_select.currentText()
        selected_model = None
        if selected_display_name and selected_display_name in self.model_display_mapping:
            selected_model = self.model_display_mapping[selected_display_name]

        # Get available LoRA adapters
        all_loras = self.model_manager.get_installed_loras()

        # Filter LoRAs based on model compatibility
        available_loras = []
        if selected_model:
            # Filter LoRAs compatible with the selected model type
            for lora in all_loras:
                # LoRA is compatible if:
                # 1. It has no base_model_type specified (works with any model), or
                # 2. Its base_model_type matches the selected model's type
                if (lora.base_model_type is None or
                    lora.base_model_type == selected_model.model_type):
                    available_loras.append(lora)
        else:
            # No model selected, show all LoRAs
            available_loras = all_loras

        if not available_loras:
            if selected_model:
                QMessageBox.information(None, "No Compatible LoRA Adapters",
                                      f"No LoRA adapters are compatible with the selected model '{selected_display_name}'.\n\n"
                                      f"Model type: {selected_model.model_type.value}\n\n"
                                      "Please install compatible LoRA adapters or select a different model.")
            else:
                QMessageBox.information(None, "No LoRA Adapters",
                                      "No LoRA adapters are currently installed.\n\n"
                                      "Please install LoRA adapters first using the Model Management panel.")
            return

        # Create LoRA selection dialog
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QPushButton

        dialog = QDialog(self.sidebar)
        dialog.setWindowTitle("Select LoRA Adapter")
        dialog.setModal(True)

        layout = QVBoxLayout()

        # List of available LoRAs
        lora_list = QListWidget()
        for lora in available_loras:
            display_name = lora.display_name or lora.name
            item = QListWidgetItem(f"{display_name} ({lora.base_model_type.value if lora.base_model_type else 'Any'})")
            item.setData(Qt.UserRole, lora)
            lora_list.addItem(item)

        layout.addWidget(lora_list)

        # Buttons
        buttons_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        add_btn = QPushButton("Add LoRA")
        add_btn.clicked.connect(dialog.accept)
        add_btn.setDefault(True)

        buttons_layout.addWidget(cancel_btn)
        buttons_layout.addWidget(add_btn)
        layout.addLayout(buttons_layout)

        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted and lora_list.currentItem():
            selected_lora = lora_list.currentItem().data(Qt.UserRole)
            self._create_lora_item_widget(selected_lora)

    def _create_lora_item_widget(self, lora):
        """Create a widget for managing a LoRA adapter."""
        from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton

        # Create container frame
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setLineWidth(1)
        layout = QHBoxLayout()
        frame.setLayout(layout)

        # LoRA name label
        display_name = lora.display_name or lora.name
        name_label = QLabel(display_name)
        name_label.setToolTip(f"Path: {lora.path}\nDescription: {lora.description}")
        layout.addWidget(name_label)

        # Scaling factor input
        scaling_label = QLabel("Scale:")
        layout.addWidget(scaling_label)

        scaling_input = QDoubleSpinBox()
        scaling_input.setRange(-5.0, 5.0)
        scaling_input.setValue(lora.default_scaling)
        scaling_input.setSingleStep(0.1)
        scaling_input.setFixedWidth(60)
        layout.addWidget(scaling_input)

        # Remove button
        remove_btn = QPushButton("❌")
        remove_btn.setFixedWidth(30)
        remove_btn.setToolTip("Remove this LoRA adapter")
        remove_btn.clicked.connect(lambda: self._remove_lora_item(frame, lora.name))
        layout.addWidget(remove_btn)

        # Store references for later use
        frame.lora_name = lora.name
        frame.scaling_input = scaling_input

        # Add to the LoRA list
        self.lora_list_layout.addWidget(frame)

    def _remove_lora_item(self, frame, lora_name):
        """Remove a LoRA adapter item from the UI and unload it from the service."""
        # Unload from the image service first
        if self.image_service:
            self.image_service.unload_lora(lora_name)

        # Remove from layout
        self.lora_list_layout.removeWidget(frame)
        frame.setParent(None)
        frame.deleteLater()

    def _get_selected_loras(self):
        """Get list of selected LoRA adapters with their scaling factors and paths."""
        lora_configs = []

        # Get all installed LoRAs for path lookup
        installed_loras = self.model_manager.get_installed_loras()
        lora_path_map = {lora.name: lora.path for lora in installed_loras}

        # Iterate through all LoRA item frames
        for i in range(self.lora_list_layout.count()):
            item = self.lora_list_layout.itemAt(i)
            if item:
                widget = item.widget()
                if hasattr(widget, 'lora_name') and hasattr(widget, 'scaling_input'):
                    lora_name = widget.lora_name
                    lora_path = lora_path_map.get(lora_name)

                    if lora_path:  # Only include if we can find the path
                        lora_config = {
                            'name': lora_name,
                            'path': lora_path,
                            'scaling': widget.scaling_input.value()
                        }
                        lora_configs.append(lora_config)
                    else:
                        print(f"Warning: Could not find path for LoRA '{lora_name}'")

        return lora_configs

    def _get_model_type_for_enhancement(self) -> str:
        """Get the model type (sd/sdxl) for prompt enhancement based on selected model."""
        selected_display_name = self.model_select.currentText()
        if not selected_display_name:
            return "sd"  # Default fallback to SD

        selected_model = self.model_display_mapping.get(selected_display_name)
        if selected_model and selected_model.model_type == ModelType.STABLE_DIFFUSION_XL:
            return "sdxl"
        return "sd"

    def _get_model_specific_enhancement_prompt(self, model_type: str) -> str:
        """Get enhancement prompt optimized for the specific model type."""
        if model_type == "sdxl":
            return """You are an expert at enhancing prompts for SDXL models. SDXL can handle longer, more detailed prompts with complex compositions.

CRITICAL RULES FOR SDXL ENHANCEMENT:
- Output ONLY positive prompt|negative prompt (separated by single |)
- SDXL works best with detailed, descriptive language and complex scenes
- Include specific styles, lighting conditions, camera angles, and artistic details
- Keep positive prompt under 150 words, negative prompt under 60 words
- Focus on visual storytelling, composition, and artistic elements
- Use descriptive adjectives and specific terminology
- Consider multiple subjects, backgrounds, and atmospheric effects

REQUIRED FORMAT: detailed positive prompt|concise negative prompt"""
        else:  # SD 1.5
            return """You are a professional prompt engineer specializing in Stable Diffusion 1.5. Create highly detailed, professional-quality prompts that produce stunning results.

CRITICAL RULES FOR SD ENHANCEMENT:
- Output ONLY positive prompt|negative prompt (separated by single |)
- Use weighted terms in parentheses: (important term:weight) where weight is 1.1-1.4 for emphasis
- Include artistic style references, famous artists, and professional techniques
- Create comprehensive positive prompts with detailed descriptions, lighting, composition, and quality
- Generate detailed negative prompts with weighted quality issues to avoid
- Keep positive prompt under 200 words, negative prompt under 80 words
- Focus on professional photography/art styles, technical quality, and artistic excellence
- Include specific camera angles, lighting styles, and artistic influences
- Use descriptive adjectives, technical terms, and professional vocabulary

PROFESSIONAL ENHANCEMENT STRUCTURE:
1. Start with detailed subject description with weights
2. Add visual details, lighting, and atmosphere
3. Include artistic style and technique references
4. Add composition and camera details
5. Include quality and technical specifications
6. End with artistic influences and masters

NEGATIVE PROMPT STRUCTURE:
1. Start with major quality issues with high weights (1.3-1.5)
2. Add technical problems and artifacts
3. Include style issues and unwanted elements
4. Add composition and focus problems
5. Include color and lighting issues

REQUIRED FORMAT: professional detailed positive prompt|comprehensive negative prompt"""

    def enhance_prompt(self):
        """Enhance the current prompt using AI with model-specific optimization."""
        # Get current prompt
        current_prompt = self.prompt_input.toPlainText().strip()
        if not current_prompt:
            QMessageBox.warning(self.sidebar, "No Prompt",
                              "Please enter a prompt before enhancing it.")
            return

        # Get settings manager from parent window
        main_window = self.sidebar.window()
        if not hasattr(main_window, 'settings_manager'):
            QMessageBox.warning(self.sidebar, "Settings Error",
                              "Settings manager not available.")
            return

        settings = main_window.settings_manager.load_settings()

        # Check if text model is configured
        if not settings.text_model_path:
            QMessageBox.warning(self.sidebar, "Text Model Not Configured",
                              "Please configure a text model path in Settings > Text Enhancement.")
            return

        # Check if enhancement is already in progress
        if self.text_enhancer.is_loading:
            # Cancel current enhancement
            if self.text_enhancer.cancel_enhancement():
                self.enhance_btn.setEnabled(True)
                self.enhance_btn.setText("✨ Enhance Prompt")
                self.image_label.setText("Enhancement cancelled")
            return

        # Auto-detect model type for optimized enhancement
        model_type = self._get_model_type_for_enhancement()
        enhancement_prompt = self._get_model_specific_enhancement_prompt(model_type)

        # Show which model type is being used for enhancement
        model_name = self.model_select.currentText() or "Unknown"
        print(f"🔍 DEBUG: Enhancing prompt for {model_name} ({model_type.upper()} model)")

        # Disable enhance button during processing
        self.enhance_btn.setEnabled(False)
        self.enhance_btn.setText("✨ Enhancing... (Click to cancel)")

        # Start enhancement with model-specific prompt
        success = self.text_enhancer.enhance_prompt_async(
            model_path=settings.text_model_path,
            enhancement_prompt=enhancement_prompt,  # Use model-specific prompt
            user_prompt=current_prompt,
            on_success=self._on_prompt_enhanced,
            on_error=self._on_prompt_enhancement_error,
            enable_openai=settings.enable_openai_enhancement,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model
        )

        if not success:
            # Re-enable button if enhancement failed to start
            self.enhance_btn.setEnabled(True)
            self.enhance_btn.setText("✨ Enhance Prompt")

    def _on_prompt_enhanced(self, enhanced_prompt: str):
        """Handle successful prompt enhancement."""
        # Re-enable enhance button
        self.enhance_btn.setEnabled(True)
        self.enhance_btn.setText("✨ Enhance Prompt")

        # Show accept/reject dialog
        dialog = EnhancePromptDialog(
            original_prompt=self.prompt_input.toPlainText().strip(),
            enhanced_prompt=enhanced_prompt,
            parent=self.sidebar
        )

        if dialog.exec_() == dialog.Accepted:
            accepted, selected_prompt = dialog.get_result()
            if accepted:
                # User accepted the enhanced prompt - split on "|" to separate positive and negative
                if "|" in selected_prompt:
                    positive_prompt, negative_prompt = selected_prompt.split("|", 1)
                    # Remove extra whitespace
                    positive_prompt = positive_prompt.strip()
                    negative_prompt = negative_prompt.strip()

                    # Populate both prompt fields
                    self.prompt_input.setPlainText(positive_prompt)
                    self.negative_prompt_input.setPlainText(negative_prompt)
                else:
                    # Fallback: if no "|" separator, put everything in positive prompt
                    self.prompt_input.setPlainText(selected_prompt)

    def _on_prompt_enhancement_error(self, error_msg: str):
        """Handle prompt enhancement error."""
        # Re-enable enhance button
        self.enhance_btn.setEnabled(True)
        self.enhance_btn.setText("✨ Enhance Prompt")

        # Show error message
        QMessageBox.warning(self.sidebar, "Enhancement Error", error_msg)

    def save_image(self):
        """Save the generated image."""
        if not hasattr(self, 'current_image'):
            return

        file_path, _ = QFileDialog.getSaveFileName(
            None, 'Save Image', '', 'PNG Files (*.png);;JPEG Files (*.jpg)'
        )
        if file_path:
            self.current_image.save(file_path)

    def reset_image(self):
        """Reset the image display to clear the generated image."""
        # Clear the current image
        if hasattr(self, 'current_image'):
            delattr(self, 'current_image')

        # Reset the image label to default state
        self.image_label.clear()
        self.image_label.setText("No image generated yet")
        self.image_label.setStyleSheet("""
            border: 2px dashed #aaa;
            background-color: #f0f0f0;
            QLabel {
                qproperty-alignment: AlignCenter;
            }
        """)

        # Update button states
        self.save_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)

        print("Image display reset")

    def _add_ip_adapter(self):
        """Add an IP-Adapter selection widget."""
        # Get available IP-Adapters
        available_ip_adapters = self.model_manager.get_all_ip_adapters()

        if not available_ip_adapters:
            QMessageBox.information(None, "No IP-Adapters",
                                  "No IP-Adapters are currently installed.\n\n"
                                  "Please install IP-Adapters first using the Model Management panel.")
            return

        # Create IP-Adapter selection dialog
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QPushButton

        dialog = QDialog(self.sidebar)
        dialog.setWindowTitle("Select IP-Adapter")
        dialog.setModal(True)

        layout = QVBoxLayout()

        # List of available IP-Adapters
        ip_adapter_list = QListWidget()
        for ip_adapter in available_ip_adapters:
            display_name = ip_adapter.display_name or ip_adapter.name
            item = QListWidgetItem(f"{display_name} ({ip_adapter.adapter_type})")
            item.setData(Qt.UserRole, ip_adapter)
            ip_adapter_list.addItem(item)

        layout.addWidget(ip_adapter_list)

        # Buttons
        buttons_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        add_btn = QPushButton("Add IP-Adapter")
        add_btn.clicked.connect(dialog.accept)
        add_btn.setDefault(True)

        buttons_layout.addWidget(cancel_btn)
        buttons_layout.addWidget(add_btn)
        layout.addLayout(buttons_layout)

        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted and ip_adapter_list.currentItem():
            selected_ip_adapter = ip_adapter_list.currentItem().data(Qt.UserRole)
            self._create_ip_adapter_item_widget(selected_ip_adapter)

    def _create_ip_adapter_item_widget(self, ip_adapter):
        """Create a widget for managing an IP-Adapter."""
        from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton

        # Create container frame
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setLineWidth(1)
        layout = QHBoxLayout()
        frame.setLayout(layout)

        # IP-Adapter name label
        display_name = ip_adapter.display_name or ip_adapter.name
        name_label = QLabel(display_name)
        name_label.setToolTip(f"Path: {ip_adapter.path}\nType: {ip_adapter.adapter_type}\nDescription: {ip_adapter.description}")
        layout.addWidget(name_label)

        # Scale factor input
        scale_label = QLabel("Scale:")
        layout.addWidget(scale_label)

        scale_input = QDoubleSpinBox()
        scale_input.setRange(0.0, 2.0)
        scale_input.setValue(ip_adapter.default_scale)
        scale_input.setSingleStep(0.1)
        scale_input.setFixedWidth(60)
        scale_input.setToolTip("IP-Adapter conditioning scale (0.0-2.0)")
        layout.addWidget(scale_input)

        # Remove button
        remove_btn = QPushButton("❌")
        remove_btn.setFixedWidth(30)
        remove_btn.setToolTip("Remove this IP-Adapter")
        remove_btn.clicked.connect(lambda: self._remove_ip_adapter_item(frame, ip_adapter.name))
        layout.addWidget(remove_btn)

        # Store references for later use
        frame.ip_adapter_name = ip_adapter.name
        frame.scale_input = scale_input

        # Add to the IP-Adapter list
        self.ip_adapter_list_layout.addWidget(frame)

    def _remove_ip_adapter_item(self, frame, ip_adapter_name):
        """Remove an IP-Adapter item from the UI."""
        # Remove from layout
        self.ip_adapter_list_layout.removeWidget(frame)
        frame.setParent(None)
        frame.deleteLater()

    def _get_selected_ip_adapters(self):
        """Get list of selected IP-Adapters with their scale factors and paths."""
        ip_adapter_configs = []

        # Get all installed IP-Adapters for path lookup
        installed_ip_adapters = self.model_manager.get_all_ip_adapters()
        ip_adapter_path_map = {adapter.name: adapter.path for adapter in installed_ip_adapters}

        # Iterate through all IP-Adapter item frames
        for i in range(self.ip_adapter_list_layout.count()):
            item = self.ip_adapter_list_layout.itemAt(i)
            if item:
                widget = item.widget()
                if hasattr(widget, 'ip_adapter_name') and hasattr(widget, 'scale_input'):
                    ip_adapter_name = widget.ip_adapter_name
                    ip_adapter_path = ip_adapter_path_map.get(ip_adapter_name)

                    if ip_adapter_path:  # Only include if we can find the path
                        ip_adapter_config = {
                            'name': ip_adapter_name,
                            'path': ip_adapter_path,
                            'scale': widget.scale_input.value()
                        }
                        ip_adapter_configs.append(ip_adapter_config)
                    else:
                        print(f"Warning: Could not find path for IP-Adapter '{ip_adapter_name}'")

        return ip_adapter_configs

    def _upload_reference_image(self):
        """Upload a reference image for IP-Adapter."""
        file_path, _ = QFileDialog.getOpenFileName(
            None, 'Select Reference Image', '',
            'Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)'
        )

        if file_path:
            self.ref_image_path = file_path
            # Display image thumbnail in the label
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                # Scale to fit the label while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    self.ref_image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.ref_image_label.setPixmap(scaled_pixmap)
                self.ref_image_label.setText("")  # Clear text when showing image
            else:
                self.ref_image_label.setText("Invalid image file")
                self.ref_image_path = None

    def _clear_reference_image(self):
        """Clear the reference image."""
        self.ref_image_path = None
        self.ref_image_label.clear()
        self.ref_image_label.setText("No reference image")
