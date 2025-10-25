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
from models.generation_params import GenerationParams
from ui.dialogs.model_loading_dialog import ModelLoadingDialog


class ImageGenerationPanel:
    """Panel for image generation functionality."""

    def __init__(self, model_manager: ModelManager, image_service: ImageGenerationService):
        self.model_manager = model_manager
        self.image_service = image_service
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
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText('Enter your prompt here...')
        self.prompt_input.setMaximumHeight(100)
        prompts_layout.addWidget(self.prompt_input)

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

        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 999999)
        self.seed_input.setValue(0)
        params_layout.addRow("Seed (0=random):", self.seed_input)

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

        # Toggle Sidebar Button
        self.toggle_btn = QPushButton('◀')
        self.toggle_btn.clicked.connect(self.toggle_sidebar)
        self.toggle_btn.setObjectName("toggle_btn")
        layout.addWidget(self.toggle_btn, alignment=Qt.AlignLeft)

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
        layout.addWidget(self.progress_bar)

        # Save Button
        self.save_btn = QPushButton('💾 Save Image')
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)
        self.save_btn.setMinimumHeight(35)
        layout.addWidget(self.save_btn, alignment=Qt.AlignCenter)

        # Reduce margins and spacing for better space utilization
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        widget.setLayout(layout)
        return widget

    def generate_image(self):
        """Generate image with current parameters using lazy loading."""
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            return

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
        self.loading_dialog = ModelLoadingDialog(selected_model.display_name or selected_model.name, self.generate_btn.parentWidget().window())

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
        # Get parameters
        steps = self.steps_input.value()
        guidance = self.guidance_input.value()
        seed = self.seed_input.value()
        negative = self.negative_prompt_input.toPlainText().strip()
        width = self.width_input.value()
        height = self.height_input.value()
        scheduler = self.scheduler_select.currentText()
        quantization = self.quantization_select.currentText()
        use_xformers = self.xformers_checkbox.isChecked()
        vae_tiling = self.vae_tiling_checkbox.isChecked()

        # Get selected LoRA adapters
        lora_adapters = self._get_selected_loras()

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
            lora_adapters=lora_adapters
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

        # Re-enable UI and disable cancel
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.save_btn.setEnabled(True)

    def cancel_generation(self):
        """Cancel the current image generation."""
        if self.current_generator and self.current_generator.isRunning():
            print("Cancelling image generation...")
            self.current_generator.requestInterruption()
            self.current_generator.wait(5000)  # Wait up to 5 seconds for clean shutdown

            if self.current_generator.isRunning():
                print("Force terminating generation thread...")
                self.current_generator.terminate()
                self.current_generator.wait(2000)

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

    def toggle_sidebar(self):
        """Toggle sidebar visibility."""
        # This will be handled by the parent splitter
        pass

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
        seed = self.seed_input.value()
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
        # Get available LoRA adapters
        available_loras = self.model_manager.get_installed_loras()

        if not available_loras:
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
        scaling_input.setRange(0.0, 2.0)
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
        """Remove a LoRA adapter item from the UI."""
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

    def save_image(self):
        """Save the generated image."""
        if not hasattr(self, 'current_image'):
            return

        file_path, _ = QFileDialog.getSaveFileName(
            None, 'Save Image', '', 'PNG Files (*.png);;JPEG Files (*.jpg)'
        )
        if file_path:
            self.current_image.save(file_path)
