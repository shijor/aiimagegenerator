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

        # Parameters Group
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout()

        # Model Selection with refresh button
        model_layout = QHBoxLayout()
        self.model_select = QComboBox()
        self._refresh_model_dropdown()
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
        """Refresh the model dropdown with current installed models."""
        current_selection = self.model_select.currentText()
        self.model_select.clear()

        # Get all installed models - they should all be valid for image generation
        # since they were validated during installation
        installed_models = self.model_manager.get_installed_models()
        model_names = [model.name for model in installed_models]

        self.model_select.addItems(model_names)

        # Try to restore previous selection
        if current_selection in model_names:
            self.model_select.setCurrentText(current_selection)
        else:
            # Default to the first model or "Stable Diffusion v1.4" if available
            default_model = "Stable Diffusion v1.4"
            if default_model in model_names:
                self.model_select.setCurrentText(default_model)
            elif model_names:
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

        # Get selected model
        selected_model_name = self.model_select.currentText()
        if not selected_model_name:
            return

        # Get the model info for the selected model
        selected_model = None
        for model in self.model_manager.get_installed_models():
            if model.name == selected_model_name:
                selected_model = model
                break

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

    def _on_aspect_ratio_changed(self, aspect_ratio: str):
        """Handle aspect ratio selection change."""
        if aspect_ratio == "1:1":
            # Square aspect ratio
            self.width_input.setValue(512)
            self.height_input.setValue(512)
        elif aspect_ratio == "9:16":
            # Portrait aspect ratio (9:16 means width:height = 9:16)
            self.width_input.setValue(384)  # 512 * (9/16) ≈ 288, but let's use 384 for better compatibility
            self.height_input.setValue(640)  # 512 * (16/9) ≈ 910, but let's use 640
        elif aspect_ratio == "16:9":
            # Landscape aspect ratio (16:9 means width:height = 16:9)
            self.width_input.setValue(640)  # 512 * (16/9) ≈ 910, but let's use 640
            self.height_input.setValue(384)  # 512 * (9/16) ≈ 288, but let's use 384
        elif aspect_ratio == "Custom":
            # Don't change current values for custom
            pass



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

    def save_image(self):
        """Save the generated image."""
        if not hasattr(self, 'current_image'):
            return

        file_path, _ = QFileDialog.getSaveFileName(
            None, 'Save Image', '', 'PNG Files (*.png);;JPEG Files (*.jpg)'
        )
        if file_path:
            self.current_image.save(file_path)
