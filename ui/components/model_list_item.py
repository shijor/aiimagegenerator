"""
UI components for model management - extracted from model_management_panel.py
"""
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QListWidget, QListWidgetItem, QSizePolicy
from PyQt5.QtCore import Qt

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.model_info import ModelInfo, LoRAInfo


class ModelListItem(QWidget):
    """Reusable widget for displaying model information in lists."""

    def __init__(self, model_info: dict, is_available: bool = True, parent=None):
        super().__init__(parent)
        self.model_info = model_info
        self.is_available = is_available
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI."""
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

        name_label = QLabel(self.model_info['name'])
        if self.is_available:
            name_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #1565C0;")
        else:
            # For installed models, show display name with star if default
            display_name = self.model_info.get('display_name', self.model_info['name'])
            if self.model_info.get('is_default'):
                display_name += " ⭐"
            name_label.setText(display_name)
            name_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #1565C0;")
        name_label.setToolTip(self.model_info['name'])  # Full name on hover
        name_label.setMinimumWidth(150)  # Ensure minimum width for text
        name_layout.addWidget(name_label)

        name_layout.addStretch()
        layout.addLayout(name_layout, stretch=4)  # Give name column more space

        # Column 2: Type (fixed width)
        type_text = self.model_info.get('model_type', 'Unknown')
        if not self.is_available and hasattr(self.model_info, 'model_type'):
            # For installed models, get the enum value
            type_text = self.model_info.model_type.value if self.model_info.model_type else 'Unknown'
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
        if self.is_available:
            size_label = QLabel(f"{self.model_info['size_mb']:.1f} MB")
        else:
            # For installed models
            try:
                size_mb = float(self.model_info.size_mb) if self.model_info.size_mb and str(self.model_info.size_mb).replace('.', '').isdigit() else 0.0
            except (ValueError, TypeError):
                size_mb = 0.0
            size_label = QLabel(f"{size_mb:.1f} MB")
        size_label.setStyleSheet("color: #555; font-size: 11px; font-weight: 500;")
        size_label.setFixedWidth(70)
        size_label.setMinimumWidth(70)
        size_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(size_label)

        # Column 4: Path/Description (flexible width, truncated)
        if self.is_available:
            path_text = self.model_info['path']
            if len(path_text) > 50:
                path_text = "..." + path_text[-47:]
            path_label = QLabel(path_text)
            path_label.setStyleSheet("""
                color: #777;
                font-size: 10px;
                font-family: 'Segoe UI', monospace;
            """)
            path_label.setToolTip(self.model_info['path'])  # Full path on hover
            layout.addWidget(path_label, stretch=3)
        else:
            # For installed models, show description
            desc_text = self.model_info.description or "No description"
            if len(desc_text) > 50:
                desc_text = "..." + desc_text[-47:]
            desc_label = QLabel(desc_text)
            desc_label.setStyleSheet("""
                color: #777;
                font-size: 10px;
                font-family: 'Segoe UI', monospace;
            """)
            desc_label.setToolTip(self.model_info.description or "No description")
            desc_label.setMinimumWidth(200)  # Ensure minimum width for description text
            desc_label.setWordWrap(False)  # Prevent wrapping
            layout.addWidget(desc_label, stretch=3)

        # Column 5: Action button (fixed width)
        if self.is_available:
            install_btn = QPushButton('Install')
            install_btn.setProperty("model_name", self.model_info['name'])
            install_btn.setProperty("model_path", self.model_info['path'])
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
        else:
            # For installed models, show Edit/Delete buttons
            buttons_widget = QWidget()
            buttons_layout = QHBoxLayout()
            buttons_layout.setContentsMargins(0, 0, 0, 0)
            buttons_layout.setSpacing(2)

            # Edit button
            edit_btn = QPushButton('Edit')
            edit_btn.setProperty("model_unique_id", self.model_info.unique_id)
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
            delete_btn.setProperty("model_unique_id", self.model_info.unique_id)
            delete_btn.setFixedSize(55, 24)
            delete_btn.setEnabled(not self.model_info.is_default)  # Can't delete default
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

        # Set layout and styling
        self.setLayout(layout)
        if self.is_available:
            self.setStyleSheet("""
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
        else:
            self.setStyleSheet("""
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
        self.setMinimumHeight(32)  # Slightly taller to accommodate text
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)


class LoRAListItem(QWidget):
    """Reusable widget for displaying LoRA information in lists."""

    def __init__(self, lora_info: dict, is_available: bool = True, parent=None):
        super().__init__(parent)
        self.lora_info = lora_info
        self.is_available = is_available
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI."""
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

        if self.is_available:
            name_label = QLabel(self.lora_info['name'])
            name_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #E65100;")
        else:
            # For installed LoRAs, show display name
            display_name = self.lora_info.display_name if self.lora_info.display_name else self.lora_info.name
            name_label = QLabel(display_name)
            name_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #E65100;")
            name_label.setToolTip(f"Name: {self.lora_info.name}\nDisplay: {display_name}")
        name_label.setMinimumWidth(150)  # Ensure minimum width for text
        name_layout.addWidget(name_label)

        name_layout.addStretch()
        layout.addLayout(name_layout, stretch=3)  # Give name column more space

        # Column 2: Type/Base Model Type (fixed width)
        if self.is_available:
            type_label = QLabel("LoRA Adapter")
        else:
            base_model_text = self.lora_info.base_model_type.value if self.lora_info.base_model_type else 'Any'
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
        layout.addWidget(type_label)

        # Column 3: Size (fixed width)
        if self.is_available:
            size_label = QLabel(f"{self.lora_info['size_mb']:.1f} MB")
        else:
            # For installed LoRAs
            try:
                size_mb = float(self.lora_info.size_mb) if self.lora_info.size_mb and str(self.lora_info.size_mb).replace('.', '').isdigit() else 0.0
            except (ValueError, TypeError):
                size_mb = 0.0
            size_label = QLabel(f"{size_mb:.1f} MB")
        size_label.setStyleSheet("color: #555; font-size: 11px; font-weight: 500;")
        size_label.setFixedWidth(70)
        size_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(size_label)

        # Column 4: Path/Trigger Words (flexible width, truncated)
        if self.is_available:
            path_text = self.lora_info['path']
            if len(path_text) > 40:
                path_text = "..." + path_text[-37:]
            path_label = QLabel(path_text)
            path_label.setStyleSheet("""
                color: #777;
                font-size: 10px;
                font-family: 'Segoe UI', monospace;
            """)
            path_label.setToolTip(self.lora_info['path'])  # Full path on hover
            layout.addWidget(path_label, stretch=2)
        else:
            # For installed LoRAs, show trigger words
            trigger_text = ", ".join(self.lora_info.trigger_words[:5]) if self.lora_info.trigger_words else "No triggers"
            if len(self.lora_info.trigger_words or []) > 5:
                trigger_text = trigger_text[:40] + "..."
            trigger_label = QLabel(trigger_text)
            trigger_label.setStyleSheet("""
                color: #777;
                font-size: 10px;
                font-family: 'Segoe UI', monospace;
            """)
            trigger_label.setToolTip(", ".join(self.lora_info.trigger_words) if self.lora_info.trigger_words else "No trigger words")
            trigger_label.setMinimumWidth(200)  # Ensure minimum width for trigger words text
            trigger_label.setWordWrap(False)  # Prevent wrapping
            layout.addWidget(trigger_label, stretch=3)

        # Column 5: Action button (fixed width)
        if self.is_available:
            install_btn = QPushButton('Install')
            install_btn.setProperty("lora_name", self.lora_info['name'])
            install_btn.setProperty("lora_path", self.lora_info['path'])
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
        else:
            # For installed LoRAs, show Edit/Delete buttons
            buttons_widget = QWidget()
            buttons_layout = QHBoxLayout()
            buttons_layout.setContentsMargins(0, 0, 0, 0)
            buttons_layout.setSpacing(2)

            # Edit button
            edit_btn = QPushButton('Edit')
            edit_btn.setProperty("lora_name", self.lora_info.name)
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
            delete_btn.setProperty("lora_name", self.lora_info.name)
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

        # Set layout and styling
        self.setLayout(layout)
        if self.is_available:
            self.setStyleSheet("""
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
        else:
            self.setStyleSheet("""
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
        self.setFixedHeight(30)


class ProgressDisplay(QWidget):
    """Reusable progress display widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)  # Minimal margins
        layout.setSpacing(2)  # Minimal spacing

        self.progress_label = QLabel("Ready to install models")
        self.progress_label.setStyleSheet("font-size: 11px; margin: 0px; padding: 0px;")
        layout.addWidget(self.progress_label)

        from PyQt5.QtWidgets import QProgressBar
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
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)
        self.setFixedHeight(50)  # Minimal fixed height to fit progress bar
        self.setStyleSheet("""
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

    def show_progress(self, message: str, percentage: int):
        """Show progress with message and percentage."""
        self.progress_label.setText(message)
        self.progress_bar.setValue(percentage)
        self.progress_bar.setVisible(True)

    def hide_progress(self):
        """Hide the progress bar and reset."""
        self.progress_bar.setVisible(False)
        self.progress_label.setText("Ready to install models")
        self.progress_bar.setValue(0)
