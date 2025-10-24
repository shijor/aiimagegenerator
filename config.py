"""
Application configuration constants.
"""
import os

# Application info
APP_NAME = "AI Image Generator"
APP_VERSION = "1.0.0"

# Window settings
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700
MIN_WINDOW_WIDTH = 700
MIN_WINDOW_HEIGHT = 500

# UI dimensions
VERTICAL_TOOLBAR_WIDTH = 50
SIDEBAR_MAX_WIDTH = 350
IMAGE_DISPLAY_SIZE = 512

# File extensions
MODEL_EXTENSIONS = ['.safetensors', '.ckpt']
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']

# Directories
MODELS_DIR = "models"
RESOURCES_DIR = "resources"

# Default settings
DEFAULT_MODEL = "CompVis/stable-diffusion-v1-4"
DEFAULT_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 7.5

# Performance settings
ENABLE_ATTENTION_SLICING = True
USE_FP16 = True

# UI styling
STYLESHEET = """
QWidget {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 11px;
}
QGroupBox {
    font-weight: bold;
    border: 2px solid #cccccc;
    border-radius: 5px;
    margin-top: 1ex;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 10px 0 10px;
}
QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    padding: 6px;
    border: 1px solid #cccccc;
    border-radius: 4px;
    background-color: white;
}
QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 2px solid #0078d4;
}
QPushButton {
    padding: 10px 20px;
    background-color: #0078d4;
    color: white;
    border: none;
    border-radius: 4px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #106ebe;
}
QPushButton:pressed {
    background-color: #005a9e;
}
QPushButton:disabled {
    background-color: #cccccc;
    color: #666666;
}
QLabel {
    color: #333333;
}
#toggle_btn {
    padding: 2px 8px;
    font-size: 14px;
    min-width: 30px;
    max-width: 30px;
    background-color: #f0f0f0;
    color: #333;
    border: 1px solid #ccc;
}
#toggle_btn:hover {
    background-color: #e0e0e0;
}

/* Minimalist Black Buttons with Green Active State */
#gen_icon, #model_icon, #settings_icon {
    background-color: #000000;
    border: none;
    border-radius: 4px;
}

#gen_icon:hover, #model_icon:hover, #settings_icon:hover {
    background-color: #333333;
}

/* Active state - Green */
#active_icon {
    background-color: #28a745 !important;
    border: 2px solid #1e7e34;
}

#active_icon:hover {
    background-color: #218838 !important;
}
"""
