"""
Styling constants for model management UI components.
"""

# Model list item styles
MODEL_LIST_ITEM_AVAILABLE_STYLE = """
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
"""

MODEL_LIST_ITEM_INSTALLED_STYLE = """
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
"""

LORA_LIST_ITEM_AVAILABLE_STYLE = """
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
"""

LORA_LIST_ITEM_INSTALLED_STYLE = """
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
"""

# Button styles
INSTALL_BUTTON_STYLE = """
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
"""

LORA_INSTALL_BUTTON_STYLE = """
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
"""

EDIT_BUTTON_STYLE = """
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
"""

DELETE_BUTTON_STYLE = """
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
"""

# Progress bar styles
PROGRESS_BAR_STYLE = """
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
"""

# Label styles
TYPE_LABEL_STYLE = """
    font-size: 11px;
    color: #666;
    background-color: #f0f0f0;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 500;
"""

LORA_TYPE_LABEL_STYLE = """
    font-size: 11px;
    color: #666;
    background-color: #fff3e0;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 500;
"""

SIZE_LABEL_STYLE = "color: #555; font-size: 11px; font-weight: 500;"

PATH_LABEL_STYLE = """
    color: #777;
    font-size: 10px;
    font-family: 'Segoe UI', monospace;
"""

DESCRIPTION_LABEL_STYLE = """
    color: #777;
    font-size: 10px;
    font-family: 'Segoe UI', monospace;
"""

# Progress display styles
PROGRESS_DISPLAY_STYLE = """
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
"""

PROGRESS_LABEL_STYLE = "font-size: 11px; margin: 0px; padding: 0px;"
