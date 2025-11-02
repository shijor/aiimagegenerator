"""
Vertical toolbar component for main navigation.
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import pyqtSignal


class VerticalToolbar(QWidget):
    """Vertical toolbar with icon buttons for navigation."""

    mode_changed = pyqtSignal(int)  # Emits mode index (0=image gen, 1=model mgmt, 2=settings)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(50)
        self.current_mode = 0
        self._init_ui()
        self._setup_styling()
        self._change_mode(0)  # Set initial active state for image generator

    def _init_ui(self):
        """Initialize the toolbar UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 10, 5, 10)
        layout.setSpacing(8)

        # Image Generation Button
        self.image_gen_btn = QPushButton('🎨')
        self.image_gen_btn.setToolTip("Image Generation")
        self.image_gen_btn.clicked.connect(lambda: self._change_mode(0))
        self.image_gen_btn.setFixedSize(40, 40)
        self.image_gen_btn.setObjectName("gen_icon")
        layout.addWidget(self.image_gen_btn)

        # Model Management Button
        self.model_btn = QPushButton('🤖')
        self.model_btn.setToolTip("Model Management")
        self.model_btn.clicked.connect(lambda: self._change_mode(1))
        self.model_btn.setFixedSize(40, 40)
        self.model_btn.setObjectName("model_icon")
        layout.addWidget(self.model_btn)

        # Settings Button
        self.settings_btn = QPushButton('⚙️')
        self.settings_btn.setToolTip("Settings")
        self.settings_btn.clicked.connect(lambda: self._change_mode(2))
        self.settings_btn.setFixedSize(40, 40)
        self.settings_btn.setObjectName("settings_icon")
        layout.addWidget(self.settings_btn)

        layout.addStretch()
        self.setLayout(layout)

    def _setup_styling(self):
        """Set up theme-independent styling for toolbar buttons."""
        # Apply stylesheet to ensure emoji icons are always visible regardless of theme
        self.setStyleSheet("""
            /* Base button styling - high contrast for visibility */
            QPushButton#gen_icon, QPushButton#model_icon, QPushButton#settings_icon {
                background-color: transparent;
                border: none;
                border-radius: 6px;
                font-size: 20px;
                color: #000000;  /* Black text for maximum visibility */
                padding: 0px;
                margin: 0px;
                text-align: center;
            }

            /* Active button styling */
            QPushButton#gen_icon.active_icon, QPushButton#model_icon.active_icon, QPushButton#settings_icon.active_icon {
                background-color: #1976D2;
                color: #ffffff;  /* White text on blue background */
            }

            /* Hover effects for inactive buttons */
            QPushButton#gen_icon:hover, QPushButton#model_icon:hover, QPushButton#settings_icon:hover {
                background-color: #f0f0f0;
                color: #000000;
            }

            /* Active button hover (override) */
            QPushButton#gen_icon.active_icon:hover, QPushButton#model_icon.active_icon:hover, QPushButton#settings_icon.active_icon:hover {
                background-color: #1565C0;
                color: #ffffff;
            }
        """)

    def _change_mode(self, mode: int):
        """Change the active mode and update button styling."""
        if mode == self.current_mode:
            return

        self.current_mode = mode

        # Reset all buttons to their base styles
        self.image_gen_btn.setObjectName("gen_icon")
        self.model_btn.setObjectName("model_icon")
        self.settings_btn.setObjectName("settings_icon")

        # Add active overlay to the selected button
        if mode == 0:
            self.image_gen_btn.setObjectName("gen_icon active_icon")
        elif mode == 1:
            self.model_btn.setObjectName("model_icon active_icon")
        elif mode == 2:
            self.settings_btn.setObjectName("settings_icon active_icon")

        # Force style refresh
        for btn in [self.image_gen_btn, self.model_btn, self.settings_btn]:
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        # Emit signal
        self.mode_changed.emit(mode)

    def get_current_mode(self) -> int:
        """Get the current active mode."""
        return self.current_mode
