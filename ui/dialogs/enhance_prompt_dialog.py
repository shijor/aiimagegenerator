"""
Dialog for accepting or rejecting enhanced prompts.
"""
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class EnhancePromptDialog(QDialog):
    """Dialog to show enhanced prompt and allow user to accept or reject it."""

    def __init__(self, original_prompt: str, enhanced_prompt: str, parent=None):
        super().__init__(parent)
        self.original_prompt = original_prompt
        self.enhanced_prompt = enhanced_prompt
        self.accepted_enhancement = False

        self.setWindowTitle("Enhanced Prompt")
        self.setModal(True)
        self.setFixedSize(600, 400)

        # Center the dialog on the screen
        self.center_on_screen()

        self._init_ui()

    def _init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title_label = QLabel("✨ Prompt Enhancement")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #1976D2;")
        layout.addWidget(title_label)

        # Original prompt section
        original_group = QVBoxLayout()
        original_label = QLabel("Original Prompt:")
        original_label.setStyleSheet("font-weight: bold; color: #666;")
        original_group.addWidget(original_label)

        self.original_text = QTextEdit()
        self.original_text.setPlainText(self.original_prompt)
        self.original_text.setReadOnly(True)
        self.original_text.setMaximumHeight(60)
        self.original_text.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
                color: #666;
            }
        """)
        original_group.addWidget(self.original_text)
        layout.addLayout(original_group)

        # Enhanced prompt section
        enhanced_group = QVBoxLayout()
        enhanced_label = QLabel("Enhanced Prompt:")
        enhanced_label.setStyleSheet("font-weight: bold; color: #1976D2;")
        enhanced_group.addWidget(enhanced_label)

        self.enhanced_text = QTextEdit()
        self.enhanced_text.setPlainText(self.enhanced_prompt)
        self.enhanced_text.setReadOnly(True)
        self.enhanced_text.setMaximumHeight(100)
        self.enhanced_text.setStyleSheet("""
            QTextEdit {
                background-color: #f0f8ff;
                border: 2px solid #1976D2;
                border-radius: 4px;
                padding: 5px;
                color: #000;
            }
        """)
        enhanced_group.addWidget(self.enhanced_text)
        layout.addLayout(enhanced_group)

        # Info text
        info_label = QLabel("Would you like to use the enhanced prompt?")
        info_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(info_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        button_layout.addStretch()

        reject_btn = QPushButton("❌ Keep Original")
        reject_btn.clicked.connect(self.reject)
        reject_btn.setMinimumWidth(120)
        button_layout.addWidget(reject_btn)

        accept_btn = QPushButton("✅ Use Enhanced")
        accept_btn.clicked.connect(self._accept_enhancement)
        accept_btn.setDefault(True)
        accept_btn.setMinimumWidth(120)
        accept_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        button_layout.addWidget(accept_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _accept_enhancement(self):
        """Accept the enhanced prompt."""
        self.accepted_enhancement = True
        self.accept()

    def center_on_screen(self):
        """Center this dialog on the screen."""
        # Get the screen where the dialog will be shown
        screen = self.screen()
        if not screen:
            return

        # Get available screen geometry (excluding taskbar, etc.)
        screen_rect = screen.availableGeometry()

        # Calculate center position of screen
        screen_center_x = screen_rect.x() + screen_rect.width() // 2
        screen_center_y = screen_rect.y() + screen_rect.height() // 2

        # Calculate position to center dialog on screen
        dialog_x = screen_center_x - self.width() // 2
        dialog_y = screen_center_y - self.height() // 2

        # Ensure dialog stays within screen bounds
        dialog_x = max(screen_rect.x(), min(dialog_x, screen_rect.x() + screen_rect.width() - self.width()))
        dialog_y = max(screen_rect.y(), min(dialog_y, screen_rect.y() + screen_rect.height() - self.height()))

        # Move dialog to calculated position
        self.move(dialog_x, dialog_y)

    def get_result(self) -> tuple[bool, str]:
        """
        Get the dialog result.

        Returns:
            tuple: (accepted_enhancement, selected_prompt)
        """
        if self.accepted_enhancement:
            return True, self.enhanced_prompt
        else:
            return False, self.original_prompt
