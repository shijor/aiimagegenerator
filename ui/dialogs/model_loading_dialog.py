"""
Minimalistic model loading dialog with essential progress information.
"""
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton, QHBoxLayout, QWidget
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import time


class ModelLoadingDialog(QDialog):
    """Minimalistic dialog shown during model loading with progress and cancel option."""

    cancelled = pyqtSignal()  # Emitted when user cancels loading

    def __init__(self, model_name: str, model_path: str = "", model_size_mb: float = 0, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.model_path = model_path
        self.model_size_mb = model_size_mb

        self.setWindowTitle("Loading AI Model")
        self.setModal(True)
        self.setFixedSize(400, 180)

        # Center the dialog
        if parent:
            parent_rect = parent.geometry()
            self.move(
                parent_rect.x() + (parent_rect.width() - self.width()) // 2,
                parent_rect.y() + (parent_rect.height() - self.height()) // 2
            )

        self._init_ui()

    def _init_ui(self):
        """Initialize the minimalistic dialog UI with essential elements."""
        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Model name
        model_label = QLabel(f"Loading: {self.model_name}")
        model_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #333;")
        layout.addWidget(model_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Status message
        self.status_label = QLabel("Initializing...")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #666; font-size: 12px; padding: 5px 0;")
        layout.addWidget(self.status_label)

        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        self.cancel_btn.setMinimumHeight(30)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        layout.addWidget(self.cancel_btn, alignment=Qt.AlignCenter)

        # Set main layout
        self.setLayout(layout)

        # Simple stylesheet
        self.setStyleSheet("QDialog { background-color: white; }")



    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Cancelling...")
        self.status_label.setText("Cancelling model loading...")
        self.cancelled.emit()

    def update_progress(self, message: str, percentage: int):
        """Update the progress bar and status message."""
        self.status_label.setText(message)
        self.progress_bar.setValue(percentage)
