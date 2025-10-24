"""
Model loading dialog for lazy loading.
"""
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


class ModelLoadingDialog(QDialog):
    """Dialog shown during model loading with progress and cancel option."""

    cancelled = pyqtSignal()  # Emitted when user cancels loading

    def __init__(self, model_name: str, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.setWindowTitle("Loading AI Model")
        self.setModal(True)
        self.setFixedSize(400, 200)

        # Center the dialog
        if parent:
            parent_rect = parent.geometry()
            self.move(
                parent_rect.x() + (parent_rect.width() - self.width()) // 2,
                parent_rect.y() + (parent_rect.height() - self.height()) // 2
            )

        self._init_ui()

    def _init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout()

        # Title
        title_label = QLabel("🚀 Loading AI Model")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # Model name
        model_label = QLabel(f"Model: {self.model_name}")
        model_label.setAlignment(Qt.AlignCenter)
        model_font = QFont()
        model_font.setPointSize(10)
        model_label.setFont(model_font)
        layout.addWidget(model_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(25)
        layout.addWidget(self.progress_bar)

        # Status message
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Buttons
        button_layout = QHBoxLayout()

        # Cancel button
        self.cancel_btn = QPushButton("❌ Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        self.cancel_btn.setMinimumHeight(35)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

        # Set layout
        self.setLayout(layout)

        # Style the dialog
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                border-radius: 8px;
            }
            QLabel {
                color: #333;
                margin: 5px;
            }
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 4px;
                text-align: center;
                background-color: #fff;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)

    def update_progress(self, message: str, percentage: int):
        """Update the progress bar and status message."""
        self.status_label.setText(message)
        self.progress_bar.setValue(percentage)

        # Force UI update
        self.repaint()

    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Cancelling...")
        self.status_label.setText("Cancelling model loading...")
        self.cancelled.emit()

    def closeEvent(self, event):
        """Handle dialog close event."""
        # Don't allow closing via X button - must use cancel
        event.ignore()
