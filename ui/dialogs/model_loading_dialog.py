"""
Model loading dialog for lazy loading.
"""
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton, QHBoxLayout, QWidget
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QPainter, QColor
import time


class ModelLoadingDialog(QDialog):
    """Dialog shown during model loading with progress and cancel option."""

    cancelled = pyqtSignal()  # Emitted when user cancels loading

    def __init__(self, model_name: str, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_percentage = 0
        self.setWindowTitle("Loading AI Model")
        self.setModal(True)
        self.setFixedSize(500, 320)

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
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header section with icon and title
        header_widget = QWidget()
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(10)

        # Icon
        icon_label = QLabel("🤖")
        icon_font = QFont()
        icon_font.setPointSize(24)
        icon_label.setFont(icon_font)
        header_layout.addWidget(icon_label)

        # Title and subtitle
        title_widget = QWidget()
        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(2)

        title_label = QLabel("Loading AI Model")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #1976D2;")
        title_layout.addWidget(title_label)

        model_label = QLabel(f"📦 {self.model_name}")
        model_font = QFont()
        model_font.setPointSize(11)
        model_label.setFont(model_font)
        model_label.setStyleSheet("color: #666;")
        title_layout.addWidget(model_label)

        title_widget.setLayout(title_layout)
        header_layout.addWidget(title_widget, stretch=1)

        header_widget.setLayout(header_layout)
        layout.addWidget(header_widget)

        # Add separator
        separator = QWidget()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #e0e0e0;")
        layout.addWidget(separator)

        # Progress section
        progress_widget = QWidget()
        progress_layout = QVBoxLayout()
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(8)

        # Progress bar with percentage
        progress_container = QWidget()
        progress_container_layout = QVBoxLayout()
        progress_container_layout.setContentsMargins(0, 0, 0, 0)
        progress_container_layout.setSpacing(4)

        progress_header = QWidget()
        progress_header_layout = QHBoxLayout()
        progress_header_layout.setContentsMargins(0, 0, 0, 0)

        progress_label = QLabel("Progress:")
        progress_label.setStyleSheet("font-weight: bold; color: #333;")
        progress_header_layout.addWidget(progress_label)

        self.percentage_label = QLabel("0%")
        self.percentage_label.setStyleSheet("color: #1976D2; font-weight: bold;")
        progress_header_layout.addStretch()
        progress_header_layout.addWidget(self.percentage_label)

        progress_header.setLayout(progress_header_layout)
        progress_container_layout.addWidget(progress_header)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(20)
        self.progress_bar.setTextVisible(False)  # Hide default percentage
        progress_container_layout.addWidget(self.progress_bar)

        progress_container.setLayout(progress_container_layout)
        progress_layout.addWidget(progress_container)

        # Status message
        self.status_label = QLabel("Initializing model loading...")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #555; padding: 5px; background-color: #f8f9fa; border-radius: 4px;")
        self.status_label.setMinimumHeight(40)
        progress_layout.addWidget(self.status_label)

        # Time and speed info
        info_widget = QWidget()
        info_layout = QHBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)

        self.time_label = QLabel("⏱️ Elapsed: 0s")
        self.time_label.setStyleSheet("color: #666; font-size: 11px;")
        info_layout.addWidget(self.time_label)

        info_layout.addStretch()

        self.speed_label = QLabel("")
        self.speed_label.setStyleSheet("color: #666; font-size: 11px;")
        info_layout.addWidget(self.speed_label)

        info_widget.setLayout(info_layout)
        progress_layout.addWidget(info_widget)

        progress_widget.setLayout(progress_layout)
        layout.addWidget(progress_widget)

        # Buttons section
        button_widget = QWidget()
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)

        button_layout.addStretch()

        # Cancel button with better styling
        self.cancel_btn = QPushButton("❌ Cancel Loading")
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        self.cancel_btn.setMinimumHeight(35)
        self.cancel_btn.setMinimumWidth(120)
        button_layout.addWidget(self.cancel_btn)

        button_widget.setLayout(button_layout)
        layout.addWidget(button_widget)

        # Set layout
        self.setLayout(layout)

        # Modern stylesheet
        self.setStyleSheet("""
            QDialog {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 12px;
            }

            QProgressBar {
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                background-color: #f8f9fa;
                padding: 2px;
            }

            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:0.5 #45a049, stop:1 #4CAF50);
                border-radius: 6px;
            }

            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 12px;
            }

            QPushButton:hover {
                background-color: #d32f2f;
                transform: translateY(-1px);
            }

            QPushButton:pressed {
                background-color: #b71c1c;
                transform: translateY(0px);
            }

            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)

        # Start timer for updating elapsed time
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_elapsed_time)
        self.timer.start(1000)  # Update every second

    def update_progress(self, message: str, percentage: int):
        """Update the progress bar and status message."""
        self.status_label.setText(message)
        self.progress_bar.setValue(percentage)
        self.percentage_label.setText(f"{percentage}%")

        # Update speed calculation
        current_time = time.time()
        time_diff = current_time - self.last_update_time

        if time_diff > 0 and percentage > self.last_percentage:
            # Calculate progress per second
            progress_diff = percentage - self.last_percentage
            if progress_diff > 0:
                speed = progress_diff / time_diff
                if speed > 0:
                    eta_seconds = (100 - percentage) / speed
                    if eta_seconds < 60:
                        eta_text = f"⏱️ ETA: {int(eta_seconds)}s"
                    elif eta_seconds < 3600:
                        eta_text = f"⏱️ ETA: {int(eta_seconds/60)}m"
                    else:
                        eta_text = f"⏱️ ETA: {int(eta_seconds/3600)}h"
                    self.speed_label.setText(eta_text)

        self.last_percentage = percentage

        self.last_update_time = current_time

        # Force UI update
        self.repaint()

    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Cancelling...")
        self.status_label.setText("Cancelling model loading...")
        self.cancelled.emit()

    def _update_elapsed_time(self):
        """Update the elapsed time display."""
        elapsed_seconds = int(time.time() - self.start_time)
        if elapsed_seconds < 60:
            time_text = f"⏱️ Elapsed: {elapsed_seconds}s"
        elif elapsed_seconds < 3600:
            minutes = elapsed_seconds // 60
            seconds = elapsed_seconds % 60
            time_text = f"⏱️ Elapsed: {minutes}m {seconds}s"
        else:
            hours = elapsed_seconds // 3600
            minutes = (elapsed_seconds % 3600) // 60
            time_text = f"⏱️ Elapsed: {hours}h {minutes}m"

        self.time_label.setText(time_text)

    def closeEvent(self, event):
        """Handle dialog close event."""
        # Don't allow closing via X button - must use cancel
        event.ignore()
