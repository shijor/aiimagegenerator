"""
Base dialog class for model/LoRA metadata dialogs.
"""
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, QScrollArea
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class BaseMetadataDialog(QDialog):
    """Base class for model/LoRA metadata dialogs with common functionality."""

    def __init__(self, title: str, icon: str, item_name: str, item_path: str, parent=None):
        super().__init__(parent)
        self.item_name = item_name
        self.item_path = item_path

        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(700, 600)

        # Center the dialog
        if parent:
            parent_rect = parent.geometry()
            self.move(
                parent_rect.x() + (parent_rect.width() - self.width()) // 2,
                parent_rect.y() + (parent_rect.height() - self.height()) // 2
            )

        self.accepted = False  # Track if user accepted
        self._init_common_ui(title, icon)

    def _init_common_ui(self, title: str, icon: str):
        """Initialize common UI elements for all metadata dialogs."""
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header section with icon and title
        header_widget = QWidget()
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(15)

        # Icon
        icon_label = QLabel(icon)
        icon_font = QFont()
        icon_font.setPointSize(32)
        icon_label.setFont(icon_font)
        header_layout.addWidget(icon_label)

        # Title and subtitle
        title_widget = QWidget()
        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(5)

        title_label = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #1976D2;")
        title_layout.addWidget(title_label)

        item_label = QLabel(f"📁 {self.item_name}")
        item_font = QFont()
        item_font.setPointSize(12)
        item_label.setFont(item_font)
        item_label.setStyleSheet("color: #666;")
        title_layout.addWidget(item_label)

        title_widget.setLayout(title_layout)
        header_layout.addWidget(title_widget, stretch=1)

        header_widget.setLayout(header_layout)
        layout.addWidget(header_widget)

        # Add separator
        separator = QWidget()
        separator.setFixedHeight(2)
        separator.setStyleSheet("background-color: #e0e0e0; border-radius: 1px;")
        layout.addWidget(separator)

        # Scrollable content area - to be implemented by subclasses
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setMinimumHeight(350)
        self.scroll_area.setMaximumHeight(350)

        # Content widget to be set by subclasses
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(15)

        # File path info
        self._add_path_info()

        # Metadata form - to be implemented by subclasses
        self._setup_metadata_form()

        self.content_widget.setLayout(self.content_layout)
        self.scroll_area.setWidget(self.content_widget)
        layout.addWidget(self.scroll_area)

        # Buttons section
        button_widget = QWidget()
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(10)

        button_layout.addStretch()

        # Cancel button
        self.cancel_btn = QPushButton("❌ Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setMinimumWidth(100)
        button_layout.addWidget(self.cancel_btn)

        # Install/Save button - to be customized by subclasses
        self.action_btn = self._create_action_button()
        self.action_btn.clicked.connect(self.accept)
        self.action_btn.setMinimumHeight(40)
        self.action_btn.setDefault(True)
        button_layout.addWidget(self.action_btn)

        button_widget.setLayout(button_layout)
        layout.addWidget(button_widget)

        # Set layout
        self.setLayout(layout)

    def _add_path_info(self):
        """Add file path information section."""
        path_group = QWidget()
        path_layout = QVBoxLayout()
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.setSpacing(5)

        path_label = QLabel("📂 Source Path:")
        path_label.setStyleSheet("font-weight: bold; color: #333;")
        path_layout.addWidget(path_label)

        path_display = QLabel(self.item_path)
        path_display.setWordWrap(True)
        path_display.setStyleSheet("""
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 8px;
            color: #666;
            font-family: 'Segoe UI', monospace;
            font-size: 11px;
        """)
        path_layout.addWidget(path_display)

        path_group.setLayout(path_layout)
        self.content_layout.addWidget(path_group)

    def _setup_metadata_form(self):
        """Setup metadata form - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _setup_metadata_form")

    def _create_action_button(self) -> QPushButton:
        """Create the action button - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _create_action_button")

    def accept(self):
        """Handle accept button click."""
        self.accepted = True
        super().accept()

    def reject(self):
        """Handle reject button click."""
        self.accepted = False
        super().reject()

    def get_metadata(self) -> dict:
        """Get the collected metadata - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_metadata")
