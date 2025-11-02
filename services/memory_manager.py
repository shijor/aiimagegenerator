"""
Memory management service for automatic model unloading.
"""
import os
from PyQt5.QtCore import QTimer, QObject, pyqtSignal, QEvent
from PyQt5.QtWidgets import QApplication
import time


class MemoryManager(QObject):
    """Manages automatic unloading of models during inactivity."""

    model_auto_unloaded = pyqtSignal(str)  # Emitted when model is auto-unloaded (model_name)

    def __init__(self, image_service, settings_manager):
        super().__init__()
        self.image_service = image_service
        self.settings_manager = settings_manager
        self.inactivity_timer = QTimer(self)
        self.inactivity_timer.timeout.connect(self._on_inactivity_timeout)
        self.inactivity_timer.setSingleShot(True)
        self.last_activity_time = time.time()
        self.is_active = False

        # Install event filter to detect user activity
        self._install_event_filter()

    def _install_event_filter(self):
        """Install event filter to detect user interactions."""
        app = QApplication.instance()
        if app:
            app.installEventFilter(self)

    def eventFilter(self, obj, event):
        """Filter events to detect user activity."""
        # List of events that indicate user activity
        activity_events = [
            QEvent.MouseButtonPress,
            QEvent.MouseButtonRelease,
            QEvent.MouseMove,
            QEvent.KeyPress,
            QEvent.KeyRelease,
            QEvent.Wheel,
            QEvent.FocusIn,
            QEvent.WindowActivate
        ]

        if event.type() in activity_events:
            self._reset_inactivity_timer()

        return super().eventFilter(obj, event)

    def _reset_inactivity_timer(self):
        """Reset the inactivity timer when user activity is detected."""
        self.last_activity_time = time.time()

        settings = self.settings_manager.load_settings()
        if settings.model_timeout_enabled and self.image_service.model and not self.inactivity_timer.isActive():
            timeout_ms = settings.model_timeout_minutes * 60 * 1000
            self.inactivity_timer.start(timeout_ms)

    def start_monitoring(self):
        """Start monitoring for inactivity."""
        self.is_active = True
        self._reset_inactivity_timer()

    def stop_monitoring(self):
        """Stop monitoring for inactivity."""
        self.is_active = False
        self.inactivity_timer.stop()

    def _on_inactivity_timeout(self):
        """Called when inactivity timeout is reached."""
        if self.image_service.model:
            # Get current model name for notification
            current_path = self.image_service._current_model_path
            model_name = "Unknown Model"

            if current_path:
                # Try to extract model name from path
                if os.path.isdir(current_path):
                    model_name = os.path.basename(current_path)
                else:
                    model_name = os.path.basename(current_path)

            # Unload the model
            self.image_service.unload_model()

            # Emit signal for UI notification
            self.model_auto_unloaded.emit(model_name)

    def get_time_until_unload(self) -> int:
        """Get seconds until automatic unload (or -1 if not active)."""
        if not self.inactivity_timer.isActive():
            return -1

        elapsed = time.time() - self.last_activity_time
        remaining = (self.INACTIVITY_TIMEOUT / 1000) - elapsed
        return max(0, int(remaining))
