"""
OpenAI service for fetching and managing OpenAI models.
"""
import os
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.database_manager import DatabaseManager


class OpenAIService:
    """Service for managing OpenAI models and API interactions."""

    def __init__(self):
        self.db = DatabaseManager()

    def fetch_and_store_models(self, api_key: str) -> tuple[bool, str]:
        """
        Fetch available models from OpenAI API and store them in the database.

        Args:
            api_key: OpenAI API key

        Returns:
            tuple: (success, message)
        """
        if not OPENAI_AVAILABLE:
            return False, "OpenAI library not available. Please install openai."

        if not api_key:
            return False, "OpenAI API key is required."

        try:
            # Initialize OpenAI client
            client = OpenAI(api_key=api_key)

            print("🔄 Fetching available models from OpenAI API...")

            # Fetch models from OpenAI
            models_response = client.models.list()

            # Store models in database
            stored_count = 0
            for model in models_response.data:
                # Store all models (no filtering)
                success = self.db.save_text_model(
                    model_id=model.id,
                    display_name=self._get_display_name(model.id),
                    description=self._get_model_description(model.id),
                    context_window=self._get_context_window(model.id),
                    input_pricing=self._get_input_pricing(model.id),
                    output_pricing=self._get_output_pricing(model.id),
                    is_active=True
                )

                if success:
                    stored_count += 1
                    print(f"✅ Stored model: {model.id}")
                else:
                    print(f"⚠️  Failed to store model: {model.id}")

            return True, f"Successfully fetched and stored {stored_count} models from OpenAI API."

        except Exception as e:
            error_msg = f"Failed to fetch models from OpenAI API: {str(e)}"
            print(f"❌ {error_msg}")
            return False, error_msg

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get all active models from the database.

        Returns:
            List of model dictionaries with keys: model_id, display_name, description, etc.
        """
        return self.db.get_all_text_models(active_only=True)

    def get_model_display_name(self, model_id: str) -> str:
        """
        Get the display name for a model ID.

        Args:
            model_id: OpenAI model ID

        Returns:
            Display name string
        """
        model = self.db.get_text_model(model_id)
        if model:
            return model.get('display_name', model_id)
        return model_id

    def _get_display_name(self, model_id: str) -> str:
        """Get user-friendly display name for a model ID."""
        display_names = {
            'gpt-4o': 'GPT-4o (Latest)',
            'gpt-4o-mini': 'GPT-4o Mini (Fast & Cheap)',
            'gpt-4-turbo': 'GPT-4 Turbo',
            'gpt-4': 'GPT-4',
            'gpt-3.5-turbo': 'GPT-3.5 Turbo (Nano)',
            'gpt-3.5-turbo-16k': 'GPT-3.5 Turbo 16K',
            'gpt-3.5-turbo-0125': 'GPT-3.5 Turbo (Latest)',
            'gpt-4-0125-preview': 'GPT-4 Turbo Preview',
            'gpt-4-1106-preview': 'GPT-4 Turbo (Nov 2023)',
        }

        return display_names.get(model_id, model_id.replace('-', ' ').title())

    def _get_model_description(self, model_id: str) -> Optional[str]:
        """Get description for a model."""
        descriptions = {
            'gpt-4o': 'Most advanced GPT-4 model, optimized for chat and supports vision',
            'gpt-4o-mini': 'Fast and cost-effective GPT-4 level model',
            'gpt-4-turbo': 'Latest GPT-4 Turbo model with improved performance',
            'gpt-4': 'Original GPT-4 model with high reasoning capabilities',
            'gpt-3.5-turbo': 'Fast and cost-effective model for most use cases',
            'gpt-3.5-turbo-16k': 'GPT-3.5 Turbo with 16K context window',
        }

        return descriptions.get(model_id)

    def _get_context_window(self, model_id: str) -> Optional[int]:
        """Get context window size for a model."""
        context_windows = {
            'gpt-4o': 128000,
            'gpt-4o-mini': 128000,
            'gpt-4-turbo': 128000,
            'gpt-4': 8192,
            'gpt-3.5-turbo': 16385,
            'gpt-3.5-turbo-16k': 16385,
            'gpt-3.5-turbo-0125': 16385,
        }

        return context_windows.get(model_id)

    def _get_input_pricing(self, model_id: str) -> Optional[float]:
        """Get input pricing per 1K tokens (in USD)."""
        # Note: These are approximate pricing and may change
        input_pricing = {
            'gpt-4o': 0.005,
            'gpt-4o-mini': 0.00015,
            'gpt-4-turbo': 0.01,
            'gpt-4': 0.03,
            'gpt-3.5-turbo': 0.0005,
            'gpt-3.5-turbo-16k': 0.0005,
        }

        return input_pricing.get(model_id)

    def _get_output_pricing(self, model_id: str) -> Optional[float]:
        """Get output pricing per 1K tokens (in USD)."""
        # Note: These are approximate pricing and may change
        output_pricing = {
            'gpt-4o': 0.015,
            'gpt-4o-mini': 0.0006,
            'gpt-4-turbo': 0.03,
            'gpt-4': 0.06,
            'gpt-3.5-turbo': 0.0015,
            'gpt-3.5-turbo-16k': 0.0015,
        }

        return output_pricing.get(model_id)
